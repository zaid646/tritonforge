#!/usr/bin/env python
"""Sweep OpenVINO performance hints, thread counts, streams and CPU affinity.

The default is ``PERFORMANCE_HINT=LATENCY`` with two inference threads. Each
configuration recompiles the same int4 IR and then runs the same timed greedy
decoding loop as ``scripts/bench_backends.py``.

Usage:
    taskset -c 0,1 python scripts/tune_openvino.py --variant int4_g128 --runs 8
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_backends import TimerStreamer, load_prompts, percentile, run_once  # noqa: E402

CONFIGS = [
    {
        "name": "default_latency_t2_s1",
        "hint": "LATENCY",
        "threads": 2,
        "streams": 1,
        "affinity": [0, 1],
        "default": True,
    },
    {
        "name": "latency_t1_s1",
        "hint": "LATENCY",
        "threads": 1,
        "streams": 1,
        "affinity": [0, 1],
    },
    {
        "name": "throughput_t2_s1",
        "hint": "THROUGHPUT",
        "threads": 2,
        "streams": 1,
        "affinity": [0, 1],
    },
    {
        "name": "throughput_t1_s1",
        "hint": "THROUGHPUT",
        "threads": 1,
        "streams": 1,
        "affinity": [0, 1],
    },
    {
        "name": "throughput_t2_s2",
        "hint": "THROUGHPUT",
        "threads": 2,
        "streams": 2,
        "affinity": [0, 1],
    },
    {
        "name": "throughput_t1_s2",
        "hint": "THROUGHPUT",
        "threads": 1,
        "streams": 2,
        "affinity": [0, 1],
    },
    {
        "name": "latency_t2_s1_affinity0",
        "hint": "LATENCY",
        "threads": 2,
        "streams": 1,
        "affinity": [0],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="int4_g128")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--models-dir", default="/content/models")
    parser.add_argument("--out", default=str(ROOT / "results/x/tuning.json"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    prompts = load_prompts(ROOT / "data/bench_prompts.jsonl", args.batch_size)

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    streamer = TimerStreamer(tokenizer, args.batch_size)

    hw = json.loads((ROOT / "results/x/hardware.json").read_text())["cpu"]["model_name"]
    rows = []
    for config in CONFIGS:
        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, config["affinity"])
        ov_config = {
            "PERFORMANCE_HINT": config["hint"],
            "INFERENCE_NUM_THREADS": str(config["threads"]),
            "NUM_STREAMS": str(config["streams"]),
        }
        print(f"[{config['name']}] compiling", flush=True)
        start = time.perf_counter()
        model = OVModelForCausalLM.from_pretrained(
            str(pathlib.Path(args.models_dir) / f"ov_{args.variant}"), ov_config=ov_config
        )
        for _ in range(args.warmup):
            run_once(model, tokenizer, prompts, args.batch_size, streamer)
        compile_warmup = time.perf_counter() - start

        runs = []
        for _ in range(args.runs):
            runs.append(run_once(model, tokenizer, prompts, args.batch_size, streamer))
        tokens_s = [run["tokens_per_s"] for run in runs]
        ttft = [run["ttft_ms"] for run in runs]
        itl = [run["itl_ms"] for run in runs]
        row = {
            "name": config["name"],
            "hint": config["hint"],
            "threads": config["threads"],
            "streams": config["streams"],
            "affinity": config["affinity"],
            "default": config.get("default", False),
            "runs": args.runs,
            "tokens_per_s_mean": statistics.fmean(tokens_s),
            "tokens_per_s_p95": percentile(tokens_s, 0.95),
            "ttft_ms_mean": statistics.fmean(ttft),
            "itl_ms_mean": statistics.fmean(itl),
            "compile_warmup_seconds": round(compile_warmup, 2),
        }
        rows.append(row)
        print(
            f"[{config['name']}] {row['tokens_per_s_mean']:.2f} tok/s, "
            f"TTFT {row['ttft_ms_mean']:.0f} ms, ITL {row['itl_ms_mean']:.1f} ms",
            flush=True,
        )
        del model
        os.sched_setaffinity(0, original_affinity)

    default = next(row for row in rows if row["default"])
    best = max(rows, key=lambda row: row["tokens_per_s_mean"])
    for row in rows:
        row["gain_vs_default_pct"] = (
            100.0 * (row["tokens_per_s_mean"] - default["tokens_per_s_mean"])
            / default["tokens_per_s_mean"]
        )
    payload = {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "hardware_cpu": hw,
        "note": (
            "Generated by scripts/tune_openvino.py; same prompts and decode loop as "
            "bench_backends.py."
        ),
        "default": default["name"],
        "best": best["name"],
        "best_gain_vs_default_pct": best["gain_vs_default_pct"],
        "configs": rows,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"best {best['name']} {best['tokens_per_s_mean']:.2f} tok/s "
        f"({best['gain_vs_default_pct']:+.1f}% vs default) -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
