#!/usr/bin/env python
"""Measure TTFT, inter-token latency, throughput and peak RSS for one backend.

One backend is measured per process so that peak RSS is attributable. The
orchestrator is ``scripts/run_x_benchmarks.sh``; every row is written to
``results/x/bench/<backend>_b<batch>.json`` and later merged into the report.

Usage:
    taskset -c 0,1 python scripts/bench_backends.py \
        --backend ov_int4_g128 --batch-size 1 --runs 10
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import resource
import statistics
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 2


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def load_prompts(path: pathlib.Path, count: int) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row["prompt"] for row in rows[:count]]


def dir_size_mb(path: pathlib.Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6, 1)


def model_size_mb(backend: str, models_dir: pathlib.Path) -> float | None:
    if backend.startswith("ov_"):
        path = models_dir / f"ov_{backend[3:]}"
    elif backend == "onnxruntime":
        path = models_dir / "onnx"
    elif backend in {"pytorch_eager", "torch_compile"}:
        from huggingface_hub import snapshot_download

        path = pathlib.Path(snapshot_download(MODEL_ID, local_files_only=True))
    else:
        return None
    return dir_size_mb(path) if path.exists() else None


def build_model(backend: str, models_dir: pathlib.Path):
    if backend in {"pytorch_eager", "torch_compile"}:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        model.eval()
        if backend == "torch_compile":
            model = torch.compile(model, mode="default", dynamic=True)
        return model, {}
    if backend == "onnxruntime":
        from optimum.onnxruntime import ORTModelForCausalLM

        model = ORTModelForCausalLM.from_pretrained(
            str(models_dir / "onnx"),
            provider="CPUExecutionProvider",
            use_cache=True,
        )
        return model, {}
    if backend.startswith("ov_"):
        from optimum.intel import OVModelForCausalLM

        name = backend[3:]
        ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "2",
            "NUM_STREAMS": "1",
        }
        model = OVModelForCausalLM.from_pretrained(
            str(models_dir / f"ov_{name}"), ov_config=ov_config
        )
        return model, ov_config
    raise SystemExit(f"unknown backend {backend}")


class TimerStreamer:
    """StoppingCriteria that timestamps each decode step for any batch size."""

    def __init__(self, tokenizer, batch_size: int):
        self.batch_size = batch_size
        self.times: list[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        self.times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def end(self):
        pass


def run_once(model, tokenizer, prompts: list[str], batch_size: int, streamer) -> dict:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=batch_size > 1,
        truncation=False,
    )
    streamer.times = []
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            stopping_criteria=[streamer],
            pad_token_id=tokenizer.pad_token_id,
        )
    end = time.perf_counter()
    generated = out.shape[1] - enc["input_ids"].shape[1]
    total_tokens = generated * batch_size
    if len(streamer.times) < 2:
        raise RuntimeError(f"streamer saw {len(streamer.times)} steps")
    ttft_ms = (streamer.times[0] - start) * 1000.0
    gaps = [
        (streamer.times[i + 1] - streamer.times[i]) * 1000.0
        for i in range(len(streamer.times) - 1)
    ]
    itl_ms = sum(gaps) / len(gaps)
    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "total_ms": (end - start) * 1000.0,
        "generated_tokens": int(total_tokens),
        "tokens_per_s": total_tokens / (end - start),
        "prompt_tokens_mean": float(enc["input_ids"].shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--models-dir", default="/content/models")
    parser.add_argument("--out-dir", default=str(ROOT / "results/x/bench"))
    parser.add_argument("--prompts", default=str(ROOT / "data/bench_prompts.jsonl"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    models_dir = pathlib.Path(args.models_dir)
    prompts = load_prompts(pathlib.Path(args.prompts), args.batch_size)
    if len(prompts) < args.batch_size:
        raise SystemExit("not enough prompts for the requested batch size")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{args.backend}] loading", flush=True)
    load_start = time.perf_counter()
    model, ov_config = build_model(args.backend, models_dir)
    load_seconds = time.perf_counter() - load_start
    streamer = TimerStreamer(tokenizer, args.batch_size)

    for index in range(args.warmup):
        print(f"[{args.backend}] warmup {index + 1}/{args.warmup}", flush=True)
        run_once(model, tokenizer, prompts, args.batch_size, streamer)
    setup_seconds = time.perf_counter() - load_start

    runs = []
    for index in range(args.runs):
        metrics = run_once(model, tokenizer, prompts, args.batch_size, streamer)
        runs.append(metrics)
        print(
            f"[{args.backend}] run {index + 1}/{args.runs}: "
            f"{metrics['tokens_per_s']:.2f} tok/s, TTFT {metrics['ttft_ms']:.0f} ms",
            flush=True,
        )

    def col(key: str) -> list[float]:
        return [run[key] for run in runs]

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    row = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "warmup_runs": args.warmup,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hardware_cpu": json.loads((ROOT / "results/x/hardware.json").read_text())["cpu"][
            "model_name"
        ],
        "ttft_ms_mean": statistics.fmean(col("ttft_ms")),
        "ttft_ms_p95": percentile(col("ttft_ms"), 0.95),
        "itl_ms_mean": statistics.fmean(col("itl_ms")),
        "itl_ms_p95": percentile(col("itl_ms"), 0.95),
        "tokens_per_s_mean": statistics.fmean(col("tokens_per_s")),
        "tokens_per_s_p95": percentile(col("tokens_per_s"), 0.95),
        "tokens_per_s_stdev": statistics.stdev(col("tokens_per_s")) if len(runs) > 1 else 0.0,
        "peak_rss_mib": round(rss_mb, 1),
        "load_seconds": round(load_seconds, 2),
        "setup_including_warmup_seconds": round(setup_seconds, 2),
        "model_size_mb": model_size_mb(args.backend, models_dir),
        "ov_config": ov_config,
        "runs_detail": runs,
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.backend}_b{args.batch_size}.json"
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"[{args.backend} b{args.batch_size}] "
        f"{row['tokens_per_s_mean']:.2f} tok/s, TTFT {row['ttft_ms_mean']:.0f} ms, "
        f"ITL {row['itl_ms_mean']:.1f} ms, RSS {row['peak_rss_mib']:.0f} MiB -> {out}"
    )
    del model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
