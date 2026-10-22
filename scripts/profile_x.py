#!/usr/bin/env python
"""Collect CPU profiles: torch profiler for eager decode, OpenVINO per-layer counters.

Outputs:
  results/x/profiles/torch_eager_trace.json  (Chrome trace)
  results/x/profiles/ov_report.csv           (benchmark_app detailed counters)
  results/x/profiling.json                   (top hotspots, parsed)

Usage:
    python scripts/profile_x.py --tokens 8
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def torch_profile(tokens: int, out_dir: pathlib.Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    enc = tokenizer("Explain how a CPU cache improves performance.", return_tensors="pt")

    with torch.no_grad():
        model.generate(**enc, max_new_tokens=2, do_sample=False)

    trace = out_dir / "torch_eager_trace.json"
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad(), record_function("decode"):
            model.generate(**enc, max_new_tokens=tokens, do_sample=False)
    prof.export_chrome_trace(str(trace))

    rows = []
    for event in prof.key_averages():
        if event.self_cpu_time_total <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "self_cpu_ms": event.self_cpu_time_total / 1000.0,
                "cpu_total_ms": event.cpu_time_total / 1000.0,
                "calls": event.count,
            }
        )
    rows.sort(key=lambda row: row["self_cpu_ms"], reverse=True)
    total = sum(row["self_cpu_ms"] for row in rows)
    for row in rows:
        row["self_pct"] = 100.0 * row["self_cpu_ms"] / total if total else 0.0
    return {"trace": str(trace), "total_cpu_ms": total, "top": rows[:20]}


def ov_benchmark_app(models_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    app = shutil.which("benchmark_app")
    model = models_dir / "ov_fp16/openvino_model.xml"
    if not app or not model.exists():
        return {"skipped": "benchmark_app or model missing"}
    report_dir = out_dir / "ov_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    beam_path = report_dir / "beam_idx.bin"
    import numpy as np

    np.zeros(1, dtype=np.int32).tofile(beam_path)
    command = [
        app,
        "-m",
        str(model),
        "-d",
        "CPU",
        "-niter",
        "3",
        "-nireq",
        "1",
        "-hint",
        "none",
        "-nthreads",
        "2",
        "-report_type",
        "detailed_counters",
        "-report_folder",
        str(report_dir),
        "-data_shape",
        "input_ids[1,32],attention_mask[1,32],position_ids[1,32],beam_idx[1]",
        "-i",
        f"beam_idx:{beam_path}",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    (out_dir / "benchmark_app.log").write_text(proc.stdout + "\n" + proc.stderr)
    csv_files = sorted(report_dir.glob("*detailed_counters*.csv"))
    rows = []
    used = None
    if csv_files:
        used = str(csv_files[0])
        with open(csv_files[0], newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                if (row.get("layerName") or "").strip().lower() == "total":
                    continue
                time_key = next(
                    (key for key in row if key and "realtime" in key.lower().replace(" ", "")),
                    None,
                )
                if not time_key:
                    continue
                try:
                    value = float(row[time_key]) * 1000.0
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "layer": row.get("layerName") or row.get("layerType") or "?",
                        "layer_type": row.get("layerType", ""),
                        "exec_type": row.get("execType", ""),
                        "real_time_us": value,
                    }
                )
    rows.sort(key=lambda row: row["real_time_us"], reverse=True)
    total = sum(row["real_time_us"] for row in rows)
    for row in rows:
        row["pct"] = 100.0 * row["real_time_us"] / total if total else 0.0
    return {
        "returncode": proc.returncode,
        "report_csv": used,
        "total_us": total,
        "top": rows[:20],
        "stderr_tail": proc.stderr[-1000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--models-dir", default="/content/models")
    parser.add_argument("--out", default=str(ROOT / "results/x/profiling.json"))
    args = parser.parse_args()

    out_dir = ROOT / "results/x/profiles"
    hw = json.loads((ROOT / "results/x/hardware.json").read_text())["cpu"]["model_name"]
    torch_rows = torch_profile(args.tokens, out_dir)
    ov_rows = ov_benchmark_app(pathlib.Path(args.models_dir), out_dir)

    payload = {
        "hardware_cpu": hw,
        "note": "Generated by scripts/profile_x.py.",
        "torch_eager": torch_rows,
        "openvino_benchmark_app": ov_rows,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"torch top: {torch_rows['top'][0]['name']} ({torch_rows['top'][0]['self_pct']:.1f}%)")
    if ov_rows.get("top"):
        print(f"OV top: {ov_rows['top'][0]['layer']} ({ov_rows['top'][0]['pct']:.1f}%)")
    else:
        reason = ov_rows.get("skipped") or ov_rows.get("stderr_tail", "")[-200:]
        print(f"OV counters skipped: {reason}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
