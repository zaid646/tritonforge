#!/usr/bin/env python
"""Fail CI when a committed TritonForge-X metric regresses past its baseline.

The gate reads the same JSON files that produce the report, so it is cheap in
CI (no model download, no benchmark rerun). Thresholds live in
``baselines/x_baseline.json`` and are intentionally set slightly below the
measured Colab values.

Usage:
    python scripts/check_regression_x.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/x"
BASELINE = ROOT / "baselines/x_baseline.json"


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    baseline = load(BASELINE)
    if not baseline:
        print(f"missing baseline: {BASELINE}")
        return 2
    hardware = load(RESULTS / "hardware.json")
    validation = load(RESULTS / "validation.json")
    sycl = load(RESULTS / "sycl.json")
    bench = {}
    for path in (RESULTS / "bench").glob("*_b*.json"):
        row = json.loads(path.read_text())
        bench[(row["backend"], row["batch_size"])] = row

    failures = []
    checks = []

    def check(condition: bool, label: str, detail: str) -> None:
        checks.append((condition, label, detail))
        if not condition:
            failures.append(f"{label}: {detail}")

    expected_cpu = baseline.get("hardware_cpu")
    check(
        bool(hardware) and hardware["cpu"]["model_name"] == expected_cpu,
        "hardware",
        f"expected {expected_cpu!r}, got {hardware['cpu']['model_name'] if hardware else None!r}",
    )

    for backend, batches in baseline.get("bench", {}).items():
        for batch_key, rule in batches.items():
            batch = int(batch_key.removeprefix("batch_"))
            row = bench.get((backend, batch))
            if row is None:
                check(False, f"bench {backend} b{batch}", "missing result")
                continue
            if "min_tokens_per_s" in rule:
                check(
                    row["tokens_per_s_mean"] >= rule["min_tokens_per_s"],
                    f"bench {backend} b{batch} tok/s",
                    f"{row['tokens_per_s_mean']:.2f} >= {rule['min_tokens_per_s']}",
                )
            if "max_ttft_ms" in rule:
                check(
                    row["ttft_ms_mean"] <= rule["max_ttft_ms"],
                    f"bench {backend} b{batch} TTFT",
                    f"{row['ttft_ms_mean']:.1f} <= {rule['max_ttft_ms']}",
                )

    variants = {row["variant"]: row for row in (validation or {}).get("variants", [])}
    for variant, rule in baseline.get("accuracy", {}).items():
        row = variants.get(variant)
        if row is None:
            check(False, f"accuracy {variant}", "missing result")
            continue
        if "min_top1_pct" in rule:
            check(
                row["top1_agreement_pct"] >= rule["min_top1_pct"],
                f"accuracy {variant} top1",
                f"{row['top1_agreement_pct']:.1f} >= {rule['min_top1_pct']}",
            )
        if "max_ppl_delta_pct" in rule:
            check(
                row["ppl_delta_pct"] <= rule["max_ppl_delta_pct"],
                f"accuracy {variant} PPL delta",
                f"{row['ppl_delta_pct']:+.2f} <= {rule['max_ppl_delta_pct']}",
            )

    if sycl:
        for op, rule in baseline.get("sycl", {}).items():
            best = sycl.get("best", {}).get(op)
            if best is None:
                check(False, f"sycl {op}", "missing result")
                continue
            if "min_pct_peak_bw" in rule:
                check(
                    best["pct_measured_peak_bw"] >= rule["min_pct_peak_bw"],
                    f"sycl {op} % peak BW",
                    f"{best['pct_measured_peak_bw']:.1f} >= {rule['min_pct_peak_bw']}",
                )
            for row in sycl.get("validation", []):
                if row["op"] == op and "max_abs_err" in rule:
                    check(
                        row["max_abs_err"] <= rule["max_abs_err"],
                        f"sycl {op} max abs",
                        f"{row['max_abs_err']:.2e} <= {rule['max_abs_err']:.2e}",
                    )

    for ok, label, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label:34s} {detail}")
    if failures:
        print(f"\n{len(failures)} regression(s) detected", file=sys.stderr)
        return 1
    print(f"\nall {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
