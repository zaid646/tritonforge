#!/usr/bin/env python
"""Fail CI when a kernel or end-to-end metric regresses past the baseline.

Usage:
    python scripts/check_regression.py \
        --results results/kernels.json \
        --baseline baselines/kernel_baseline.json \
        --e2e results/e2e.json

Exit code 0 = all rules pass, 1 = at least one regression, 2 = bad input.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

OK = "PASS"
BAD = "FAIL"


def load(path: str) -> dict | None:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def index_rows(results: dict) -> dict[tuple, dict]:
    idx = {}
    for row in results.get("rows", []):
        idx[(row["kernel"], row["m"], row["n"], row["k"])] = row
    return idx


def check_kernels(results: dict, baseline: dict) -> list[tuple[str, str, str]]:
    rows = index_rows(results)
    checks: list[tuple[str, str, str]] = []
    for rule in baseline.get("rules", []):
        key = (rule["kernel"], rule["m"], rule["n"], rule["k"])
        label = f"{rule['kernel']} m={rule['m']} n={rule['n']} k={rule['k']}"
        row = rows.get(key)
        if row is None:
            checks.append((BAD, label, "missing from results"))
            continue

        failures = []
        if "min_speedup" in rule and row["speedup_vs_eager"] < rule["min_speedup"]:
            failures.append(
                f"speedup {row['speedup_vs_eager']:.2f}x < {rule['min_speedup']:.2f}x"
            )
        if "max_abs_err" in rule and row["max_abs_err"] > rule["max_abs_err"]:
            failures.append(f"max_abs_err {row['max_abs_err']:.4f} > {rule['max_abs_err']}")
        if "max_ms" in rule and row["tuned_ms"] > rule["max_ms"]:
            failures.append(f"tuned_ms {row['tuned_ms']:.4f} > {rule['max_ms']}")
        if (
            "min_pct_peak" in rule
            and row.get("pct_peak_bw") is not None
            and row["pct_peak_bw"] < rule["min_pct_peak"]
        ):
            failures.append(f"pct_peak_bw {row['pct_peak_bw']:.1f}% < {rule['min_pct_peak']}%")

        if failures:
            checks.append((BAD, label, "; ".join(failures)))
        else:
            detail = (
                f"{row['speedup_vs_eager']:.2f}x eager, "
                f"err {row['max_abs_err']:.4f}, {row['tuned_ms']:.4f} ms"
            )
            checks.append((OK, label, detail))
    return checks


def check_e2e(e2e: dict, baseline: dict) -> list[tuple[str, str, str]]:
    rules = baseline.get("e2e", {})
    checks: list[tuple[str, str, str]] = []
    results = e2e.get("results", [])
    tf = next(
        (r for r in results if r.get("name") == "tritonforge" and r.get("status") == "ok"), None
    )
    if tf is None:
        checks.append((BAD, "e2e tritonforge", "missing or errored"))
        return checks

    if "max_ppl_delta_vs_fp16" in rules:
        delta = tf.get("ppl_delta_vs_fp16")
        if delta is None or delta > rules["max_ppl_delta_vs_fp16"]:
            checks.append((BAD, "e2e ppl delta", f"{delta} > {rules['max_ppl_delta_vs_fp16']}"))
        else:
            checks.append((OK, "e2e ppl delta", f"+{delta:.3f} vs fp16"))
    if "min_decode_tokens_per_s" in rules:
        tps = tf.get("decode_tokens_per_s")
        if tps is None or tps < rules["min_decode_tokens_per_s"]:
            checks.append((BAD, "e2e decode tok/s", f"{tps} < {rules['min_decode_tokens_per_s']}"))
        else:
            checks.append((OK, "e2e decode tok/s", f"{tps:.1f} tok/s"))
    if "max_peak_vram_gb" in rules:
        vram = tf.get("peak_vram_gb")
        if vram is None or vram > rules["max_peak_vram_gb"]:
            checks.append((BAD, "e2e peak vram", f"{vram} > {rules['max_peak_vram_gb']}"))
        else:
            checks.append((OK, "e2e peak vram", f"{vram:.2f} GB"))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/kernels.json")
    ap.add_argument("--baseline", default="baselines/kernel_baseline.json")
    ap.add_argument("--e2e", default="results/e2e.json")
    ap.add_argument("--no-e2e", action="store_true")
    args = ap.parse_args()

    baseline = load(args.baseline)
    results = load(args.results)
    if baseline is None or results is None:
        print("ERROR: missing baseline or results file", file=sys.stderr)
        return 2

    checks = check_kernels(results, baseline)
    if not args.no_e2e:
        e2e = load(args.e2e)
        if e2e is not None:
            checks += check_e2e(e2e, baseline)

    width = max(len(label) for _, label, _ in checks)
    n_fail = 0
    for status, label, detail in checks:
        print(f"[{status}] {label.ljust(width)}  {detail}")
        n_fail += status == BAD
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
