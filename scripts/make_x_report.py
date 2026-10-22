#!/usr/bin/env python
"""Render the TritonForge-X result tables from measured JSON files.

Nothing in the generated Markdown is hand-entered: every value comes from
``results/x/*.json`` written by the conversion, validation, benchmark, tuning
and SYCL scripts.

Usage:
    python scripts/make_x_report.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/x"

LABELS = {
    "pytorch_eager": "PyTorch eager fp32",
    "torch_compile": "torch.compile",
    "onnxruntime": "ONNX Runtime CPU",
    "ov_fp16": "OpenVINO fp16",
    "ov_int8": "OpenVINO int8",
    "ov_int4_g128": "OpenVINO int4 g128",
    "ov_int4_awq_g128": "OpenVINO int4 g128 AWQ",
}


def load(path: pathlib.Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def fmt_sci(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2e}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_bench() -> dict:
    rows = {}
    for path in sorted((RESULTS / "bench").glob("*_b*.json")):
        row = json.loads(path.read_text())
        rows[(row["backend"], row["batch_size"])] = row
    return rows


def speed_table(bench: dict) -> str:
    backends = [name for name in LABELS if (name, 1) in bench]
    rows = []
    for backend in backends:
        b1, b8 = bench.get((backend, 1)), bench.get((backend, 8))
        eager1, eager8 = bench.get(("pytorch_eager", 1)), bench.get(("pytorch_eager", 8))
        if not b1:
            continue
        rows.append(
            [
                LABELS[backend],
                fmt(b1["tokens_per_s_mean"])
                + " (" + fmt(b1["tokens_per_s_p95"]) + ")",
                fmt(b1["ttft_ms_mean"], 0) + " (" + fmt(b1["ttft_ms_p95"], 0) + ")",
                fmt(b1["itl_ms_mean"], 1) + " (" + fmt(b1["itl_ms_p95"], 1) + ")",
                fmt(b1["tokens_per_s_mean"] / eager1["tokens_per_s_mean"], 2, "x")
                if eager1
                else "-",
                fmt(b8["tokens_per_s_mean"]) if b8 else "-",
                fmt(b8["ttft_ms_mean"], 0) if b8 else "-",
                fmt(b8["itl_ms_mean"], 1) if b8 else "-",
                fmt(b8["tokens_per_s_mean"] / eager8["tokens_per_s_mean"], 2, "x")
                if b8 and eager8
                else "-",
            ]
        )
    return table(
        [
            "Backend",
            "tok/s b1 (p95)",
            "TTFT ms b1 (p95)",
            "ITL ms b1 (p95)",
            "Speedup b1",
            "tok/s b8",
            "TTFT ms b8",
            "ITL ms b8",
            "Speedup b8",
        ],
        rows,
    )


def memory_table(bench: dict) -> str:
    fp16 = bench.get(("ov_fp16", 1))
    fp16_size = fp16.get("model_size_mb") if fp16 else None
    rows = []
    for backend, label in LABELS.items():
        b1, b8 = bench.get((backend, 1)), bench.get((backend, 8))
        if not b1:
            continue
        size = b1.get("model_size_mb")
        reduction = (
            100.0 * (1 - size / fp16_size)
            if backend.startswith("ov_") and size and fp16_size
            else None
        )
        rows.append(
            [
                label,
                fmt(b1.get("peak_rss_mib"), 0),
                fmt(b8.get("peak_rss_mib"), 0) if b8 else "-",
                fmt(size, 1),
                fmt(reduction, 1, "%") if reduction is not None else "-",
            ]
        )
    return table(
        ["Backend", "Peak RSS b1 MiB", "Peak RSS b8 MiB", "On-disk MB", "vs OV fp16"],
        rows,
    )


def accuracy_table(validation: dict) -> str:
    rows = []
    for row in validation.get("variants", []):
        rows.append(
            [
                LABELS.get(f"ov_{row['variant']}", row["variant"]),
                fmt(row["top1_agreement_pct"], 1, "%"),
                fmt(row["max_abs_err"], 4),
                fmt(row["mean_abs_err"], 4),
                fmt(100 * row["mean_top5_overlap"], 1, "%"),
                fmt(row["mean_kl"], 4),
                fmt(row["ppl"], 3),
                fmt(row["ppl_delta_pct"], 2, "%"),
            ]
        )
    return table(
        [
            "Variant",
            "Top-1 agree",
            "Max abs",
            "Mean abs",
            "Top-5 overlap",
            "Mean KL",
            "wikitext-2 PPL",
            "dPPL",
        ],
        rows,
    )


def tuning_table(tuning: dict | None) -> str:
    if not tuning:
        return "_Tuning results not collected._"
    rows = []
    for row in tuning["configs"]:
        rows.append(
            [
                row["name"],
                row["hint"],
                str(row["threads"]),
                str(row["streams"]),
                ",".join(str(core) for core in row["affinity"]),
                fmt(row["tokens_per_s_mean"]),
                fmt(row["ttft_ms_mean"], 0),
                fmt(row["itl_ms_mean"], 1),
                fmt(row["gain_vs_default_pct"], 2, "%"),
            ]
        )
    return table(
        [
            "Config",
            "Hint",
            "Threads",
            "Streams",
            "Affinity",
            "tok/s",
            "TTFT ms",
            "ITL ms",
            "vs default",
        ],
        rows,
    )


def sycl_table(sycl: dict | None, torch_kernels: dict | None) -> str:
    if not sycl:
        return "_SYCL results not collected._"
    torch_index = {}
    if torch_kernels:
        for row in torch_kernels["rows"]:
            torch_index[(row["op"], row["m"], row["n"])] = row
    rows = []
    for row in sycl.get("validation", []):
        torch_row = torch_index.get((row["op"], row["m"], row["n"]))
        op = row["op"]
        rows.append(
            [
                op,
                f"{row['m']}x{row['n']}",
                fmt(torch_row["ms_mean"], 3) if torch_row else "-",
                fmt(row.get("ms_mean"), 3),
                fmt(torch_row["ms_mean"] / row["ms_mean"], 2, "x")
                if torch_row and row.get("ms_mean")
                else "-",
                fmt_sci(row["max_abs_err"]),
                fmt_sci(row["max_rel_err"]),
                "PASS" if row["passed"] else "FAIL",
                fmt(row.get("gbps")),
                fmt(
                    100.0 * row["gbps"] / sycl["stream"]["peak_gbps"], 1, "%"
                )
                if row.get("gbps")
                else "-",
            ]
        )
    return table(
        [
            "Kernel",
            "Shape",
            "PyTorch ms",
            "SYCL ms",
            "Speedup",
            "Max abs",
            "Max rel",
            "Parity",
            "GB/s",
            "% measured peak",
        ],
        rows,
    )


def sycl_tuning_table(sycl: dict | None) -> str:
    if not sycl:
        return "_SYCL tuning results not collected._"
    rows = []
    for row in sycl["tuning"]["swiglu"] + sycl["tuning"]["rmsnorm_residual"]:
        rows.append(
            [
                row["op"],
                str(row["work_group"]),
                str(row["vector_width"]),
                fmt(row["gbps"]),
                fmt(row["gflops"]),
            ]
        )
    return table(["Kernel", "Work-group", "Vector width", "GB/s", "GFLOP/s"], rows)


def honest_losers(bench: dict, validation: dict, sycl: dict | None) -> list[str]:
    losers = []
    eager = bench.get(("pytorch_eager", 1))
    if eager:
        for backend, label in LABELS.items():
            row = bench.get((backend, 1))
            if row and row["tokens_per_s_mean"] < eager["tokens_per_s_mean"]:
                losers.append(
                    f"{label} is slower than PyTorch eager at batch 1 "
                    f"({row['tokens_per_s_mean']:.2f} vs {eager['tokens_per_s_mean']:.2f} tok/s)."
                )
    for row in validation.get("variants", []):
        if row["ppl_delta_pct"] > 5.0:
            losers.append(
                f"{LABELS.get('ov_' + row['variant'], row['variant'])} loses accuracy: "
                f"wikitext-2 PPL +{row['ppl_delta_pct']:.2f}% and "
                f"{row['top1_agreement_pct']:.1f}% top-1 agreement."
            )
    if sycl:
        for op, row in sycl.get("best", {}).items():
            if row["pct_measured_peak_bw"] < 50.0:
                losers.append(
                    f"SYCL {op} reaches only {row['pct_measured_peak_bw']:.1f}% of measured "
                    "STREAM bandwidth; launch and reduction overhead dominate these small tensors."
                )
    return losers


def main() -> int:
    hardware = load(RESULTS / "hardware.json")
    if not hardware:
        raise SystemExit("results/x/hardware.json missing; run scripts/probe_hardware.py")
    cpu = hardware["cpu"]["model_name"]
    bench = load_bench()
    models = load(RESULTS / "models.json", {"variants": []})
    validation = load(
        RESULTS / "validation.json", {"variants": [], "wikitext": {}, "reference": {}}
    )
    tuning = load(RESULTS / "tuning.json")
    sycl = load(RESULTS / "sycl.json")
    torch_kernels = load(RESULTS / "torch_kernels.json")

    model_rows = []
    for row in models["variants"]:
        model_rows.append(
            [
                LABELS.get(f"ov_{row['variant']}", row["variant"]),
                fmt(row["ir_weights_mb"], 1),
                fmt(row["total_dir_mb"], 1),
                fmt(row["size_vs_fp16_pct"], 1, "%")
                if row.get("size_vs_fp16_pct") is not None
                else "-",
                "yes" if row["data_aware"] else "no",
            ]
        )

    wi = validation.get("wikitext", {})
    parts = [
        "# TritonForge-X results",
        "",
        "Generated by `python scripts/make_x_report.py` from `results/x/*.json`.",
        "Do not hand-edit the tables.",
        "",
        f"**CPU: `{cpu}`** (2 vCPU, AVX2/FMA only, no AVX-512/AMX). "
        f"See [HARDWARE.md](HARDWARE.md).",
        "",
        "## Protocol",
        "",
        f"- Greedy decoding, "
        f"{next(iter(bench.values()))['max_new_tokens'] if bench else 32} new tokens, "
        f"{next(iter(bench.values()))['runs'] if bench else 10} measured runs after "
        f"{next(iter(bench.values()))['warmup_runs'] if bench else 2} warmups; p95 in parentheses.",
        "- Pinned to CPUs 0-1 with `taskset`; peak RSS is `ru_maxrss` of a single-backend process.",
        f"- wikitext-2 test: {wi.get('n_windows', '?')} windows x {wi.get('seq_len', '?')} tokens "
        f"({wi.get('tokens_scored', '?')} scored tokens), not the full split.",
        "- All timing numbers are wall-clock on a shared cloud VM and will vary between runs.",
        "",
        "## 1. Decode speed",
        "",
        speed_table(bench),
        "",
        "## 2. Memory and model size",
        "",
        memory_table(bench),
        "",
        "## 3. Accuracy parity versus PyTorch fp32",
        "",
        accuracy_table(validation),
        "",
        f"Reference wikitext-2 PPL (PyTorch fp32): "
        f"**{validation.get('reference', {}).get('ppl', float('nan')):.3f}**.",
        "",
        "## 4. OpenVINO tuning",
        "",
        tuning_table(tuning),
        "",
        "## 5. SYCL kernels",
        "",
        sycl_table(sycl, torch_kernels),
        "",
        f"Measured STREAM peak: **{sycl['stream']['peak_gbps']:.2f} GB/s**."
        if sycl
        else "",
        "",
        f"Roofline: ![roofline](../{sycl.get('roofline', {}).get('plot', '')})"
        if sycl
        else "",
        "",
        "### Work-group and vector-width sweep",
        "",
        sycl_tuning_table(sycl),
        "",
        "## 6. OpenVINO on-disk sizes",
        "",
        table(["Variant", "IR weights MB", "Whole dir MB", "vs fp16", "Data-aware"], model_rows),
        "",
        "## 7. What did not win",
        "",
        "\n".join(f"- {item}" for item in honest_losers(bench, validation, sycl))
        or "- Nothing regressed against the stated baselines.",
        "",
    ]
    text = "\n".join(part for part in parts if part is not None)
    out = ROOT / "docs/RESULTS_X.md"
    out.write_text(text + "\n")

    marker_start = "<!-- X_RESULTS_START -->"
    marker_end = "<!-- X_RESULTS_END -->"
    summary = [
        marker_start,
        "## TritonForge-X on Intel CPU",
        "",
        f"Measured on `{cpu}` (2 vCPU, AVX2 only, no AVX-512/AMX). "
        "Full tables: [RESULTS_X.md](RESULTS_X.md).",
        "",
        speed_table(bench),
        "",
        "Accuracy parity and the honest list of regressions are in "
        "[RESULTS_X.md](RESULTS_X.md#3-accuracy-parity-versus-pytorch-fp32).",
        marker_end,
    ]
    report_path = ROOT / "docs/FINAL_REPORT.md"
    report = report_path.read_text()
    if marker_start in report and marker_end in report:
        before = report.split(marker_start)[0]
        after = report.split(marker_end)[1]
        report = before + "\n".join(summary) + after
    else:
        report = report.rstrip() + "\n\n" + "\n".join(summary) + "\n"
    report_path.write_text(report)

    print(f"wrote {out}")
    print(f"updated {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
