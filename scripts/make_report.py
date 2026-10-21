#!/usr/bin/env python
"""Generate the static, accessible HTML report from committed results.

    PYTHONPATH=src python scripts/make_report.py

Reads results/kernels.json, results/e2e.json and artifacts/sanitizer/*.log and
writes reports/index.html (self-contained, no JS, inline SVG charts with
accessible titles/descriptions).
"""

from __future__ import annotations

import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
SAN_DIR = ROOT / "artifacts" / "sanitizer"

PALETTE = {
    "dequant_gemm": "#1f6feb",
    "rmsnorm_residual": "#2ea043",
    "swiglu": "#9a6700",
    "accent": "#6f42c1",
}


def load_json(path: pathlib.Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def bar_chart(
    title: str,
    desc: str,
    rows: list[tuple[str, float]],
    color: str,
    unit: str,
    max_value: float | None = None,
) -> str:
    if not rows:
        return "<p>No data.</p>"
    hi = max_value or max(v for _, v in rows) or 1.0
    bar_h, gap, label_w, width = 22, 10, 210, 720
    track_w = width - label_w - 70
    height = len(rows) * (bar_h + gap) + 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-labelledby="t-{abs(hash(title)) % 100000}" class="chart">',
        f'<title id="t-{abs(hash(title)) % 100000}">{esc(title)}</title>',
        f"<desc>{esc(desc)}</desc>",
        f'<line x1="{label_w}" y1="0" x2="{label_w}" y2="{height - 10}" stroke="#d0d7de"/>',
    ]
    for i, (label, value) in enumerate(rows):
        y = i * (bar_h + gap) + 5
        w = max(1.0, track_w * value / hi)
        text = f"{value:.2f}{unit}" if unit == "x" else f"{value:.0f}{unit}"
        parts.append(
            f'<text x="0" y="{y + bar_h * 0.7}" font-size="12" fill="#1f2328">{esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}" '
            f'rx="3"><title>{esc(label)}: {esc(text)}</title></rect>'
        )
        parts.append(
            f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h * 0.7}" font-size="12" '
            f'fill="#1f2328">{esc(text)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def kernel_rows(kernels: dict) -> list[dict]:
    return kernels.get("rows", [])


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def sanitizer_summary() -> tuple[str, str]:
    if not SAN_DIR.exists():
        return "not run", "Sanitizer logs were not found."
    parts = []
    ok = True
    for tool in ("memcheck", "racecheck", "synccheck"):
        log = SAN_DIR / f"{tool}.log"
        if not log.exists():
            parts.append(f"{tool}: missing")
            ok = False
            continue
        text = log.read_text()
        m = re.search(r"(ERROR SUMMARY: (\d+) errors|RACECHECK SUMMARY: (\d+) hazards)", text)
        if "RACECHECK" in text:
            m2 = re.search(r"(\d+) errors, (\d+) warnings", text)
            detail = f"{m2.group(1)} errors, {m2.group(2)} warnings" if m2 else "see log"
        elif m:
            detail = f"{m.group(2)} errors"
        else:
            detail = "see log"
        if "0 errors" not in text and "0 errors, 0 warnings" not in text:
            ok = False
        parts.append(f"{tool}: {detail}")
    return ("clean" if ok else "issues"), "; ".join(parts)


def build_html(kernels: dict | None, e2e: dict | None) -> str:
    san_status, san_detail = sanitizer_summary()

    device = kernels["device"] if kernels else {}
    dtype = kernels.get("dtype", "n/a") if kernels else "n/a"

    dq = [r for r in kernel_rows(kernels) if r["kernel"] == "dequant_gemm"] if kernels else []
    rn = [r for r in kernel_rows(kernels) if r["kernel"] == "rmsnorm_residual"] if kernels else []
    sg = [r for r in kernel_rows(kernels) if r["kernel"] == "swiglu"] if kernels else []

    best_dq = max((r["speedup_vs_eager"] for r in dq), default=None)
    best_sg = max((r["speedup_vs_eager"] for r in sg), default=None)
    best_bw = max((r["pct_peak_bw"] or 0 for r in rn + sg), default=None)
    max_err = max((r["max_abs_err"] for r in dq), default=None)

    charts = []
    if dq:
        rows = [(f"M={r['m']} N={r['n']} K={r['k']}", r["speedup_vs_eager"]) for r in dq]
        charts.append(
            bar_chart(
                "int4 fused dequant GEMM speedup vs eager (T4)",
                "Bar chart of fused kernel latency relative to eager dequantize+matmul.",
                rows,
                PALETTE["dequant_gemm"],
                "x",
            )
        )
    if rn:
        rows = [(f"M={r['m']} N={r['n']}", r["speedup_vs_eager"]) for r in rn]
        charts.append(
            bar_chart(
                "Fused RMSNorm+residual speedup vs eager (T4)",
                "Bar chart of fused RMSNorm plus residual-add kernel speedup.",
                rows,
                PALETTE["rmsnorm_residual"],
                "x",
            )
        )
    if sg:
        rows = [(f"M={r['m']} N={r['n']}", r["speedup_vs_eager"]) for r in sg]
        charts.append(
            bar_chart(
                "Fused SwiGLU speedup vs eager (T4)",
                "Bar chart of fused SwiGLU kernel speedup.",
                rows,
                PALETTE["swiglu"],
                "x",
            )
        )
    bw_rows = [(f"{r['kernel']} M={r['m']}", r["pct_peak_bw"]) for r in rn + sg if r["pct_peak_bw"]]
    if bw_rows:
        charts.append(
            bar_chart(
                "Achieved memory bandwidth as percent of T4 peak",
                "Bar chart of achieved bandwidth relative to the 320 GB/s T4 peak.",
                bw_rows,
                PALETTE["accent"],
                "%",
                max_value=100.0,
            )
        )

    dq_table = table(
        ["M", "N", "K", "eager ms", "default ms", "tuned ms", "speedup", "autotune gain",
         "max abs err", "GB/s", "% peak"],
        [
            [
                r["m"], r["n"], r["k"], fmt(r["eager_ms"], 4), fmt(r["default_ms"], 4),
                fmt(r["tuned_ms"], 4), f"{r['speedup_vs_eager']:.2f}x",
                f"{r['autotune_gain']:.2f}x", fmt(r["max_abs_err"], 4),
                fmt(r["achieved_bw_gbps"], 1), fmt(r["pct_peak_bw"], 1),
            ]
            for r in dq
        ],
        "int4 fused dequant GEMM: latency versus eager baseline, "
        "autotune gain, error and bandwidth",
    )
    other_table = table(
        ["kernel", "M", "N", "eager ms", "tuned ms", "speedup", "autotune gain", "max abs err",
         "GB/s", "% peak"],
        [
            [
                r["kernel"], r["m"], r["n"], fmt(r["eager_ms"], 4), fmt(r["tuned_ms"], 4),
                f"{r['speedup_vs_eager']:.2f}x", f"{r['autotune_gain']:.2f}x",
                fmt(r["max_abs_err"], 4), fmt(r["achieved_bw_gbps"], 1), fmt(r["pct_peak_bw"], 1),
            ]
            for r in rn + sg
        ],
        "Fused RMSNorm plus residual and fused SwiGLU: latency, speedup and bandwidth",
    )

    e2e_html = "<p>End-to-end results are not available.</p>"
    if e2e:
        rows = []
        for r in e2e["results"]:
            if r["status"] != "ok":
                rows.append([r["name"], "error", "-", "-", "-", "-", esc(r.get("error", ""))[:60]])
                continue
            rows.append([
                r["name"], "ok", fmt(r["ttft_ms"], 1), fmt(r["itl_ms_mean"], 2),
                fmt(r["decode_tokens_per_s"], 1), fmt(r["perplexity"], 3),
                fmt(r["peak_vram_gb"], 2),
            ])
        e2e_html = table(
            ["variant", "status", "TTFT ms", "ITL ms", "decode tok/s", "perplexity",
             "peak VRAM GB"],
            rows,
            f"Qwen2.5-0.5B end-to-end inference on T4 ({e2e.get('max_new_tokens')} new tokens)",
        )

    generated = []
    if kernels:
        generated.append(
            f"kernel results captured on {esc(device.get('name'))} (cc {esc(device.get('cc'))}), "
            f"dtype {esc(dtype)}, group size {kernels.get('group_size')}"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TritonForge: custom Triton kernels for quantized LLM inference</title>
<meta name="description" content="Benchmark results for TritonForge: custom Triton kernels
(int4 fused dequant GEMM, fused RMSNorm+residual, fused SwiGLU) for quantized
LLM inference on an NVIDIA T4.">
<link rel="icon" href="data:,">
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif;
         margin: 0; color: #1f2328; background: #ffffff; line-height: 1.55; }}
  a.skip {{ position: absolute; left: -999px; }}
  a.skip:focus {{ left: 8px; top: 8px; background: #1f2328; color: #fff;
                 padding: 8px 12px; z-index: 10; }}
  header, main {{ max-width: 960px; margin: 0 auto; padding: 0 20px; }}
  header {{ padding-top: 32px; }}
  h1 {{ font-size: 1.9rem; margin-bottom: 4px; }}
  h2 {{ margin-top: 40px; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; }}
  h3 {{ margin-top: 28px; }}
  .lede {{ color: #57606a; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92rem; }}
  caption {{ text-align: left; font-weight: 600; padding: 6px 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #f6f8fa; }}
  .chart {{ margin: 10px 0 20px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 0; list-style: none; }}
  .cards li {{ border: 1px solid #d0d7de; border-radius: 8px;
              padding: 12px 16px; min-width: 180px; }}
  .cards .k {{ display: block; color: #57606a; font-size: 0.85rem; }}
  .cards .v {{ font-size: 1.4rem; font-weight: 700; }}
  .status-ok {{ color: #1a7f37; font-weight: 600; }}
  .status-bad {{ color: #cf222e; font-weight: 600; }}
  footer {{ color: #57606a; padding: 32px 0 48px; font-size: 0.9rem; }}
  code {{ background: #f6f8fa; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<a class="skip" href="#main">Skip to main content</a>
<header>
  <h1>TritonForge</h1>
  <p class="lede">Custom Triton kernels for quantized LLM inference: int4 group-128 dequantized
  GEMM, fused residual RMSNorm, and fused SwiGLU, with autotuning, profiling, and
  re-runnable benchmarks.</p>
</header>
<main id="main">
  <h2>Summary</h2>
  <ul class="cards">
    <li><span class="k">Best int4 GEMM speedup vs eager</span>
        <span class="v">{fmt(best_dq)}x</span></li>
    <li><span class="k">Best SwiGLU speedup vs eager</span>
        <span class="v">{fmt(best_sg)}x</span></li>
    <li><span class="k">Peak achieved bandwidth</span><span class="v">{fmt(best_bw, 1)}%</span></li>
    <li><span class="k">Max int4 abs error</span><span class="v">{fmt(max_err, 4)}</span></li>
    <li><span class="k">compute-sanitizer</span>
        <span class="v {'status-ok' if san_status == 'clean' else 'status-bad'}">\
{esc(san_status)}</span></li>
  </ul>
  <p>{'; '.join(generated)}. Sanitizer: {esc(san_detail)}.</p>

  <h2>Kernel benchmarks</h2>
  <p>All timings are medians from <code>triton.testing.do_bench</code>. "Eager" is the
  PyTorch reference (full dequantize then matmul for the GEMM; native ops for the
  activations). "Default" is a single hand-picked config; "tuned" is the autotuner result.</p>
  {''.join(charts)}
  <h3>int4 fused dequant GEMM</h3>
  {dq_table}
  <h3>Fused RMSNorm+residual and SwiGLU</h3>
  {other_table}

  <h2>End-to-end inference</h2>
  {e2e_html}

  <h2>Method</h2>
  <ol>
    <li>Weights are quantized offline to int4 with per-group (128 along K) asymmetric scales
    and stored K-major so the kernel loads dequantized tiles without any register transpose.</li>
    <li>The fused GEMM splits the packed byte into even- and odd-k nibbles and issues two
    tensor-core dots per K step; scales/zeros are fp16.</li>
    <li>Every kernel is validated against a PyTorch oracle and swept by
    <code>triton.autotune</code>.</li>
    <li>Results are captured on the target GPU by <code>scripts/run_gpu_suite.sh</code> and
    gated in CI by <code>scripts/check_regression.py</code>.</li>
  </ol>
</main>
<footer>
  Generated by <code>scripts/make_report.py</code>. Numbers are produced only by the
  repository scripts; nothing on this page is hand-entered.
</footer>
</body>
</html>
"""


def main() -> None:
    kernels = load_json(RESULTS / "kernels.json")
    e2e = load_json(RESULTS / "e2e.json")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "index.html").write_text(build_html(kernels, e2e))
    print(f"wrote {REPORTS / 'index.html'}")


if __name__ == "__main__":
    main()
