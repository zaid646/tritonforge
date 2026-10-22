#!/usr/bin/env python
"""Validate, tune and benchmark the fused SYCL kernels against PyTorch oracles.

The C++ binary writes raw float32 inputs/outputs; this script loads them,
recomputes the reference with ``src/tritonforge/reference.py`` and reports
max abs/rel error, then sweeps work-group size and vector width.

Usage:
    python scripts/run_sycl.py --binary build/tfx_sycl \
        --out results/x/sycl.json --plot reports/x/roofline.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tritonforge.reference import (  # noqa: E402
    max_abs_error,
    max_rel_error,
    rmsnorm_residual_reference,
    swiglu_reference,
)

WG_SWEEP = [64, 128, 256, 512]
VEC_SWEEP = [1, 4]
SHAPES = {
    "swiglu": [(512, 4864), (1, 896), (17, 1000), (3, 895)],
    "rmsnorm_residual": [(4096, 896), (1, 896), (7, 1000), (3, 895)],
}
TOLERANCES = {
    "swiglu": {"max_abs": 2e-5, "max_rel": 1e-4},
    "rmsnorm_residual": {"max_abs": 2e-5, "max_rel": 1e-4},
}


def run_binary(binary: pathlib.Path, args: list[str], env: dict | None = None) -> dict:
    proc = subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"binary failed: {' '.join(args)}\n{proc.stdout}\n{proc.stderr}")
    return {"stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def load_bin(path: pathlib.Path, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    array = np.fromfile(path, dtype=np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return torch.from_numpy(array.copy())


def bench_once(binary: pathlib.Path, op: str, m: int, n: int, wg: int, vec: int, iters: int,
               json_path: pathlib.Path, dump: str | None = None) -> dict:
    args = [
        "--op",
        op,
        "--m",
        str(m),
        "--n",
        str(n),
        "--wg",
        str(wg),
        "--vec",
        str(vec),
        "--iters",
        str(iters),
        "--json",
        str(json_path),
    ]
    if dump:
        args.extend(["--dump", dump])
    result = run_binary(binary, args)
    payload = json.loads(json_path.read_text())
    payload["stdout"] = result["stdout"]
    return payload


def bench_reps(
    binary: pathlib.Path,
    op: str,
    m: int,
    n: int,
    wg: int,
    vec: int,
    iters: int,
    reps: int = 3,
) -> dict:
    payloads = [
        bench_once(binary, op, m, n, wg, vec, iters, pathlib.Path(tempfile.mktemp()))
        for _ in range(reps)
    ]
    payloads.sort(key=lambda row: row["ms_mean"])
    return payloads[len(payloads) // 2]


def validate(binary: pathlib.Path, op: str, m: int, n: int, wg: int, tmp: pathlib.Path) -> dict:
    dump = tmp / f"{op}_{m}_{n}"
    dump.mkdir(parents=True, exist_ok=True)
    bench_once(binary, op, m, n, wg, 4, 3, tmp / "validate.json", dump=str(dump))
    if op == "swiglu":
        gate = load_bin(dump / "gate.bin", (m, n))
        up = load_bin(dump / "up.bin", (m, n))
        out = load_bin(dump / "out.bin", (m, n))
        expected = swiglu_reference(gate, up)
    else:
        x = load_bin(dump / "x.bin", (m, n))
        residual = load_bin(dump / "residual.bin", (m, n))
        weight = load_bin(dump / "weight.bin", (n,))
        out = load_bin(dump / "out.bin", (m, n))
        expected, _ = rmsnorm_residual_reference(x, residual, weight)
    abs_err = max_abs_error(out, expected)
    rel_err = max_rel_error(out, expected, eps=1e-3)
    tol = TOLERANCES[op]
    return {
        "op": op,
        "m": m,
        "n": n,
        "work_group": wg,
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "tol_max_abs": tol["max_abs"],
        "tol_max_rel": tol["max_rel"],
        "passed": bool(abs_err <= tol["max_abs"] and rel_err <= tol["max_rel"]),
    }


def tidy(value: float) -> float:
    return round(float(value), 6)


def roofline(binary: pathlib.Path, stream: dict, tuned: dict, out_png: pathlib.Path) -> dict:
    points = []
    for op, row in tuned.items():
        if op == "swiglu":
            flops, byts = 5 * row["m"] * row["n"], 3 * 4 * row["m"] * row["n"]
        else:
            flops, byts = 6 * row["m"] * row["n"], 4 * 4 * row["m"] * row["n"]
        ai = flops / byts
        points.append(
            {
                "op": op,
                "m": row["m"],
                "n": row["n"],
                "arithmetic_intensity_flop_per_byte": tidy(ai),
                "achieved_gflops": tidy(row["gflops"]),
                "achieved_gbps": tidy(row["gbps"]),
            }
        )

    # Theoretical AVX2 FP32 peak: 2 FMA units * 8 lanes * 2 FLOP * 2.2 GHz.
    peak_gflops = 2.199998 * 32.0
    peak_gbps = stream["peak_gbps"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.logspace(-2, 2, 200)
        roof = np.minimum(peak_gflops, xs * peak_gbps)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, roof, "k--", linewidth=1.2, label="roofline (measured BW, AVX2 peak)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for point in points:
            ax.scatter(
                point["arithmetic_intensity_flop_per_byte"],
                point["achieved_gflops"],
                s=55,
                label=f"{point['op']} {point['m']}x{point['n']}",
                zorder=3,
            )
        ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title("TritonForge-X CPU roofline (Intel Xeon @ 2.20 GHz, AVX2)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        fig.savefig(out_png.with_suffix(".svg"))
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        print(f"plot skipped: {exc}")

    return {
        "peak_gflops_theoretical_avx2": tidy(peak_gflops),
        "peak_gbps_measured": tidy(peak_gbps),
        "points": points,
        "plot": str(out_png),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "build/tfx_sycl"))
    parser.add_argument("--out", default=str(ROOT / "results/x/sycl.json"))
    parser.add_argument("--plot", default=str(ROOT / "reports/x/roofline.png"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--stream-mb", type=int, default=128)
    args = parser.parse_args()

    binary = pathlib.Path(args.binary)
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}; build sycl/tfx_sycl.cpp first")
    torch.set_num_threads(1)
    hw = json.loads((ROOT / "results/x/hardware.json").read_text())["cpu"]["model_name"]

    info = run_binary(binary, ["--op", "info"])["stdout"]
    device = next(
        (line.split(":", 1)[1].strip() for line in info.splitlines() if "device:" in line), "?"
    )

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="tfx_sycl_"))
    validation = []
    try:
        for op, shapes in SHAPES.items():
            for m, n in shapes:
                row = validate(binary, op, m, n, 128, tmp)
                validation.append(row)
                print(
                    f"validate {op} {m}x{n}: abs {row['max_abs_err']:.2e} "
                    f"rel {row['max_rel_err']:.2e} {'PASS' if row['passed'] else 'FAIL'}"
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tuning: dict[str, list[dict]] = {}
    for op, (m, n) in {"swiglu": (512, 4864), "rmsnorm_residual": (4096, 896)}.items():
        rows = []
        for vec in VEC_SWEEP:
            for wg in WG_SWEEP:
                payload = bench_reps(binary, op, m, n, wg, vec, args.iters)
                row = {
                    "op": op,
                    "m": m,
                    "n": n,
                    "work_group": wg,
                    "vector_width": vec,
                    "ms_mean": tidy(payload["ms_mean"]),
                    "gbps": tidy(payload["gbps"]),
                    "gflops": tidy(payload["gflops"]),
                }
                rows.append(row)
                print(f"tune {op} wg={wg} vec={vec}: {row['gbps']:.2f} GB/s")
        tuning[op] = rows

    stream_rows = []
    for vec in VEC_SWEEP:
        for wg in [128, 256]:
            payload = bench_reps(binary, "stream", args.stream_mb, 0, wg, vec, 20)
            stream_rows.append(
                {
                    "work_group": wg,
                    "vector_width": vec,
                    "mb": args.stream_mb,
                    "gbps": tidy(payload["gbps"]),
                    "ms_mean": tidy(payload["ms_mean"]),
                }
            )
            print(f"stream wg={wg} vec={vec}: {payload['gbps']:.2f} GB/s")
    peak = max(stream_rows, key=lambda row: row["gbps"])

    best = {}
    for op, rows in tuning.items():
        row = max(rows, key=lambda item: item["gbps"])
        row["pct_measured_peak_bw"] = tidy(100.0 * row["gbps"] / peak["gbps"])
        best[op] = row

    default = {
        op: next(row for row in rows if row["work_group"] == 256 and row["vector_width"] == 4)
        for op, rows in tuning.items()
    }
    for op in best:
        best[op]["gain_vs_default_wg256_vec4_pct"] = tidy(
            100.0 * (best[op]["gbps"] - default[op]["gbps"]) / default[op]["gbps"]
        )

    for row in validation:
        cfg = best[row["op"]]
        payload = bench_reps(
            binary,
            row["op"],
            row["m"],
            row["n"],
            cfg["work_group"],
            cfg["vector_width"],
            args.iters,
        )
        row["work_group"] = cfg["work_group"]
        row["vector_width"] = cfg["vector_width"]
        row["ms_mean"] = tidy(payload["ms_mean"])
        row["gbps"] = tidy(payload["gbps"])
        row["gflops"] = tidy(payload["gflops"])

    payload = {
        "hardware_cpu": hw,
        "device": device,
        "device_info": info,
        "note": (
            "Generated by scripts/run_sycl.py; reference is "
            "src/tritonforge/reference.py (PyTorch fp32)."
        ),
        "validation": validation,
        "all_validation_passed": all(row["passed"] for row in validation),
        "tuning": tuning,
        "default_config": default,
        "best": best,
        "stream": {
            "peak_gbps": peak["gbps"],
            "best": peak,
            "rows": stream_rows,
        },
        "roofline": roofline(binary, {"peak_gbps": peak["gbps"]}, best, pathlib.Path(args.plot)),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"best swiglu: {best['swiglu']['gbps']:.2f} GB/s "
          f"({best['swiglu']['pct_measured_peak_bw']:.1f}% peak)")
    print(f"best rmsnorm: {best['rmsnorm_residual']['gbps']:.2f} GB/s "
          f"({best['rmsnorm_residual']['pct_measured_peak_bw']:.1f}% peak)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
