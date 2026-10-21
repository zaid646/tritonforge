"""Kernel-level benchmark suite: speedup vs eager, autotune gain, error, bandwidth.

Run on a GPU host:
    PYTHONPATH=src python benchmarks/run_kernels.py

Writes results/kernels.json and results/kernels.parquet (git-committable).
"""

from __future__ import annotations

import json
import pathlib
import platform

import pandas as pd
import torch

from tritonforge import env, kernels, quant, reference
from tritonforge.bench import bandwidth_gbps, pct_of_peak, time_ms
from tritonforge.kernels.autotune import (
    DEFAULT_RMSNORM_CONFIG,
    DEFAULT_SWIGLU_CONFIG,
    default_dequant_config,
)

GROUP = 128
DEQUANT_SHAPES = [
    (1, 4864, 896),
    (1, 896, 4864),
    (8, 4864, 896),
    (16, 4864, 896),
    (32, 4864, 896),
    (64, 4864, 896),
    (128, 4864, 896),
    (256, 4864, 896),
    (512, 4864, 896),
    (512, 896, 4864),
    (1024, 4864, 896),
]
RMSNORM_SHAPES = [(1, 896), (8, 896), (64, 896), (512, 896), (1024, 896), (512, 1536)]
SWIGLU_SHAPES = [(1, 4864), (8, 4864), (64, 4864), (512, 4864), (1024, 4864)]


def _elem_bytes(dtype: torch.dtype) -> int:
    return dtype.itemsize


def bench_dequant(m: int, n: int, k: int, dtype: torch.dtype) -> list[dict]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=dtype)
    weight = torch.randn(n, k)
    packed, scales, zeros, k_pad = quant.quantize_int4_group(weight, GROUP, dtype)
    packed = packed.cuda()
    scales = scales.cuda()
    zeros = zeros.cuda()

    def eager():
        return reference.quantized_linear_reference(x, packed, scales, zeros, None, GROUP)

    default_cfg = default_dequant_config(m)

    def default_run():
        return kernels.quantized_linear(x, packed, scales, zeros, config=default_cfg)

    def tuned_run():
        return kernels.quantized_linear(x, packed, scales, zeros)

    ref = eager()
    got_default = default_run()
    got_tuned = tuned_run()

    rows = []
    err_default = reference.max_abs_error(got_default, ref)
    err_tuned = reference.max_abs_error(got_tuned, ref)
    rel_tuned = reference.max_rel_error(got_tuned, ref)
    eager_ms = time_ms(eager)
    default_ms = time_ms(default_run)
    tuned_ms = time_ms(tuned_run)

    observable = (
        x.numel() * _elem_bytes(dtype)
        + packed.numel()
        + scales.numel() * _elem_bytes(scales.dtype)
        + zeros.numel() * _elem_bytes(zeros.dtype)
        + m * n * _elem_bytes(dtype)
    )
    bw = bandwidth_gbps(observable, tuned_ms)
    peak = env.peak_bandwidth_gbps()
    rows.append(
        {
            "kernel": "dequant_gemm",
            "m": m,
            "n": n,
            "k": k,
            "k_pad": k_pad,
            "dtype": str(dtype),
            "eager_ms": eager_ms,
            "default_ms": default_ms,
            "tuned_ms": tuned_ms,
            "speedup_vs_eager": eager_ms / tuned_ms,
            "autotune_gain": default_ms / tuned_ms,
            "max_abs_err": max(err_default, err_tuned),
            "max_rel_err": rel_tuned,
            "observable_bytes": observable,
            "achieved_bw_gbps": bw,
            "pct_peak_bw": pct_of_peak(bw, peak),
        }
    )
    return rows


def bench_rmsnorm(m: int, n: int, dtype: torch.dtype) -> list[dict]:
    torch.manual_seed(1)
    device = torch.device("cuda")
    x = torch.randn(m, n, device=device, dtype=dtype)
    residual = torch.randn(m, n, device=device, dtype=dtype)
    weight = torch.randn(n, device=device, dtype=dtype)

    def eager():
        return reference.rmsnorm_residual_reference(x, residual, weight)

    def default_run():
        return kernels.rmsnorm_residual(x, residual, weight, config=DEFAULT_RMSNORM_CONFIG)

    def tuned_run():
        return kernels.rmsnorm_residual(x, residual, weight)

    ref_normed, _ = eager()
    got_normed, _ = default_run()
    tuned_normed, _ = tuned_run()

    eager_ms = time_ms(eager)
    default_ms = time_ms(default_run)
    tuned_ms = time_ms(tuned_run)

    observable = 4 * m * n * _elem_bytes(dtype) + n * _elem_bytes(dtype)
    bw = bandwidth_gbps(observable, tuned_ms)
    peak = env.peak_bandwidth_gbps()
    return [
        {
            "kernel": "rmsnorm_residual",
            "m": m,
            "n": n,
            "k": n,
            "k_pad": n,
            "dtype": str(dtype),
            "eager_ms": eager_ms,
            "default_ms": default_ms,
            "tuned_ms": tuned_ms,
            "speedup_vs_eager": eager_ms / tuned_ms,
            "autotune_gain": default_ms / tuned_ms,
            "max_abs_err": max(
                reference.max_abs_error(got_normed, ref_normed),
                reference.max_abs_error(tuned_normed, ref_normed),
            ),
            "max_rel_err": reference.max_rel_error(tuned_normed, ref_normed),
            "observable_bytes": observable,
            "achieved_bw_gbps": bw,
            "pct_peak_bw": pct_of_peak(bw, peak),
        }
    ]


def bench_swiglu(m: int, n: int, dtype: torch.dtype) -> list[dict]:
    torch.manual_seed(2)
    device = torch.device("cuda")
    gate = torch.randn(m, n, device=device, dtype=dtype)
    up = torch.randn(m, n, device=device, dtype=dtype)

    def eager():
        return reference.swiglu_reference(gate, up)

    def default_run():
        return kernels.swiglu(gate, up, config=DEFAULT_SWIGLU_CONFIG)

    def tuned_run():
        return kernels.swiglu(gate, up)

    ref = eager()
    got_default = default_run()
    got_tuned = tuned_run()

    eager_ms = time_ms(eager)
    default_ms = time_ms(default_run)
    tuned_ms = time_ms(tuned_run)

    observable = 3 * m * n * _elem_bytes(dtype)
    bw = bandwidth_gbps(observable, tuned_ms)
    peak = env.peak_bandwidth_gbps()
    return [
        {
            "kernel": "swiglu",
            "m": m,
            "n": n,
            "k": n,
            "k_pad": n,
            "dtype": str(dtype),
            "eager_ms": eager_ms,
            "default_ms": default_ms,
            "tuned_ms": tuned_ms,
            "speedup_vs_eager": eager_ms / tuned_ms,
            "autotune_gain": default_ms / tuned_ms,
            "max_abs_err": max(
                reference.max_abs_error(got_default, ref),
                reference.max_abs_error(got_tuned, ref),
            ),
            "max_rel_err": reference.max_rel_error(got_tuned, ref),
            "observable_bytes": observable,
            "achieved_bw_gbps": bw,
            "pct_peak_bw": pct_of_peak(bw, peak),
        }
    ]


def main() -> None:
    env.require_cuda()
    info = env.get_device_info()
    dtype = env.default_dtype()
    print(f"device: {env.describe()}")
    print(f"dtype: {dtype}")

    rows: list[dict] = []
    for m, n, k in DEQUANT_SHAPES:
        r = bench_dequant(m, n, k, dtype)
        rows += r
        print(f"dequant_gemm m={m} n={n} k={k}: {r[0]['speedup_vs_eager']:.2f}x eager")
    for m, n in RMSNORM_SHAPES:
        r = bench_rmsnorm(m, n, dtype)
        rows += r
        print(f"rmsnorm_residual m={m} n={n}: {r[0]['speedup_vs_eager']:.2f}x eager")
    for m, n in SWIGLU_SHAPES:
        r = bench_swiglu(m, n, dtype)
        rows += r
        print(f"swiglu m={m} n={n}: {r[0]['speedup_vs_eager']:.2f}x eager")

    df = pd.DataFrame(rows)
    out = pathlib.Path("results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "kernels.parquet")
    payload = {
        "device": {
            "name": info.name,
            "cc": info.cc_str,
            "sm_count": info.sm_count,
            "memory_gb": round(info.total_memory_gb, 2),
            "peak_bw_gbps": info.peak_bw_gbps,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "dtype": str(dtype),
        "group_size": GROUP,
        "rows": rows,
    }
    (out / "kernels.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out / 'kernels.json'} and {out / 'kernels.parquet'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
