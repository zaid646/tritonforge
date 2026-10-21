"""Timing and bandwidth helpers shared by the kernel and end-to-end benchmarks."""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    from triton.testing import do_bench as _triton_do_bench
except Exception:  # pragma: no cover - triton always present in GPU envs
    _triton_do_bench = None


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn: Callable[[], object], warmup: int = 25, rep: int = 100) -> float:
    """Median kernel time in milliseconds."""
    if _triton_do_bench is not None and torch.cuda.is_available():
        return float(_triton_do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
    for _ in range(3):
        fn()
    _cuda_sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    _cuda_sync()
    return start.elapsed_time(end) / rep


def bandwidth_gbps(n_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return n_bytes / (ms * 1e-3) / 1e9


def pct_of_peak(bw_gbps: float, peak_gbps: float | None) -> float | None:
    if not peak_gbps:
        return None
    return 100.0 * bw_gbps / peak_gbps
