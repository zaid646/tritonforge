#!/usr/bin/env python3
"""Benchmark SVD low-rank approximation: compression ratio vs accuracy trade-off.

Sweeps rank values on weight matrices of various shapes and reports
compression ratio, Frobenius error, and energy retained. Outputs JSON for
CI/reporting.

Usage::

    python benchmarks/svd_benchmark.py --out results/svd_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tritonforge.svd import svd_low_rank, approximation_error, compression_ratio


SHAPES = [
    (256, 256),    # small linear layer
    (512, 512),    # medium
    (1024, 1024),  # large
    (4096, 4096),  # very large (LLM attention)
    (768, 3072),   # MLP up-projection
    (3072, 768),   # MLP down-projection
]

RANKS = [1, 4, 8, 16, 32, 64, 128, 256]


def benchmark_shape(m: int, n: int, ranks: list[int], device: str) -> list[dict[str, Any]]:
    torch.manual_seed(42)
    W = torch.randn(m, n, device=device)
    results = []

    for rank in ranks:
        if rank > min(m, n):
            continue
        start = time.perf_counter()
        W_r, info = svd_low_rank(W, rank=rank)
        elapsed = time.perf_counter() - start

        err = approximation_error(W, W_r)

        results.append({
            "m": m,
            "n": n,
            "rank": rank,
            "compression_ratio": info["compression_ratio"],
            "frobenius_error": err["frobenius_error"],
            "relative_frobenius_error": err["relative_frobenius_error"],
            "spectral_error": err["spectral_error"],
            "energy_retained": info["energy_retained"],
            "compute_seconds": round(elapsed, 6),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/svd_benchmark.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"

    all_results: list[dict[str, Any]] = []
    for m, n in SHAPES:
        shape_results = benchmark_shape(m, n, RANKS, device)
        all_results.extend(shape_results)
        for r in shape_results:
            print(
                f"{m}x{n} rank={r['rank']:3d} "
                f"compress={r['compression_ratio']:6.2f}x "
                f"rel_err={r['relative_frobenius_error']:.4f} "
                f"energy={r['energy_retained']:.4f}"
            )

    out = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        "shapes": SHAPES,
        "ranks": RANKS,
        "results": all_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
