"""Minimal single-launch driver for compute-sanitizer runs.

Usage: compute-sanitizer --tool memcheck python tests/sanitizer_run.py [tool]
"""

from __future__ import annotations

import sys

import torch

from tritonforge import kernels, quant
from tritonforge.kernels.autotune import (
    DEFAULT_RMSNORM_CONFIG,
    DEFAULT_SWIGLU_CONFIG,
    default_dequant_config,
)

GROUP = 128


def run() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    for m, n, k in [(1, 64, 128), (16, 128, 256), (64, 256, 512)]:
        x = torch.randn(m, k, device=device, dtype=dtype)
        weight = torch.randn(n, k)
        packed, scales, zeros, _ = quant.quantize_int4_group(weight, GROUP, dtype)
        bias = torch.randn(n, device=device, dtype=dtype)
        kernels.quantized_linear(
            x,
            packed.cuda(),
            scales.cuda(),
            zeros.cuda(),
            bias,
            config=default_dequant_config(m),
        )

    for m, n in [(1, 896), (64, 1024)]:
        x = torch.randn(m, n, device=device, dtype=dtype)
        r = torch.randn(m, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype)
        kernels.rmsnorm_residual(x, r, w, config=DEFAULT_RMSNORM_CONFIG)

    for shape in [(1, 4864), (64, 4864)]:
        g = torch.randn(*shape, device=device, dtype=dtype)
        u = torch.randn(*shape, device=device, dtype=dtype)
        kernels.swiglu(g, u, config=DEFAULT_SWIGLU_CONFIG)

    torch.cuda.synchronize()


if __name__ == "__main__":
    run()
    print("sanitizer driver finished:", " ".join(sys.argv[1:]) or "default")
