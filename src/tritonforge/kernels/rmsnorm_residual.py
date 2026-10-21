"""Fused residual-add + RMSNorm kernel.

``HAS_RESIDUAL`` selects between the fused ``norm(x + residual)`` form and a
plain RMSNorm; ``WRITE_SUM`` toggles writing the post-add tensor that the
residual form needs for the following residual connection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune import DEFAULT_RMSNORM_CONFIG, ConfigDict, resolve

_CONFIGS = [
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=1),
]


@triton.jit
def _rmsnorm_residual_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    o_ptr,
    y_ptr,
    M,
    N,
    eps,
    sxm,
    sxn,
    srm,
    srn,
    som,
    son,
    sym,
    syn,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    sum_sq = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(r_ptr + row * srm + cols * srn, mask=mask, other=0.0).to(tl.float32)
            s = x + r
        else:
            s = x
        sum_sq += s * s

    mean_sq = tl.sum(sum_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask, other=0.0)
        if HAS_RESIDUAL:
            r = tl.load(r_ptr + row * srm + cols * srn, mask=mask, other=0.0)
            s = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
        else:
            s = x
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        normed = (s.to(tl.float32) * rstd * w.to(tl.float32)).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + row * som + cols * son, normed, mask=mask)
        if WRITE_SUM:
            tl.store(y_ptr + row * sym + cols * syn, s, mask=mask)


_autotuned = triton.autotune(configs=_CONFIGS, key=["N"])(_rmsnorm_residual_kernel)


def _launch(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    write_sum: bool,
    config: ConfigDict | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if x.ndim != 2:
        raise ValueError(f"expected 2-D input, got {tuple(x.shape)}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError(f"weight must be 1-D of length {x.shape[1]}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != x {tuple(x.shape)}")

    x = x.contiguous()
    r = x if residual is None else residual.contiguous()
    weight = weight.contiguous()
    m, n = x.shape
    normed = torch.empty_like(x)
    summed = torch.empty_like(x) if write_sum else x
    args = (
        x,
        r,
        weight,
        normed,
        summed,
        m,
        n,
        eps,
        x.stride(0),
        x.stride(1),
        r.stride(0),
        r.stride(1),
        normed.stride(0),
        normed.stride(1),
        summed.stride(0),
        summed.stride(1),
    )
    has_residual = residual is not None
    flags = {"HAS_RESIDUAL": has_residual, "WRITE_SUM": write_sum}

    if config is None:
        _autotuned[(m,)](*args, **flags)
    else:
        cfg = resolve(config, DEFAULT_RMSNORM_CONFIG)
        _rmsnorm_residual_kernel[(m,)](
            *args,
            **flags,
            BLOCK=cfg["BLOCK"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return normed, (summed if write_sum else None)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    config: ConfigDict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (norm(x + residual), x + residual)."""
    normed, summed = _launch(x, residual, weight, eps, True, config)
    assert summed is not None
    return normed, summed


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    config: ConfigDict | None = None,
) -> torch.Tensor:
    normed, _ = _launch(x, None, weight, eps, False, config)
    return normed
