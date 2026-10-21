"""Pure-PyTorch reference implementations used as correctness oracles."""

from __future__ import annotations

import torch

from . import quant


def dequant_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dense, K-major weight [K_pad, N]."""
    return quant.dequantize_int4(packed, scales, zeros, group_size, dtype)


def quantized_linear_reference(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Eager reference: dequantize the full weight, then a dense matmul."""
    weight = dequant_weight(packed, scales, zeros, group_size, x.dtype)
    if weight.shape[0] != x.shape[1]:
        weight = weight[: x.shape[1]]
    out = x @ weight
    if bias is not None:
        out = out + bias
    return out


def rmsnorm_residual_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x + residual
    acc = summed.to(torch.float32)
    variance = acc.pow(2).mean(-1, keepdim=True)
    normed = acc * torch.rsqrt(variance + eps)
    normed = normed.to(x.dtype) * weight
    return normed, summed


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_f = gate.to(torch.float32)
    up_f = up.to(torch.float32)
    out = (gate_f * torch.sigmoid(gate_f)) * up_f
    return out.to(gate.dtype)


def relu2_reference(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.float32).clamp_min(0.0) ** 2).to(x.dtype)


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> float:
    denom = b.float().abs().clamp_min(eps)
    return ((a.float() - b.float()).abs() / denom).max().item()


__all__ = [
    "dequant_weight",
    "quantized_linear_reference",
    "rmsnorm_residual_reference",
    "swiglu_reference",
    "relu2_reference",
    "max_abs_error",
    "max_rel_error",
]
