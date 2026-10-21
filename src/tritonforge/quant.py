"""Group-wise int4 quantization with a K-major packed layout.

The packed tensor is stored transposed, ``packed[j, n]`` for ``j = k // 2``:

    low  nibble of packed[j, n] = code for k = 2*j
    high nibble of packed[j, n] = code for k = 2*j + 1

Scales and zeros are likewise stored K-major as ``[num_groups, N]``. This layout
lets the GEMM kernel load dequantized weight tiles directly as ``[BLOCK_K/2, N]``
and feed them to ``tl.dot`` with no register transpose, no gather, and no
interleave inside the hot loop. The extra cost is one transpose at pack time.
"""

from __future__ import annotations

import torch

LOW_MASK = 0x0F


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def padded_k(k: int, group_size: int) -> int:
    return _ceil_div(k, max(group_size, 2)) * max(group_size, 2)


def quantize_int4_group(
    weight: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2-D weight [N, K] into the K-major int4 layout.

    Returns (packed [K_pad/2, N] uint8, scales [G, N] dtype, zeros [G, N] dtype,
    K_pad). K_pad is K rounded up to a multiple of group_size.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(weight.shape)}")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    n, k = weight.shape
    k_pad = padded_k(k, group_size)
    w = weight.detach().to(torch.float32)
    if k_pad != k:
        w = torch.nn.functional.pad(w, (0, k_pad - k))

    groups = k_pad // group_size
    wg = w.reshape(n, groups, group_size)
    w_min = wg.amin(dim=-1)
    w_max = wg.amax(dim=-1)
    scale = ((w_max - w_min) / 15.0).clamp_min(1e-8)
    zero = torch.round(-w_min / scale).clamp(0, 15)

    q = torch.round(wg / scale.unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0, 15)
    q = q.reshape(n, k_pad).to(torch.uint8)

    even = q[:, 0::2].t().contiguous()
    odd = q[:, 1::2].t().contiguous()
    packed = (even | (odd << 4)).to(torch.uint8).contiguous()
    return packed, scale.t().contiguous().to(dtype), zero.t().contiguous().to(dtype), k_pad


def unpack_int4(packed: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Expand K-major packed nibbles [K_pad/2, N] to uint8 codes [K_pad, N]."""
    if packed.ndim != 2:
        raise ValueError("packed must be 2-D")
    half, n = packed.shape
    out = torch.empty((half * 2, n), device=packed.device, dtype=torch.uint8)
    out[0::2, :] = packed & LOW_MASK
    out[1::2, :] = (packed >> 4) & LOW_MASK
    return out if k is None else out[:k]


def _group_expand(param: torch.Tensor, k_pad: int, n: int, group_size: int) -> torch.Tensor:
    groups = k_pad // group_size
    if param.shape != (groups, n):
        got = tuple(param.shape)
        raise ValueError(f"expected group params [G, N] = [{groups}, {n}], got {got}")
    return (
        param.to(torch.float32)
        .reshape(groups, 1, n)
        .expand(groups, group_size, n)
        .reshape(k_pad, n)
    )


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct the dense, K-major weight [K_pad, N] from int4 + group params."""
    half, n = packed.shape
    k_pad = half * 2
    if k_pad % group_size != 0:
        raise ValueError(f"K_pad={k_pad} must be a multiple of group_size={group_size}")
    codes = unpack_int4(packed).to(torch.float32)
    scale = _group_expand(scales, k_pad, n, group_size)
    zero = _group_expand(zeros, k_pad, n, group_size)
    return ((codes - zero) * scale).to(dtype)


def quantize_dequantize(
    weight: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    packed, scales, zeros, k_pad = quantize_int4_group(weight, group_size, dtype)
    dense = dequantize_int4(packed, scales, zeros, group_size, dtype)
    return dense[: weight.shape[1]].t().contiguous()


def quantization_error(weight: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    a = weight.detach().to(torch.float32)
    b = reconstructed.detach().to(torch.float32).reshape_as(a)
    abs_err = (a - b).abs()
    rel = (abs_err / a.abs().clamp_min(1e-3)).mean()
    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "mean_rel": rel.item(),
    }
