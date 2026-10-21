"""Layer-wise int4 (group-128) dequant + GEMM fused kernel.

y = x @ dequant(packed) + bias

Layout (see tritonforge.quant): packed is [K_pad/2, N] uint8 where
packed[j, n] = code(2j, n) | code(2j+1, n) << 4, and scales/zeros are [G, N].

Because the weight is stored K-major, a packed tile [BLOCK_KH, BLOCK_N] loads
fully coalesced and, after nibble extraction, splits into the even-k and odd-k
halves of the dequantized weight tile with no gather/transpose. Those two
[BLOCK_KH, BLOCK_N] halves are combined with the even/odd activation columns
via two tensor-core dots per iteration.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune import DEFAULT_DEQUANT_CONFIG, ConfigDict, resolve

_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_KH": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_KH": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_KH": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_KH": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_KH": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_KH": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_KH": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_KH": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_KH": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_KH": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_KH": 32}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_KH": 32}, num_warps=8, num_stages=3),
]


@triton.jit
def _dequant_gemm_kernel(
    x_ptr,
    wq_ptr,
    s_ptr,
    z_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    K,
    G,
    sxm,
    sxk,
    swn,
    ssn,
    son,
    sok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    dtype = x_ptr.dtype.element_ty
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, BLOCK_KH)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, 2 * BLOCK_KH):
        j = k0 // 2 + offs_j
        raw = tl.load(
            wq_ptr + j[:, None] * swn + offs_n[None, :],
            mask=(j[:, None] < K // 2) & mask_n[None, :],
            other=0,
        )
        lo = (raw & 0x0F).to(dtype)
        hi = ((raw >> 4) & 0x0F).to(dtype)

        group = k0 // GROUP
        scale = tl.load(s_ptr + group * ssn + offs_n, mask=mask_n, other=0.0)
        zero = tl.load(z_ptr + group * ssn + offs_n, mask=mask_n, other=0.0)
        w_even = ((lo - zero[None, :]) * scale[None, :]).to(dtype)
        w_odd = ((hi - zero[None, :]) * scale[None, :]).to(dtype)

        k_even = k0 + 2 * offs_j
        x_even = tl.load(
            x_ptr + offs_m[:, None] * sxm + k_even[None, :] * sxk,
            mask=mask_m[:, None] & (k_even[None, :] < K),
            other=0.0,
        )
        x_odd = tl.load(
            x_ptr + offs_m[:, None] * sxm + (k_even[None, :] + 1) * sxk,
            mask=mask_m[:, None] & ((k_even[None, :] + 1) < K),
            other=0.0,
        )
        acc = tl.dot(x_even, w_even, acc, out_dtype=tl.float32)
        acc = tl.dot(x_odd, w_odd, acc, out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    out = acc.to(o_ptr.dtype.element_ty)
    tl.store(
        o_ptr + offs_m[:, None] * son + offs_n[None, :] * sok,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


_autotuned = triton.autotune(
    configs=_CONFIGS, key=["M", "N", "K", "GROUP"]
)(_dequant_gemm_kernel)


def _validate(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int]:
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x dtype must be fp16/bf16, got {x.dtype}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}")
    if packed.is_cuda and not packed.is_contiguous():
        packed = packed.contiguous()
    k_half, n = packed.shape
    k_pad = k_half * 2
    if k_pad % group_size != 0:
        raise ValueError(f"packed K={k_pad} not a multiple of group_size={group_size}")
    groups = k_pad // group_size
    if scales.shape != (groups, n) or zeros.shape != (groups, n):
        raise ValueError(
            f"scales/zeros must be [G, N] = [{groups}, {n}], "
            f"got {tuple(scales.shape)}/{tuple(zeros.shape)}"
        )
    if x.shape[1] > k_pad:
        raise ValueError(f"x has K={x.shape[1]} > packed K={k_pad}")
    return x.shape[0], n, k_pad


def quantized_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
    config: ConfigDict | None = None,
) -> torch.Tensor:
    """Fused int4 dequant + GEMM. Pass ``config`` to bypass autotuning."""
    m, n, k_pad = _validate(x, packed, scales, zeros, group_size)
    x = x.contiguous()
    packed = packed.contiguous()
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    if x.shape[1] < k_pad:
        x = torch.nn.functional.pad(x, (0, k_pad - x.shape[1]))

    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    bias_arg = bias if bias is not None else out
    has_bias = bias is not None and bias.numel() > 0
    groups = k_pad // group_size

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

    args = (
        x,
        packed,
        scales,
        zeros,
        bias_arg,
        out,
        m,
        n,
        k_pad,
        groups,
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        scales.stride(0),
        out.stride(0),
        out.stride(1),
    )
    common = {"GROUP": group_size, "HAS_BIAS": has_bias}

    if config is None:
        _autotuned[grid](*args, **common)
    else:
        cfg = resolve(config, DEFAULT_DEQUANT_CONFIG)
        _dequant_gemm_kernel[(triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))](
            *args,
            **common,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_KH=cfg["BLOCK_KH"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out
