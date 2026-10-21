"""Fused SwiGLU activation: out = silu(gate) * up."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune import DEFAULT_SWIGLU_CONFIG, ConfigDict, resolve

_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
]


@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    out = (silu * up).to(o_ptr.dtype.element_ty)
    tl.store(o_ptr + offs, out, mask=mask)


_autotuned = triton.autotune(configs=_CONFIGS, key=["n"])(_swiglu_kernel)


def swiglu(gate: torch.Tensor, up: torch.Tensor, config: ConfigDict | None = None) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    if config is None:
        _autotuned[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](gate, up, out, n)
        return out
    cfg = resolve(config, DEFAULT_SWIGLU_CONFIG)
    block = cfg["BLOCK"]
    _swiglu_kernel[(triton.cdiv(n, block),)](
        gate,
        up,
        out,
        n,
        BLOCK=block,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out
