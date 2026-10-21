"""Drop-in Qwen2.5 integration for the TritonForge int4 + fused kernels.

``patch_model`` performs three substitutions:

1. every ``nn.Linear`` projection (q/k/v/o, gate/up/down) becomes a
   :class:`QuantLinear` backed by the int4 fused GEMM;
2. each decoder layer's post-attention ``residual + RMSNorm`` becomes a single
   fused kernel launch;
3. each MLP's ``silu(gate) * up`` becomes a single fused SwiGLU launch.

``QuantLinear`` dispatches by token count: the fused dequant GEMM wins in the
memory-bound decode regime, while for large prefill matrices the kernel is
latency-bound and re-dequantizes the weight once per row tile, so a
dequantize-once + dense matmul is used instead. The crossover (M ~= 128) was
measured in ``benchmarks/probe_dequant.py`` and ``results/kernels.parquet``.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from .. import kernels, quant

DEFAULT_FUSE_MAX_M = 128


class QuantLinear(nn.Module):
    """Linear layer holding a K-major int4 weight plus group scales/zeros."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
        group_size: int = 128,
        fuse_max_m: int = DEFAULT_FUSE_MAX_M,
    ) -> None:
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.fuse_max_m = fuse_max_m

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        fuse_max_m: int = DEFAULT_FUSE_MAX_M,
    ) -> QuantLinear:
        weight = linear.weight.detach().float()
        packed, scales, zeros, _ = quant.quantize_int4_group(weight, group_size, dtype)
        device = linear.weight.device
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(dtype).to(device).contiguous()
        return cls(
            packed.to(device).contiguous(),
            scales.to(device).contiguous(),
            zeros.to(device).contiguous(),
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            group_size=group_size,
            fuse_max_m=fuse_max_m,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        x2 = x.reshape(-1, in_shape[-1]).contiguous()
        m = x2.shape[0]
        if m <= self.fuse_max_m:
            out = kernels.quantized_linear(
                x2.contiguous(), self.packed, self.scales, self.zeros, self.bias, self.group_size
            )
        else:
            weight = quant.dequantize_int4(
                self.packed, self.scales, self.zeros, self.group_size, x.dtype
            )[: self.in_features]
            out = x2 @ weight
            if self.bias is not None:
                out = out + self.bias
        return out.reshape(*in_shape[:-1], self.out_features)


def _fused_rmsnorm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    r2 = residual.reshape(-1, shape[-1]).contiguous()
    normed, summed = kernels.rmsnorm_residual(x2, r2, weight, eps)
    return normed.reshape(shape), summed.reshape(shape)


def _rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    x2 = hidden_states.reshape(-1, shape[-1]).contiguous()
    normed = kernels.rmsnorm(x2, self.weight, self.variance_epsilon)
    return normed.reshape(shape)


def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    return self.down_proj(kernels.swiglu(gate, up))


def _layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    use_cache: bool | None = False,
    position_embeddings=None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states, residual = _fused_rmsnorm_residual(
        hidden_states,
        residual,
        self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon,
    )
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


def patch_model(
    model: nn.Module,
    group_size: int = 128,
    fuse_max_m: int = DEFAULT_FUSE_MAX_M,
    dtype: torch.dtype = torch.float16,
    quantize: bool = True,
    fuse_norm: bool = True,
    fuse_mlp: bool = True,
) -> nn.Module:
    """Patch a Qwen2 model in place; returns the same model."""
    layers = model.model.layers
    for layer in layers:
        if quantize:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(attn, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, name, QuantLinear.from_linear(lin, group_size, dtype, fuse_max_m))
            mlp = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(mlp, name, QuantLinear.from_linear(lin, group_size, dtype, fuse_max_m))
        if fuse_norm:
            layer.input_layernorm.forward = types.MethodType(
                _rmsnorm_forward, layer.input_layernorm
            )
            layer.forward = types.MethodType(_layer_forward, layer)
        if fuse_mlp:
            layer.mlp.forward = types.MethodType(_mlp_forward, layer.mlp)
    return model
