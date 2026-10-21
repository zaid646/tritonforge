import pytest
import torch
from torch import nn

pytest.importorskip("triton")
pytest.importorskip("transformers")

from tritonforge import kernels, reference
from tritonforge.integration import QuantLinear, patch_model

pytestmark = pytest.mark.gpu


def test_quant_linear_matches_reference(device, dtype):
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=True).to(device=device, dtype=dtype)
    qlin = QuantLinear.from_linear(lin, group_size=128, dtype=dtype, fuse_max_m=128)
    x = torch.randn(4, 128, device=device, dtype=dtype)
    got = qlin(x)
    ref = reference.quantized_linear_reference(
        x,
        qlin.packed,
        qlin.scales,
        qlin.zeros,
        qlin.bias,
        128,
    )
    assert torch.allclose(got, ref, atol=3e-2, rtol=3e-2)


def test_quant_linear_prefill_path_matches_fused(device, dtype):
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False).to(device=device, dtype=dtype)
    qlin = QuantLinear.from_linear(lin, group_size=128, dtype=dtype, fuse_max_m=8)
    x = torch.randn(32, 256, device=device, dtype=dtype)
    got = qlin(x)
    ref = reference.quantized_linear_reference(x, qlin.packed, qlin.scales, qlin.zeros, None, 128)
    assert torch.allclose(got, ref, atol=3e-2, rtol=3e-2)


def _tiny_model(dtype):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=dtype).to("cuda").eval()
    return model, cfg


def test_patch_model_logits_close(dtype):
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype)
    ids = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits
    patch_model(model, group_size=128, dtype=dtype)
    with torch.no_grad():
        got = model(ids).logits
    rel = (got.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel.item() < 0.2, rel.item()


def test_fused_norm_and_mlp_are_near_exact(dtype):
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype)
    ids = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits
    patch_model(model, quantize=False, fuse_norm=True, fuse_mlp=True)
    with torch.no_grad():
        got = model(ids).logits
    rel = (got.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel.item() < 1e-2, rel.item()


def test_patch_model_cache_decode_close(dtype):
    torch.manual_seed(0)
    model, cfg = _tiny_model(dtype)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    with torch.no_grad():
        ref_pref = model(prompt, use_cache=True)
        nxt = ref_pref.logits[:, -1, :].argmax(-1, keepdim=True)
        ref_step = model(nxt, past_key_values=ref_pref.past_key_values, use_cache=True).logits
    patch_model(model, group_size=128, dtype=dtype)
    with torch.no_grad():
        got_pref = model(prompt, use_cache=True)
        got_step = model(nxt, past_key_values=got_pref.past_key_values, use_cache=True).logits
    rel = (got_step.float() - ref_step.float()).abs().mean() / ref_step.float().abs().mean()
    assert rel.item() < 0.2, rel.item()


def test_fused_rmsnorm_matches_reference(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(8, 128, device=device, dtype=dtype)
    weight = torch.randn(128, device=device, dtype=dtype)
    got = kernels.rmsnorm(x, weight)
    acc = x.to(torch.float32)
    var = acc.pow(2).mean(-1, keepdim=True)
    ref = (acc * torch.rsqrt(var + 1e-6)).to(dtype) * weight
    assert torch.allclose(got, ref, atol=5e-3, rtol=5e-3)
