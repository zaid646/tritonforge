import pytest
import torch

pytest.importorskip("triton")

from tritonforge import kernels, quant, reference
from tritonforge.kernels.autotune import (
    DEFAULT_RMSNORM_CONFIG,
    DEFAULT_SWIGLU_CONFIG,
    default_dequant_config,
)

pytestmark = pytest.mark.gpu

GROUP = 128


def _packed(weight, dtype):
    packed, scales, zeros, _ = quant.quantize_int4_group(weight, GROUP, dtype)
    return packed, scales, zeros


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 64, 128),
        (1, 896, 896),
        (3, 127, 129),
        (16, 128, 256),
        (17, 64, 256),
        (64, 512, 896),
        (128, 256, 1024),
    ],
)
def test_quantized_linear_matches_reference(m, n, k, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=dtype)
    weight = torch.randn(n, k)
    packed, scales, zeros = _packed(weight, dtype)
    bias = torch.randn(n, device=device, dtype=dtype)

    out = kernels.quantized_linear(
        x, packed.cuda(), scales.cuda(), zeros.cuda(), bias, config=default_dequant_config(m)
    )
    ref = reference.quantized_linear_reference(
        x, packed.cuda(), scales.cuda(), zeros.cuda(), bias, GROUP
    )

    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), reference.max_abs_error(out, ref)


def test_quantized_linear_no_bias(device, dtype):
    torch.manual_seed(1)
    x = torch.randn(4, 256, device=device, dtype=dtype)
    weight = torch.randn(32, 256)
    packed, scales, zeros = _packed(weight, dtype)
    out = kernels.quantized_linear(
        x, packed.cuda(), scales.cuda(), zeros.cuda(), config=default_dequant_config(4)
    )
    ref = reference.quantized_linear_reference(
        x, packed.cuda(), scales.cuda(), zeros.cuda(), None, GROUP
    )
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_quantized_linear_rejects_bad_dtype(device):
    x = torch.randn(4, 256, device=device, dtype=torch.float32)
    weight = torch.randn(32, 256)
    packed, scales, zeros = _packed(weight, torch.float16)
    with pytest.raises(ValueError):
        kernels.quantized_linear(x, packed.cuda(), scales.cuda(), zeros.cuda())


@pytest.mark.parametrize("m,n", [(1, 896), (4, 896), (16, 1536), (128, 896)])
def test_rmsnorm_residual_matches_reference(m, n, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, n, device=device, dtype=dtype)
    residual = torch.randn(m, n, device=device, dtype=dtype)
    weight = torch.randn(n, device=device, dtype=dtype)

    normed, summed = kernels.rmsnorm_residual(x, residual, weight, config=DEFAULT_RMSNORM_CONFIG)
    ref_normed, ref_summed = reference.rmsnorm_residual_reference(x, residual, weight)

    assert torch.allclose(summed, ref_summed, atol=1e-2, rtol=1e-2)
    assert torch.allclose(normed, ref_normed, atol=5e-3, rtol=5e-3)


def test_rmsnorm_residual_shape_mismatch(device, dtype):
    x = torch.randn(4, 64, device=device, dtype=dtype)
    residual = torch.randn(4, 32, device=device, dtype=dtype)
    weight = torch.randn(64, device=device, dtype=dtype)
    with pytest.raises(ValueError):
        kernels.rmsnorm_residual(x, residual, weight)


@pytest.mark.parametrize("shape", [(1, 4864), (3, 127), (16, 4864), (129, 4864)])
def test_swiglu_matches_reference(shape, device, dtype):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device=device, dtype=dtype)
    up = torch.randn(*shape, device=device, dtype=dtype)
    out = kernels.swiglu(gate, up, config=DEFAULT_SWIGLU_CONFIG)
    ref = reference.swiglu_reference(gate, up)
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)


def test_swiglu_shape_mismatch(device, dtype):
    gate = torch.randn(4, 8, device=device, dtype=dtype)
    up = torch.randn(4, 9, device=device, dtype=dtype)
    with pytest.raises(ValueError):
        kernels.swiglu(gate, up)
