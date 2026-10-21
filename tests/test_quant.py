import torch

from tritonforge import quant


def _pack_manual(q: torch.Tensor) -> torch.Tensor:
    even = q[:, 0::2].t().contiguous()
    odd = q[:, 1::2].t().contiguous()
    return (even | (odd << 4)).to(torch.uint8)


def test_pack_roundtrip_known_values():
    q = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.uint8
    )
    packed = _pack_manual(q)
    unpacked = quant.unpack_int4(packed)
    assert torch.equal(unpacked, q.t())


def test_layout_is_k_major():
    w = torch.zeros(3, 8)
    packed, scales, zeros, k_pad = quant.quantize_int4_group(w, group_size=4)
    assert k_pad == 8
    assert packed.shape == (4, 3)
    assert scales.shape == (2, 3)
    assert zeros.shape == (2, 3)


def test_roundtrip_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    recon = quant.quantize_dequantize(w, group_size=128)
    err = quant.quantization_error(w, recon)
    assert err["max_abs"] < 0.3
    assert err["mean_abs"] < 0.12


def test_odd_k_is_padded():
    w = torch.randn(4, 129)
    packed, scales, zeros, k_pad = quant.quantize_int4_group(w, group_size=128)
    assert k_pad == 256
    assert packed.shape == (128, 4)
    recon = quant.dequantize_int4(packed, scales, zeros, group_size=128)
    assert recon.shape == (256, 4)
    assert torch.allclose(recon[129:], torch.zeros_like(recon[129:]), atol=1e-6)


def test_dequant_matches_manual():
    torch.manual_seed(1)
    w = torch.randn(8, 128)
    packed, scales, zeros, _ = quant.quantize_int4_group(w, group_size=128)
    recon = quant.dequantize_int4(packed, scales, zeros, group_size=128)
    assert recon.dtype == torch.float16
    assert recon.shape == (128, 8)
    assert recon.t().shape == w.shape


def test_invalid_group_shapes_raise():
    packed = torch.zeros(64, 4, dtype=torch.uint8)
    scales = torch.zeros(3, 4)
    zeros = torch.zeros(3, 4)
    try:
        quant.dequantize_int4(packed, scales, zeros, group_size=128)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched scales")
