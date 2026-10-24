"""Unit tests for the SVD low-rank approximation module (tritonforge/svd.py).

All tests run on CPU — no GPU required.
"""

from __future__ import annotations

import torch

from tritonforge.svd import (
    approximation_error,
    compression_ratio,
    eigen_decompose,
    svd_low_rank,
)


# ---------------------------------------------------------------------------
# svd_low_rank tests
# ---------------------------------------------------------------------------


def test_svd_low_rank_reconstructs_full_rank():
    """Rank = min(m,n) should reconstruct the matrix exactly."""
    torch.manual_seed(0)
    W = torch.randn(10, 8)
    W_r, info = svd_low_rank(W, rank=8)  # min(10,8) = 8
    err = approximation_error(W, W_r)
    assert err["relative_frobenius_error"] < 1e-5


def test_svd_low_rank_rank_one():
    """Rank-1 approximation of an outer product should be exact."""
    torch.manual_seed(1)
    u = torch.randn(6, 1)
    v = torch.randn(1, 4)
    W = u @ v  # rank-1 matrix
    W_r, info = svd_low_rank(W, rank=1)
    err = approximation_error(W, W_r)
    assert err["relative_frobenius_error"] < 1e-5
    assert info["energy_retained"] > 0.99


def test_svd_low_rank_compression_ratio():
    """Compression ratio formula should match theory."""
    torch.manual_seed(2)
    W = torch.randn(100, 50)
    W_r, info = svd_low_rank(W, rank=10)
    expected = compression_ratio(100, 50, 10)
    assert abs(info["compression_ratio"] - expected) < 1e-4
    assert info["compression_ratio"] > 1.0


def test_svd_low_rank_energy_retained_increases_with_rank():
    """Higher rank should retain more energy."""
    torch.manual_seed(3)
    W = torch.randn(20, 15)
    _, info_r1 = svd_low_rank(W, rank=1)
    _, info_r5 = svd_low_rank(W, rank=5)
    _, info_r10 = svd_low_rank(W, rank=10)
    assert info_r1["energy_retained"] < info_r5["energy_retained"]
    assert info_r5["energy_retained"] < info_r10["energy_retained"]


def test_svd_low_rank_error_decreases_with_rank():
    """Higher rank should have lower Frobenius error."""
    torch.manual_seed(4)
    W = torch.randn(20, 15)
    _, info_r1 = svd_low_rank(W, rank=1)
    _, info_r5 = svd_low_rank(W, rank=5)
    _, info_r10 = svd_low_rank(W, rank=10)
    assert info_r1["frobenius_error"] > info_r5["frobenius_error"]
    assert info_r5["frobenius_error"] > info_r10["frobenius_error"]


def test_svd_low_rank_invalid_rank_raises():
    W = torch.randn(5, 5)
    for bad_rank in (0, 6, -1):
        try:
            svd_low_rank(W, bad_rank)
            raise AssertionError(f"expected ValueError for rank={bad_rank}")
        except ValueError:
            pass


def test_svd_low_rank_non_2d_raises():
    W = torch.randn(3, 4, 5)
    try:
        svd_low_rank(W, rank=2)
        raise AssertionError("expected ValueError for non-2D input")
    except ValueError:
        pass


def test_svd_low_rank_preserves_dtype_and_device():
    W = torch.randn(8, 8, dtype=torch.float16)
    W_r, _ = svd_low_rank(W, rank=4)
    assert W_r.dtype == torch.float16
    assert W_r.device == W.device


# ---------------------------------------------------------------------------
# eigen_decompose tests
# ---------------------------------------------------------------------------


def test_eigen_decompose_symmetric():
    """Eigendecomposition of a symmetric matrix should reconstruct it."""
    torch.manual_seed(5)
    A = torch.randn(6, 6)
    A = (A + A.T) / 2  # make symmetric
    evals, evecs, info = eigen_decompose(A)
    # A ≈ Q Λ Q^T
    recon = (evecs * evals.unsqueeze(0)) @ evecs.T
    err = (A - recon).abs().max().item()
    assert err < 1e-4


def test_eigen_decompose_eigenvalues_sorted():
    torch.manual_seed(6)
    A = torch.randn(5, 5)
    A = (A + A.T) / 2
    evals, _, _ = eigen_decompose(A)
    assert torch.all(evals[:-1] >= evals[1:])


def test_eigen_decompose_non_square_raises():
    A = torch.randn(3, 4)
    try:
        eigen_decompose(A)
        raise AssertionError("expected ValueError for non-square matrix")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compression_ratio tests
# ---------------------------------------------------------------------------


def test_compression_ratio_formula():
    assert compression_ratio(100, 100, 10) == 10000 / (10 * (100 + 100 + 1))
    assert compression_ratio(10, 10, 5) == 100 / (5 * (10 + 10 + 1))


def test_compression_ratio_rank_one():
    """Rank-1 on a square matrix always compresses if n > 2."""
    assert compression_ratio(100, 100, 1) > 1.0
    assert compression_ratio(3, 3, 1) == 9 / (1 * (3 + 3 + 1))  # 9/7 > 1


# ---------------------------------------------------------------------------
# approximation_error tests
# ---------------------------------------------------------------------------


def test_approximation_error_zero_for_identical():
    W = torch.randn(5, 5)
    err = approximation_error(W, W)
    assert err["frobenius_error"] < 1e-6
    assert err["relative_frobenius_error"] < 1e-6


def test_approximation_error_positive_for_different():
    W1 = torch.randn(5, 5)
    W2 = torch.randn(5, 5)
    err = approximation_error(W1, W2)
    assert err["frobenius_error"] > 0
    assert err["spectral_error"] >= 0
