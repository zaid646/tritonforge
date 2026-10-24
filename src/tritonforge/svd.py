"""Singular Value Decomposition (SVD) low-rank approximation for weight compression.

Implements truncated SVD (a.k.a. low-rank approximation) of a 2-D weight matrix,
plus eigen-decomposition analysis for symmetric matrices. Provides:

- ``svd_low_rank`` — rank-r truncated SVD approximation W ≈ U_r Σ_r V_r^T
- ``compression_ratio`` — storage savings vs the dense matrix
- ``approximation_error`` — spectral / Frobenius error vs the original
- ``eigen_decompose`` — symmetric eigendecomposition for PCA-style analysis

Mathematical formulation:

For a weight matrix W ∈ R^{m×n}, the full SVD is:

    W = U Σ V^T

where U ∈ R^{m×m}, V ∈ R^{n×n} are orthogonal and Σ = diag(σ_1, ..., σ_p)
with p = min(m,n) and σ_1 ≥ σ_2 ≥ ... ≥ σ_p ≥ 0.

The rank-r truncated approximation keeps only the top-r singular triplets:

    W_r = Σ_{i=1}^{r} σ_i u_i v_i^T = U_r Σ_r V_r^T

Storage:
- Dense:     m × n parameters
- Low-rank:  m × r + r + r × n = r(m + n + 1) parameters
- Compression: mn / [r(m + n + 1)]

Error (Eckart–Young–Mirsky theorem):
    ||W − W_r||_F² = Σ_{i=r+1}^{p} σ_i²
    ||W − W_r||_2   = σ_{r+1}

Usage::

    from tritonforge.svd import svd_low_rank, compression_ratio, approximation_error

    W = torch.randn(1024, 1024)
    W_r, info = svd_low_rank(W, rank=64)
    print(f"compression: {info['compression_ratio']:.2f}x")
    print(f"Frobenius error: {info['frobenius_error']:.6f}")
"""
from __future__ import annotations

import torch


def svd_low_rank(
    weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the rank-``rank`` truncated SVD approximation of ``weight``.

    Parameters
    ----------
    weight : torch.Tensor
        2-D weight matrix of shape (m, n).
    rank : int
        Target rank r (1 ≤ r ≤ min(m, n)).

    Returns
    -------
    W_r : torch.Tensor
        Approximated weight matrix of shape (m, n), same dtype/device as input.
    info : dict
        Dictionary with singular values, compression ratio, and error metrics.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight matrix, got shape {tuple(weight.shape)}")
    m, n = weight.shape
    p = min(m, n)
    if not 1 <= rank <= p:
        raise ValueError(f"rank must be between 1 and {p}, got {rank}")

    w = weight.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)

    # Truncate to rank r
    U_r = U[:, :rank]                      # (m, r)
    S_r = S[:rank]                         # (r,)
    V_r = Vh[:rank, :]                     # (r, n) — already transposed by SVD

    # Reconstruct: W_r = U_r @ diag(S_r) @ V_r  (V_r = Vh[:r,:] = V_r^T)
    W_r = (U_r * S_r.unsqueeze(0)) @ V_r   # (m, r) @ (r, n) = (m, n)
    W_r = W_r.to(weight.dtype)

    # Compression ratio: dense params / low-rank params
    dense_params = m * n
    low_rank_params = rank * (m + n + 1)  # U_r (m×r) + S_r (r) + V_r (n×r)
    compression_ratio = dense_params / low_rank_params if low_rank_params > 0 else float("inf")

    # Error metrics
    frob_err = torch.linalg.norm(w - W_r.to(torch.float32), ord="fro").item()
    spec_err = S[rank].item() if rank < p else 0.0  # σ_{r+1}
    total_energy = torch.sum(S ** 2).item()
    retained_energy = torch.sum(S_r ** 2).item()
    energy_retained = retained_energy / total_energy if total_energy > 0 else 0.0

    info = {
        "rank": rank,
        "m": m,
        "n": n,
        "dense_params": dense_params,
        "low_rank_params": low_rank_params,
        "compression_ratio": round(compression_ratio, 4),
        "frobenius_error": round(frob_err, 6),
        "spectral_error": round(spec_err, 6),
        "energy_retained": round(energy_retained, 4),
        "singular_values": S[: min(20, p)].tolist(),
    }
    return W_r, info


def eigen_decompose(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Eigen-decompose a symmetric matrix.

    For a symmetric matrix A ∈ R^{n×n}:

    A = Q Λ Q^T

    where Q is orthogonal and Λ = diag(λ_1, ..., λ_n) with real eigenvalues.

    Returns eigenvalues, eigenvectors, and analysis info.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected square matrix, got shape {tuple(matrix.shape)}")

    a = matrix.detach().to(torch.float32)
    # Symmetrize to guard against tiny numerical asymmetry
    a_sym = (a + a.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(a_sym)

    # Sort descending by eigenvalue
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n = a_sym.shape[0]
    total_variance = torch.sum(eigenvalues ** 2).item()
    info = {
        "n": n,
        "trace": round(torch.trace(a_sym).item(), 6),
        "sum_eigenvalues": round(eigenvalues.sum().item(), 6),
        "top_eigenvalue": round(eigenvalues[0].item(), 6),
        "bottom_eigenvalue": round(eigenvalues[-1].item(), 6),
        "condition_number": round(
            (eigenvalues[0] / eigenvalues[-1]).item() if eigenvalues[-1] != 0 else float("inf"),
            4,
        ),
        "total_variance": round(total_variance, 6),
    }
    return eigenvalues, eigenvectors, info


def compression_ratio(m: int, n: int, rank: int) -> float:
    """Return the theoretical compression ratio for an m×n matrix at rank r.

    ratio = (m × n) / (r × (m + n + 1))
    """
    low_rank_params = rank * (m + n + 1)
    if low_rank_params == 0:
        return float("inf")
    return (m * n) / low_rank_params


def approximation_error(
    weight: torch.Tensor,
    W_r: torch.Tensor,
) -> dict[str, float]:
    """Compute Frobenius and spectral approximation error between W and W_r."""
    w = weight.detach().to(torch.float32)
    w_r = W_r.detach().to(torch.float32)
    diff = w - w_r
    frob = torch.linalg.norm(diff, ord="fro").item()
    # Spectral norm = largest singular value of the difference
    _, S_diff, _ = torch.linalg.svd(diff, full_matrices=False)
    spec = S_diff[0].item() if len(S_diff) > 0 else 0.0
    rel_frob = frob / torch.linalg.norm(w, ord="fro").item() if torch.linalg.norm(w, ord="fro").item() > 0 else 0.0
    return {
        "frobenius_error": round(frob, 6),
        "spectral_error": round(spec, 6),
        "relative_frobenius_error": round(rel_frob, 6),
    }


__all__ = [
    "svd_low_rank",
    "eigen_decompose",
    "compression_ratio",
    "approximation_error",
]
