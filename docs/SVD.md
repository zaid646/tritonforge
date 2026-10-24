# SVD Low-Rank Approximation — TritonForge

## Overview

This module adds **singular value decomposition (SVD)** low-rank approximation
to TritonForge, demonstrating how weight matrices in LLM layers can be
compressed via truncated SVD with a measurable trade-off between compression
ratio and approximation error.

## Mathematical formulation

For a weight matrix **W** ∈ ℝ^{m×n}, the full singular value decomposition is:

**W = U Σ Vᵀ**

where:
- **U** ∈ ℝ^{m×m} and **V** ∈ ℝ^{n×n} are orthogonal matrices
- **Σ** = diag(σ₁, σ₂, …, σ_p) with p = min(m,n), σ₁ ≥ σ₂ ≥ … ≥ σ_p ≥ 0

The **rank-r truncated approximation** keeps only the top-r singular triplets:

**W_r = Σ_{i=1}^{r} σ_i u_i v_iᵀ = U_r Σ_r V_rᵀ**

This is the optimal rank-r approximation in both Frobenius and spectral norms
(Eckart–Young–Mirsky theorem):

- ||**W** − **W_r**||_F² = Σ_{i=r+1}^{p} σ_i²
- ||**W** − **W_r**||_2 = σ_{r+1}

## Compression ratio

Storage for the dense matrix: **m × n** parameters

Storage for the rank-r approximation:
- **U_r**: m × r
- **Σ_r**: r (singular values)
- **V_r**: n × r
- Total: **r(m + n + 1)**

Compression ratio:

**ratio = (m × n) / [r(m + n + 1)]**

## API

```python
from tritonforge.svd import svd_low_rank, compression_ratio, approximation_error

W = torch.randn(1024, 1024)
W_r, info = svd_low_rank(W, rank=64)

print(f"Compression: {info['compression_ratio']:.2f}x")
print(f"Frobenius error: {info['frobenius_error']:.6f}")
print(f"Energy retained: {info['energy_retained']:.4f}")
```

## Module reference

| Function | Description |
|---|---|
| `svd_low_rank(weight, rank)` | Rank-r truncated SVD approximation |
| `compression_ratio(m, n, rank)` | Theoretical compression ratio |
| `approximation_error(weight, W_r)` | Frobenius / spectral / relative error |
| `eigen_decompose(matrix)` | Symmetric eigendecomposition (PCA-style) |

## Tests

Run the CPU test suite:

```bash
pytest tests/test_svd.py -v
```

## Benchmark

Run the compression-vs-accuracy sweep:

```bash
python benchmarks/svd_benchmark.py --out results/svd_benchmark.json
```

## Example results

Benchmark on CPU with weight matrices of various shapes:

| Shape | Rank | Compression | Rel. Frobenius Error | Energy Retained |
|---|---:|---:|---:|---:|
| 256×256 | 16 | 8.1x | 0.1234 | 0.9512 |
| 512×512 | 32 | 8.0x | 0.0987 | 0.9634 |
| 1024×1024 | 64 | 8.0x | 0.0765 | 0.9721 |
| 4096×4096 | 128 | 8.0x | 0.0452 | 0.9845 |

*(Numbers illustrative; run the benchmark for actual values.)*

## Use case: LLM weight compression

Apply to attention projection matrices (Q, K, V, O) and MLP weights
(up, gate, down) in transformer layers. A rank-r approximation reduces
parameter count while preserving most of the spectral energy, enabling
smaller model checkpoints with minimal accuracy loss.
