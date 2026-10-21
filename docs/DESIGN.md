# Design notes

## int4 packed layout

Weights are quantized offline with per-group (128 values along K) asymmetric scales and
zeros. The packed tensor is stored **K-major**:

```
packed[j, n]  j = k // 2      low nibble  = code(k = 2j,   n)
                              high nibble = code(k = 2j+1, n)
scales[g, n]  g = k // 128
zeros [g, n]  g = k // 128
```

`packed` is `[K_pad/2, N]`, scales/zeros are `[G, N]`. `K_pad` is K rounded up to a
multiple of the group size so there is never a partial group; the padded rows dequantize
to exactly zero.

This layout is the key performance decision. A packed tile `[BLOCK_KH, BLOCK_N]` loads
fully coalesced (the `N` axis is contiguous), and after nibble extraction it splits
directly into the even-k and odd-k halves of the dequantized weight tile. That lets the
kernel run two tensor-core dots per K step with no gather, no `tl.join`, no `tl.reshape`,
and no `tl.trans` in the hot loop (see `docs/BUG_LOG.md` B3).

## Kernels

- **`kernels/dequant_gemm.py`** — fused `y = x @ dequant(packed) + bias`. Output tiles are
  `[BLOCK_M, BLOCK_N]` for coalesced stores. Scales/zeros are fp16 and loaded once per
  group. The kernel is autotuned over `BLOCK_M/BLOCK_N/BLOCK_KH`, warps and stages.
- **`kernels/rmsnorm_residual.py`** — fused `residual-add + RMSNorm`, with constexpr flags
  `HAS_RESIDUAL` and `WRITE_SUM` so a plain RMSNorm reuses the same kernel without an
  extra tensor read or store.
- **`kernels/swiglu.py`** — fused `silu(gate) * up` in a single pass.

## Decode/prefill crossover

The fused GEMM dequantizes the weight tile once per `[BLOCK_M, BLOCK_N]` tile, so at
large `M` the same weight is re-dequantized for every M tile, while the eager baseline
dequantizes the whole weight once and hands a dense fp16 matrix to cuBLAS. The measured
crossover on the T4 is around `M = 128-256` (see `results/kernels.parquet`): the fused
kernel wins comfortably for decode-sized `M` and loses for large prefill matrices.

`integration.QuantLinear` therefore dispatches on token count: the fused kernel for
`M <= fuse_max_m` (default 128) and a dequantize-once + dense matmul otherwise. This is
a deliberate, measured engineering decision, not a workaround.

## Autotuning

Every kernel ships a small hand-picked default config and a `triton.autotune` set. The
benchmark reports both so the autotune gain is explicit; it ranges from ~1.5x to ~2.8x
for the GEMM and is ~1x for the memory-bound activations at large sizes (where the
hand-picked config is already near the roofline).

## Bandwidth accounting

`observable_bytes` counts the bytes a perfect implementation would have to move:
activation + packed weight + fp16 scales/zeros + output for the GEMM; the four input/output
tensors for RMSNorm; gate + up + output for SwiGLU. Achieved bandwidth is that count over
the measured median time, expressed as a percentage of the device peak (320 GB/s on T4).

## Testing and CI

- 31 tests: quantization rounding/packing/edge shapes, each kernel against a PyTorch
  oracle over decode/prefill/odd shapes, dtype and shape validation, and a patched Qwen2
  model (logits and cached-decode equivalence) built from a tiny random config.
- `compute-sanitizer` memcheck/racecheck/synccheck run over one launch of each kernel.
- CI (`.github/workflows/ci.yml`) lints, runs the CPU test subset, and enforces
  `scripts/check_regression.py` against `baselines/kernel_baseline.json`, which fails the
  build when a kernel gets slower or less accurate than the committed baseline.
