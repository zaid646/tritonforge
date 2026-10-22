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

## TritonForge-X: Intel CPU port

The second target is a stock x86-64 cloud VM with **no GPU and no modern matrix
extensions** (Broadwell-class Xeon, AVX2/FMA, 2 vCPU). The design constraints change the
answers, so the plan is deliberately different from the T4 work:

- **Backends are compared, not assumed.** PyTorch eager fp32, `torch.compile`, ONNX
  Runtime, and four OpenVINO IR variants (fp16, int8, int4 group-128 RTN, int4 group-128
  AWQ) run the same prompts, greedy decode length and timing loop. The winner is whatever
  the table says.
- **Quantization is offline, never in the decode loop.** NNCF PTQ produces the int8 and
  int4 IRs; AWQ uses 128 wikitext calibration samples and the same group size so the
  size column is directly comparable. The runtime only dequantizes inside OpenVINO's
  kernels, which is the closest CPU analogue to the fused Triton GEMM.
- **Accuracy is a first-class metric.** Top-1 agreement, max/mean absolute error, top-5
  overlap, mean KL and wikitext-2 PPL against a PyTorch fp32 reference are all computed
  from the same 65-prompt set plus 16 x 512 wikitext windows. A faster variant that loses
  quality stays in the table and is labelled as such; see the int4 row.
- **SYCL is used as a kernel laboratory, not as a second runtime.** Two fused kernels
  (`swiglu`, `rmsnorm_residual`) plus a STREAM kernel are written in DPC++ with a
  work-group/vector-width sweep. The reference implementation is the repository's
  existing PyTorch oracle, and parity is checked before any timing. The kernels are not
  wired into the OpenVINO graph: doing so would require a custom OpenVINO extension and
  would make the end-to-end comparison less honest, not more.
- **Roofline uses measured bandwidth.** The plot's slanted roof uses the STREAM number
  from the same binary on the same VM, and the flat roof uses the AVX2 FMA peak
  (`2 FMA ports x 8 lanes x 2 FLOP x 2.2 GHz`). It does not use spec-sheet DRAM bandwidth.
- **CI cannot rely on oneAPI.** GitHub's hosted runners are not guaranteed to expose an
  Intel CPU device, so `tests/test_x_kernels_opencl.py` compiles a line-for-line OpenCL C
  port of the two kernels against POCL on a normal runner and checks parity. The reported
  performance numbers still come from the DPC++ build on the measured Intel VM.
