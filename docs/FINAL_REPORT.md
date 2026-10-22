# Final report

All numbers below were produced by the scripts in this repository on a **Tesla T4**
(compute capability 7.5, 40 SMs, 320 GB/s peak, 14.6 GiB) with torch 2.11.0+cu130,
Triton 3.6.0, Python 3.13, and fp16 (the T4 has no bf16). Raw data:
`results/kernels.json`, `results/kernels.parquet`, `results/e2e.json`. Static report:
`reports/index.html`.

## 1. Kernel results

Medians from `triton.testing.do_bench`; "eager" is the PyTorch oracle (full dequantize +
`matmul` for the GEMM, native ops for the activations).

### int4 fused dequant GEMM (N, K are Qwen2.5-0.5B projection shapes)

| M | N | K | eager ms | default ms | tuned ms | speedup | autotune gain | max abs err | GB/s | % peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4864 | 896 | 0.973 | 0.138 | 0.0915 | **10.45x** | 1.54x | 0.0156 | 25.0 | 7.8 |
| 1 | 896 | 4864 | 0.971 | 0.533 | 0.240 | 4.01x | 2.15x | 0.0312 | 9.6 | 3.0 |
| 8 | 4864 | 896 | 0.987 | 0.158 | 0.0665 | **14.09x** | 2.35x | 0.0625 | 34.6 | 10.8 |
| 16 | 4864 | 896 | 0.981 | 0.151 | 0.0723 | 12.93x | 2.05x | 0.0625 | 32.9 | 10.3 |
| 32 | 4864 | 896 | 0.980 | 0.125 | 0.129 | 7.60x | 2.03x | 0.0625 | 20.8 | 6.5 |
| 64 | 4864 | 896 | 0.981 | 0.247 | 0.258 | 3.81x | 2.75x | 0.0625 | 11.8 | 3.7 |
| 128 | 4864 | 896 | 0.983 | 0.490 | 0.526 | 1.87x | 2.64x | 0.0625 | 7.2 | 2.2 |
| 256 | 4864 | 896 | 0.996 | 1.014 | 1.057 | 0.94x | 2.63x | 0.0625 | 4.9 | 1.5 |
| 512 | 4864 | 896 | 1.055 | 2.051 | 2.094 | 0.50x | 2.69x | 0.0625 | 3.8 | 1.2 |
| 1024 | 4864 | 896 | 1.231 | 4.165 | 4.285 | 0.29x | 2.76x | 0.0625 | 3.2 | 1.0 |

The fused kernel wins strongly for decode-sized matrices (M <= 128) and crosses below
eager near **M = 200**. The reason is structural: the fused kernel dequantizes the weight
tile once per `[BLOCK_M, BLOCK_N]` tile, so the dequantization work is repeated
`ceil(M / BLOCK_M)` times, whereas the eager path dequantizes once and calls cuBLAS. This
is documented and dispatched around in `integration.QuantLinear` (see section 4).

Autotune gain is consistently ~1.5-2.8x. The autotuner matters most at prefill shapes,
where a naive single config is far from optimal.

### Fused RMSNorm+residual and fused SwiGLU

| kernel | M | N | eager ms | tuned ms | speedup | max abs err | GB/s | % peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rmsnorm_residual | 1 | 896 | 0.0255 | 0.0053 | 4.85x | 0.0020 | 1.6 | 0.5 |
| rmsnorm_residual | 8 | 896 | 0.0270 | 0.0051 | 5.31x | 0.0039 | 11.0 | 3.4 |
| rmsnorm_residual | 64 | 896 | 0.0313 | 0.0066 | 4.73x | 0.0039 | 66.8 | 20.9 |
| rmsnorm_residual | 512 | 896 | 0.0594 | 0.0197 | 3.02x | 0.0078 | 180.4 | 56.4 |
| rmsnorm_residual | 1024 | 896 | 0.1503 | 0.0346 | 4.35x | 0.0078 | 209.3 | 65.4 |
| rmsnorm_residual | 512 | 1536 | 0.1183 | 0.0305 | 3.88x | 0.0078 | 203.4 | 63.6 |
| swiglu | 1 | 4864 | 0.0184 | 0.0042 | 4.40x | 0.0000 | 6.6 | 2.1 |
| swiglu | 8 | 4864 | 0.0205 | 0.0055 | 3.74x | 0.0000 | 40.4 | 12.6 |
| swiglu | 64 | 4864 | 0.0464 | 0.0121 | 3.83x | 0.0005 | 152.0 | 47.5 |
| swiglu | 512 | 4864 | 0.5431 | 0.0640 | 8.49x | 0.0020 | 234.0 | 73.1 |
| swiglu | 1024 | 4864 | 1.0703 | 0.1205 | 8.88x | 0.0020 | 248.8 | **77.8** |

The two activation kernels are memory-bound and reach **78% of the T4 roofline** at large
sizes. Decode-sized rows are latency-bound (few programs, one wave).

### Correctness and tooling

- **31 GPU tests** pass against PyTorch oracles, covering decode, prefill, odd K/N/M,
  non-multiples of the block sizes, group-128 padding, dtype/shape validation, and a
  patched Qwen2 model (logits and cached-decode equivalence). Max abs errors above are
  the kernel-vs-oracle errors the tests bound.
- **`compute-sanitizer`**: memcheck `0 errors`, racecheck `0 hazards (0 errors, 0
  warnings)`, synccheck `0 errors` (logs in `artifacts/sanitizer/`).
- Kernel bugs found and fixed: 3 (one correctness, two performance/design); see
  `docs/BUG_LOG.md`.

### Report accessibility and CI

The generated report (`reports/index.html`) is static HTML with inline SVG charts
(`role="img"` with `<title>`/`<desc>`), table `<caption>`s and `<th scope="col">`
headers, a skip link, and no JavaScript.

- **axe-core** (WCAG 2.0/2.1 A + AA + best-practice rules): **0 violations**.
- **Lighthouse** (`reports/lighthouse.json`): performance **100**, accessibility **100**,
  best-practices **100**, SEO **100**.
- Regression gate: PR [#1](https://github.com/zaid646/tritonforge/pull/1) deliberately
  regressed one kernel metric and the `check_regression.py` CI step failed as designed
  (`docs/REGRESSION_DEMO.md`).

## 2. End-to-end: Qwen2.5-0.5B-Instruct on T4

64 generated tokens, batch 1, fp16 (NF4 for bitsandbytes). TTFT is the prefill, ITL is
median inter-token latency, perplexity is wikitext-2 over 16x512-token windows.

| variant | TTFT ms | ITL ms | decode tok/s | perplexity | dPPL vs fp16 | peak VRAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| eager fp16 | 38.4 | 35.7 | 28.0 | 22.417 | - | 1.01 |
| **TritonForge** | 64.9 | 50.0 | 20.0 | 27.125 | +4.71 | **0.51** |
| bitsandbytes NF4 | 58.0 | 52.8 | 18.9 | 25.058 | +2.64 | 0.49 |
| torch.compile | 11.2 | 11.6 | **86.3** | 22.425 | +0.01 | 1.01 |

**Honest read:** at batch-1 decode of a 0.5B model, the step is dominated by launch and
Python overhead, not memory bandwidth, so the custom kernels do not translate into an
end-to-end win. `torch.compile` fuses the whole graph and wins decisively (86 vs 28
tok/s). TritonForge is slightly ahead of bitsandbytes NF4 on throughput (20.0 vs 18.9
tok/s) at similar memory, but is behind eager overall, and its perplexity degradation
(+4.71) is larger than NF4's (+2.64). Raw kernel work is still faster than eager in
aggregate; the gap is launch overhead across ~240 small kernel launches per step.

**Where TritonForge does win end-to-end:** memory. It runs the model in 0.51 GB versus
1.01 GB for fp16 (2x less), matching NF4, which is the regime that matters when weights,
not compute, are the constraint (larger models, longer contexts, batch-1 on small GPUs).

## 3. Limitations and next steps

- Uniform RTN int4 group-128 degrades perplexity more than NF4 (non-uniform levels). Error
  compensation (GPTQ/AWQ) or group-64 would close most of the gap.
- The fused GEMM should split-K and/or persist along M to stop re-dequantizing the weight
  per row tile; that would move the crossover well beyond M=256.
- The end-to-end path needs CUDA-graph capture to remove the per-launch overhead that
  currently hides the kernel gains at batch 1.

## 4. Integration design

`integration.patch_model` replaces the Qwen2 projections with `QuantLinear`, fuses the
post-attention `residual + RMSNorm` into one kernel, fuses the input RMSNorm, and fuses the
MLP `silu(gate) * up`. `QuantLinear` dispatches on token count: fused kernel for
`M <= 128`, dequantize-once + matmul for larger prefill matrices — the crossover measured
in section 1.

## 5. Resume bullets

- Built **TritonForge**, a Triton kernel library for quantized LLM inference: an int4
  group-128 dequantized GEMM (**up to 14.1x** faster than PyTorch eager), a fused
  residual RMSNorm (**up to 5.3x**), and a fused SwiGLU reaching **78% of the T4
  memory-bandwidth roofline**, all with autotuning.
- Designed a K-major int4 packing layout that removed the register transpose and
  interleave from the GEMM hot loop after profiling showed they capped decode throughput;
  validated with **31 GPU tests** and **compute-sanitizer** (0 errors, 0 races).
- Shipped a GitHub Actions regression gate (lint + CPU tests + baseline thresholds) and an
  accessible static report generated from re-runnable benchmarks, and an end-to-end
  Qwen2.5 integration that cuts inference memory **2x** versus fp16 (0.51 GB vs 1.01 GB).

<!-- X_RESULTS_START -->
## TritonForge-X on Intel CPU

Measured on `Intel(R) Xeon(R) CPU @ 2.20GHz` (2 vCPU, AVX2 only, no AVX-512/AMX). Full tables: [RESULTS_X.md](RESULTS_X.md).

| Backend | tok/s b1 (p95) | TTFT ms b1 (p95) | ITL ms b1 (p95) | Speedup b1 | tok/s b8 | TTFT ms b8 | ITL ms b8 | Speedup b8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch eager fp32 | 4.18 (4.80) | 618 (1119) | 229.8 (257.8) | 1.00x | 11.88 | 1740 | 640.1 | 1.00x |
| torch.compile | 4.19 (4.87) | 551 (779) | 231.5 (259.5) | 1.00x | 11.79 | 1768 | 644.0 | 0.99x |
| ONNX Runtime CPU | 4.57 (4.76) | 326 (394) | 215.6 (223.5) | 1.09x | 22.32 | 1630 | 317.4 | 1.88x |
| OpenVINO fp16 | 7.41 (7.81) | 434 (556) | 125.9 (142.3) | 1.77x | 19.71 | 2243 | 346.6 | 1.66x |
| OpenVINO int8 | 8.33 (9.15) | 386 (476) | 113.0 (140.2) | 1.99x | 20.92 | 2418 | 316.7 | 1.76x |
| OpenVINO int4 g128 | 8.33 (9.21) | 406 (567) | 112.7 (142.1) | 1.99x | 21.12 | 2141 | 322.1 | 1.78x |
| OpenVINO int4 g128 AWQ | 8.29 (9.17) | 402 (527) | 113.9 (144.8) | 1.98x | 21.11 | 2104 | 323.3 | 1.78x |

Accuracy parity and the honest list of regressions are in [RESULTS_X.md](RESULTS_X.md#3-accuracy-parity-versus-pytorch-fp32).
<!-- X_RESULTS_END -->
