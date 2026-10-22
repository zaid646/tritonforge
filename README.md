# TritonForge

Custom [Triton](https://github.com/triton-lang/triton) kernels for quantized LLM
inference, with autotuning, profiling, end-to-end integration, and a CI regression gate.
Every number in the report is produced by a script in this repository — nothing is
hand-entered.

## Kernels

| Kernel | What it does | Best speedup vs eager (T4) |
| --- | --- | --- |
| `quantized_linear` | fused int4 group-128 dequant + GEMM (+bias) | **14.1x** (decode shapes) |
| `rmsnorm_residual` | fused residual-add + RMSNorm (and plain RMSNorm) | **5.3x** |
| `swiglu` | fused `silu(gate) * up` | **8.9x** (77.8% of T4 peak BW) |

Weights use a K-major int4 packing layout (`packed[k//2, n]`) so the GEMM loads
dequantized tiles with no register transpose, no gather, and no interleave in the hot
loop. See [`docs/DESIGN.md`](docs/DESIGN.md) and the three kernel bugs it fixed in
[`docs/BUG_LOG.md`](docs/BUG_LOG.md).

## Results

- Kernel-level tables and roofline charts: [`reports/index.html`](reports/index.html)
  (generated from [`results/kernels.json`](results/kernels.json)).
- End-to-end Qwen2.5-0.5B comparison (eager / torch.compile / bitsandbytes NF4 /
  TritonForge): [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md),
  [`results/e2e.json`](results/e2e.json).
- Correctness: 31 GPU tests against PyTorch oracles; `compute-sanitizer` memcheck,
  racecheck and synccheck all clean.
- CI regression gate: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) fails the
  build when a committed kernel metric gets slower or less accurate. A worked example
  (deliberate regression, failing PR, log) is in
  [`docs/REGRESSION_DEMO.md`](docs/REGRESSION_DEMO.md).

## TritonForge-X: Intel CPU

A second target: Qwen2.5-0.5B-Instruct on a GPU-less 2-vCPU Xeon (AVX2 only, no
AVX-512/AMX). Full tables and the honest list of regressions:
[`docs/RESULTS_X.md`](docs/RESULTS_X.md); reasoning:
[`docs/ANALYSIS.md`](docs/ANALYSIS.md).

| Batch 1 | tok/s | vs eager | Batch 8 | tok/s | vs eager |
| --- | --- | --- | --- | --- | --- |
| PyTorch eager fp32 | 4.18 | 1.00x | PyTorch eager fp32 | 11.88 | 1.00x |
| OpenVINO fp16 | 7.41 | 1.77x | **ONNX Runtime** | **22.32** | **1.88x** |
| **OpenVINO int8** | **8.33** | **1.99x** | OpenVINO int4 g128 | 21.12 | 1.78x |
| OpenVINO int4 g128 | 8.33 | 1.99x | OpenVINO int8 | 20.92 | 1.76x |

**The batch-8 winner is ONNX Runtime (22.32 tok/s, 1.88x eager); OpenVINO int8 wins
batch 1 (8.33 tok/s, 1.99x).**

- int8 keeps **+0.28%** wikitext-2 PPL; int4 RTN loses **+22.95%** and AWQ still
  loses **+15.74%**, so int4 is reported as unusable for quality-sensitive work.
- DPC++ fused kernels: SwiGLU **6.59x** / RMSNorm+residual **4.50x** over PyTorch at
  large shapes, with all eight parity shapes passing.
- `torch.compile` is not faster than eager here; it is in the table as a non-win.

Reproduce on the same host (paths assume the Colab VM layout):

```bash
pip install -e ".[x]"
python scripts/probe_hardware.py
python scripts/convert_openvino.py --variants fp16 int8 int4_g128
python scripts/validate_openvino.py
RUNS=10 BATCHES="1 8" bash scripts/run_x_benchmarks.sh
python scripts/bench_torch_kernels.py
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O3 -std=c++17 -march=native sycl/tfx_sycl.cpp -o build/tfx_sycl
python scripts/run_sycl.py
python scripts/tune_openvino.py
python scripts/profile_x.py
python scripts/make_x_report.py
python scripts/check_regression_x.py
```

## Layout

```
src/tritonforge/
  env.py            device detection + peak-bandwidth table
  quant.py          int4 quantization, K-major packing, dequantization
  reference.py      PyTorch oracle implementations
  bench.py          timing / bandwidth helpers
  kernels/          the three Triton kernels + autotune configs
  integration/      Drop-in Qwen2.5 QuantLinear + model patcher
benchmarks/         run_kernels.py, probe_dequant.py, run_e2e.py
scripts/            sync, sanitize, regression gate, report generator, GPU suite
baselines/          regression thresholds committed from the T4 run
tests/              quantization, kernel and integration tests
results/x/          Intel CPU JSON (hardware, models, validation, bench, sycl)
scripts/            probe, convert, validate, bench, tune, profile, report, gates
docs/               DESIGN.md, BUG_LOG.md, ANALYSIS.md, RESULTS_X.md, FINAL_REPORT.md
```

## Running

CPU-only (quantization logic + regression gate):

```bash
pip install -e ".[dev]"
pytest -m "not gpu"
python scripts/check_regression.py
```

On a CUDA GPU:

```bash
pip install -e ".[bench,e2e]"
bash scripts/run_gpu_suite.sh          # tests -> sanitizer -> benchmarks -> report -> gate
```

Individual steps: `python benchmarks/run_kernels.py`,
`python benchmarks/probe_dequant.py`, `python benchmarks/run_e2e.py`,
`python scripts/make_report.py`.

## Scope and honesty

The fused kernels win decisively in the memory-bound decode regime. At large prefill
matrices the eager baseline (dequantize once, then cuBLAS) is faster because the fused
kernel re-dequantizes the weight per row tile; `QuantLinear` dispatches around the
measured crossover. At batch-1 decode of a 0.5B model the end-to-end step is dominated by
launch overhead, so `torch.compile` beats every 4-bit path there — the full, unvarnished
numbers are in [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

## License

MIT.
