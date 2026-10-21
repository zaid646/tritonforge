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
docs/               DESIGN.md, BUG_LOG.md, FINAL_REPORT.md
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
