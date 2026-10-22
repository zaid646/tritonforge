# AI-assisted development

Short, factual record of where AI assistance was used on TritonForge-X.

## Tools

- **OpenCode CLI** (model string `opencode/deepseek-v4.1-flash-free`) was used as a
  pair-programming assistant in this repository.
- No other AI coding tool was used for the benchmark or analysis work.

## What it was used for

- Drafting the harness scripts (`probe_hardware.py`, `convert_openvino.py`,
  `validate_openvino.py`, `bench_backends.py`, `tune_openvino.py`, `run_sycl.py`).
- Drafting the SYCL kernels and the OpenCL CI parity harness.
- Diagnosing environment failures: the optimum-intel/diffusers import conflict,
  the `datasets`/`huggingface_hub` URI change, and the optimum-onnx dependency
  conflict are documented in `docs/BUG_LOG.md`.
- Drafting pytest cases, the regression gate and the Markdown report generator.

## What it did not do

- It did not invent, estimate or copy any benchmark number. Every table is rendered
  by `scripts/make_x_report.py` from JSON written by a script that ran on the
  recorded CPU. Failed or unfavourable measurements were kept and reported.
- It did not run autonomous experiments: the session was directed step by step,
  and the repository owner reviewed the changes and the PRs.

## Verification

A reviewer can rerun the pipeline with the commands in `README.md`; the committed
JSON includes the exact CPU string, command-line provenance (for conversions) and
per-run samples (for timing), so the results can be checked rather than trusted.
