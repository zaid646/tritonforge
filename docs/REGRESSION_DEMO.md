# CI regression gate demonstration

The gate is `scripts/check_regression.py`, driven by thresholds in
`baselines/kernel_baseline.json`. CI (`.github/workflows/ci.yml`) runs it on every push
and pull request.

## What was done

1. Branch `demo/regression-gate` deliberately regressed one committed metric in
   `results/kernels.json`: the int4 dequant GEMM at `M=8, N=4864, K=896` was changed from
   `tuned_ms 0.0665 -> 0.6710` and `speedup_vs_eager 14.09x -> 1.47x` (a simulated 10x
   slowdown).
2. Pull request [#1](https://github.com/zaid646/tritonforge/pull/1) was opened.
3. CI ran and **failed exactly at the regression gate**, with lint and the CPU test subset
   passing.
4. The PR was closed without merging and the branch deleted.

## CI result

```
✓ Lint
✓ Unit tests (CPU subset)
X Kernel/end-to-end regression gate
```

```
[PASS] dequant_gemm m=1 n=4864 k=896         10.45x eager, err 0.0156, 0.0931 ms
[FAIL] dequant_gemm m=8 n=4864 k=896         speedup 1.47x < 8.00x; tuned_ms 0.6710 > 0.15
[PASS] dequant_gemm m=64 n=4864 k=896        3.81x eager, err 0.0625, 0.2578 ms
...
10/11 checks passed
##[error]Process completed with exit code 1.
```

The failure message names the shape, the measured speedup, the threshold it violated, and
the measured latency versus its budget, so the cause is identifiable from the CI log
alone.

To run the same gate locally:

```bash
python scripts/check_regression.py
```
