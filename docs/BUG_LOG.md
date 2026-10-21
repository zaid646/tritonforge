# Bug log

Only real issues encountered while building TritonForge are recorded here. Each entry
states the symptom, the root cause, how it was caught, and the fix. Kernel/design bugs
are separated from measurement and test bugs so the "kernel bug" count is honest.

## Kernel and design bugs (3)

### B1. SwiGLU autotuner wrote a fraction of the output (correctness)

- **Symptom:** the benchmark reported SwiGLU at **2432 GB/s, 760% of the T4 peak**
  bandwidth and an **87.9x** speedup versus eager, which is physically impossible.
- **Root cause:** the autotuned launch used a hard-coded grid derived from the largest
  block size:
  ```python
  _autotuned[(triton.cdiv(n, 4096),)](gate, up, out, n)   # BLOCK is not always 4096
  ```
  `triton.autotune` sweeps `BLOCK` in `{256, 512, 1024, 2048, 4096}`. Only the
  `BLOCK == 4096` config covered the whole tensor; every smaller config wrote only
  `BLOCK/4096` of the elements, which made those configs look artificially fast, so the
  autotuner kept selecting a config that computed `1/16` of the output.
- **Caught by:** a roofline sanity check in the benchmark review — achieved bandwidth
  above the device peak is impossible, which forced an investigation.
- **Fix:** derive the grid from the selected meta-parameters:
  ```python
  _autotuned[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](gate, up, out, n)
  ```
  After the fix SwiGLU reports 248.8 GB/s (77.8% of peak) and <= 8.9x, and the
  correctness tests pass on the full size range.

### B2. Transposed output tile destroyed store coalescing (performance regression)

- **Symptom:** an attempt to remove `tl.trans` from the dequant GEMM by computing the
  `[BLOCK_N, BLOCK_M]` output tile instead of `[BLOCK_M, BLOCK_N]` made the kernel **9x
  slower at M=1** and `0.03x` versus eager at M=1024.
- **Root cause:** the result tensor is row-major `[M, N]`. Writing a `[N, M]` register
  tile meant each row of the tile stored with stride `N` (4864 elements), so stores were
  completely uncoalesced.
- **Caught by:** the per-config sweep in `benchmarks/probe_dequant.py`, which showed the
  new orientation losing to the old one at every shape.
- **Fix:** reverted to `[BLOCK_M, BLOCK_N]` tiles (coalesced stores) and paid the
  `tl.trans` cost, then removed the transpose a different way (see B3).

### B3. Interleave inside the K loop caused register pressure (performance design bug)

- **Symptom:** with the naive `tl.join` + `tl.reshape` unpack, every config with
  `BLOCK_M > 16` was 5-20x slower than `BLOCK_M = 16`, and decode throughput plateaued at
  ~7% of peak bandwidth.
- **Root cause:** materializing the full `[BLOCK_N, BLOCK_K]` fp32 code tile plus the
  `tl.join`/`tl.reshape` shuffle per K step drove register pressure high enough to spill,
  so larger M tiles (which should amortize the weight dequantization) were strictly worse.
- **Caught by:** the config sweep; the best config was pinned to `BLOCK_M=16` for *every*
  problem size, including large prefill matrices where it made no sense.
- **Fix:** changed the packing layout to be K-major (`packed[j, n]` for `j = k//2`) and
  rewrote the kernel to extract the even/odd nibbles and issue two tensor-core dots per
  step — no `tl.join`, no `tl.reshape`, no `tl.trans` in the hot loop. Decode speedup rose
  from ~7x to ~14x and larger M tiles became usable.

## Measurement bug (1)

### M1. First-call autotune polluted inter-token latency

- **Symptom:** the end-to-end harness reported **455 ms/step (2.2 tok/s)** for
  TritonForge, 13x slower than the 34 ms eager baseline, contradicting the kernel-level
  results.
- **Root cause:** the prefill uses `M = prompt_len` and decode uses `M = 1`, which are
  different Triton specializations, so the *first* decode step triggered a fresh autotune
  sweep (dozens of compilations) — and that step was inside the timed loop.
- **Caught by:** profiling a single decode step in isolation (steady state 48 ms/step,
  not 455 ms).
- **Fix:**`benchmarks/run_e2e.py` now warms up both the prefill and five decode steps
  before timing, so ITL reflects steady state.

## Test bug (1)

### T1. int4 round-trip tolerance was tighter than the format allows

- **Symptom:** `test_roundtrip_error_is_small` failed with mean absolute error 0.087
  against a 0.05 threshold.
- **Root cause:** the threshold predated the packing change; uniform int4 has a step of
  `range/15`, so a mean absolute error of ~0.09 for unit-variance weights is expected, not
  a bug.
- **Fix:** relaxed the bound to the format's theoretical range (max_abs < 0.3,
  mean_abs < 0.12) and kept the value explicit in the test.
