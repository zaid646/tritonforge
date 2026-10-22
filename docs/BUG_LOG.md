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

# TritonForge-X (Intel CPU) bugs

These are the real failures hit while porting to an Intel CPU host. They are kept
because each one changed a script or a conclusion in the report.

## Environment bug (4)

### E1. `optimum-intel` import failed on `diffusers` + `huggingface_hub` 1.21

- **Symptom:** `from optimum.intel import OVModelForCausalLM` raised
  `ImportError: cannot import name 'resolve_revision' from 'huggingface_hub.utils'`.
- **Root cause:** the installed `diffusers` release imports a hub helper that was
  removed in `huggingface_hub` 1.21; `optimum-intel` imports `diffusers` even when only
  text generation is used.
- **Caught by:** the first conversion run, before any model was written.
- **Fix:** uninstalled `diffusers` (not needed for this task) instead of pinning the
  whole hub stack backwards. Recorded the remaining pins in `docs/HARDWARE.md`.

### E2. `load_dataset("wikitext")` is no longer a valid script id

- **Symptom:** PPL evaluation aborted with `HfUriError: Invalid HF URI
  'hf://datasets/wikitext'`.
- **Root cause:** `datasets` 4.x removed the legacy short-name resolution.
- **Fix:** use the canonical id `Salesforce/wikitext`, config `wikitext-2-raw-v1`.

### E3. `optimum-onnx` and `optimum-intel` cannot share one venv

- **Symptom:** installing `optimum-onnx` downgraded `optimum` to a version that
  `optimum-intel` rejected, and vice versa.
- **Root cause:** both packages pin the `optimum` core tightly in opposite directions.
- **Fix:** split environments. ONNX export/benchmarks run in `/content/venv_ort`
  (optimum 2.1.0 + optimum-onnx 0.1.0 + transformers 4.57.3); OpenVINO keeps the main
  environment. `scripts/run_x_benchmarks.sh` selects the interpreter per backend.

### E4. `lscpu -J` has no cache objects on this image

- **Symptom:** `probe_hardware.py` crashed with `KeyError: 'caches'`.
- **Root cause:** util-linux on Ubuntu 24.04 omits the `caches` array in JSON mode for
  this kernel.
- **Fix:** fall back to parsing raw `lscpu` for L1d/L2/L3 sizes; cache rows are marked
  `source: "lscpu text"` in `results/x/hardware.json`.

### E5. `TextStreamer` rejects batch sizes above one

- **Symptom:** every batch-8 backend died with
  `ValueError: TextStreamer only supports batch size 1`.
- **Root cause:** the timing hook subclassed `TextStreamer`, which decodes and prints
  text and is hard-coded to one sequence.
- **Fix:** replaced it with a `StoppingCriteria` that only records timestamps (never
  stops generation). It works for any batch size and no longer spends CPU decoding text
  inside the timed loop.

## Measurement bug (5)

### M2. Streaming timer counted the prompt tokens

- **Symptom:** TTFT looked plausible but ITL was ~2x too high and tok/s too low.
- **Root cause:** `TextStreamer.put` is called for the prompt batch first; the harness
  counted that call as a generated token.
- **Fix:** only record a token when `value.numel() == batch_size`, i.e. a real decode
  step. Verified against `max_new_tokens` in every bench JSON.

### M3. `torch.compile(dynamic=False)` recompiled at every decode length

- **Symptom:** the `torch_compile` batch-1 benchmark did not finish its first 10-run
  block after several minutes, versus ~90 s for eager.
- **Root cause:** autoregressive decoding changes the sequence length every step, so
  static-shape specialization triggered a fresh compilation per step.
- **Fix:** compile with `dynamic=True`. The honest result is that `torch.compile` is
  **not faster** than eager for this batch-1 decode workload on this CPU.

### M4. Kernel timings were taken while PPL validation was running

- **Symptom:** the first SYCL sweep reported 2.99 GB/s for SWIGLU and 10.65 GB/s for
  STREAM, far below what the same binary reached later.
- **Root cause:** the validation job owned both vCPUs, so the kernel process was
  timeshared.
- **Fix:** all kernel and end-to-end timings are now run sequentially on an idle VM,
  pinned with `taskset -c 0,1`. The contaminated numbers were discarded, not reported.

### M5. Report reused the large-shape kernel time for every row

- **Symptom:** the per-shape speedup column showed identical SYCL milliseconds for
  512x4864 and 3x895.
- **Root cause:** `make_x_report.py` read `sycl["best"][op]["ms_mean"]`, which is the
  tuning shape, for every validation row.
- **Fix:** `run_sycl.py` now benchmarks each validation shape with the selected config
  and stores shape-matched `ms_mean`; the report reads that field.

### E6. `benchmark_app` rejects hints plus explicit thread counts

- **Symptom:** the OpenVINO layer profile produced an empty table; the log said
  `-nstreams, -nthreads and -pin ... explicitly set -hint option to none`.
- **Fix:** pass `-hint none` when pinning `-nthreads 2` for the detailed counters run.

## Numeric bug (1)

### N1. SYCL SwiGLU vector path dropped the tail of non-multiple-of-4 columns

- **Symptom:** `swiglu 3x895` failed parity with max abs error 0.697 while every
  multiple-of-4 shape passed at 1e-7.
- **Root cause:** the host computed the grid from `n / vec` (vec = 4) even when the
  kernel fell back to the scalar path because `n % 4 != 0`, so only the first 223
  columns were launched.
- **Fix:** compute the elements-per-thread from whether the vector path is actually
  taken (`use_vec4 = vec == 4 && n % 4 == 0`), then size the grid from that. Re-ran all
  eight validation shapes: all pass.

## Accuracy finding, not a bug

### A1. int4 g128 loses 23% PPL; AWQ halves the damage but does not remove it

- The naive RTN int4 model reaches 73.8% top-1 agreement and +22.95% wikitext-2 PPL.
  AWQ improves this to 76.9% and +15.74%, at identical size (325 MB).
- This is a real quality/size trade-off on a 0.5B model, not a harness defect: fp16 is
  exact and int8 is within 0.3% PPL. The report keeps int4 in the table and labels it as
  unsuitable for quality-sensitive use.
