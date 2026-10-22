# TritonForge-X analysis

All numbers below are from `results/x/*.json` (see [RESULTS_X.md](RESULTS_X.md)).
The host is a shared 2-vCPU Xeon at 2.20 GHz with AVX2/FMA only — no VNNI, no
AVX-512, no AMX.

## Where the time goes at batch 1

The measured STREAM peak of the same machine is **11.96 GB/s**. A pure
weight-streaming decode step for Qwen2.5-0.5B would therefore take:

| Weights | Bytes | Lower bound at 11.96 GB/s | Implied tok/s |
| --- | --- | --- | --- |
| fp16 | 989.9 MB | 82.8 ms | 12.1 |
| int8 | 497.7 MB | 41.6 ms | 24.0 |
| int4 g128 | 325.1 MB | 27.2 ms | 36.8 |

The measured OpenVINO numbers are **7.4 tok/s (fp16), 8.3 tok/s (int8) and
8.3 tok/s (int4)**. int8/int4 are only 23-35% of their bandwidth bound and do not
speed up relative to each other: on this CPU the low-bit formats save reads but
pay for unpacking in a kernel that has no VNNI/AMX instructions and runs on two
cores. The practical conclusion is that a 0.5B decode on this VM is limited by
per-layer overhead and scalar/AVX2 work, not by DRAM bandwidth alone.

fp16 OpenVINO is **1.77x** faster than PyTorch eager (7.41 vs 4.18 tok/s). The gap to the 12.1 tok/s
bound is the price of the remaining non-matmul work (attention, normalization,
sampling) and the 24 sequential layer launches.

## Batch 8 changes the ranking

Batching amortizes weight reads but the VM only has two cores:

- ONNX Runtime goes from 4.57 to **22.32 tok/s** (1.88x vs eager) and wins b8.
- OpenVINO int4 goes from 8.33 to **21.12 tok/s**; fp16 only reaches 19.71.
- PyTorch eager/torch.compile roughly double (11.88/11.79) but stay last.
- Per-sequence latency gets worse for every backend (ITL 318-644 ms), so b8 is a
  throughput choice, not a latency choice.

`torch.compile` is statistically indistinguishable from eager at both batch sizes
(4.19 vs 4.18 tok/s, 11.79 vs 11.88), so it is reported as a non-win rather than
hidden.

## Accuracy is the price of int4

fp16 matches the fp32 reference (100% top-1, 0.0012 max abs, PPL delta −0.00%).
int8 is usable: 95.4% top-1, +0.28% wikitext-2 PPL, at 48.8% of the fp16 size.
int4 RTN is not: 73.8% top-1 and **+22.95% PPL**. AWQ at the same 325 MB halves
the damage to 76.9% and +15.74%, but it is still far outside a quality-sensitive
budget. For this model on this CPU, the defensible choice is fp16 when memory is
available and int8 when it is not.

## Kernels and roofline

The DPC++ kernels are checked against the PyTorch oracles before timing:

| Kernel | Shape | PyTorch | SYCL | Speedup | % of measured STREAM |
| --- | --- | --- | --- | --- | --- |
| swiglu | 512x4864 | 16.80 ms | 2.55 ms | 6.59x | 94-98% |
| rmsnorm_residual | 4096x896 | 29.78 ms | 6.61 ms | 4.50x | 74% |

At decode-sized tensors (1x896, 3x895) the SYCL kernels are launch-bound and only
1.1-3.1x faster; the roofline plot shows them far below the roof because the
arithmetic intensity is tiny and the transfer is a few microseconds. The sweep
also shows that the usual "wider vectors are better" rule does not hold here:
vector width 1 wins for swiglu and ties for RMSNorm, because the activation is
dominated by the exp/divide and the reduction, not by load instruction count.

`benchmark_app` detailed counters attribute **19.7%** of OpenVINO time to the
`lm_head` MatMul and ~1.6% to each of the 24 `down_proj` MatMuls, i.e. the time is
spread across the layer stack with no single hot kernel. The torch profiler
attributes **77.6%** of eager CPU time to `aten::mm`, which is why replacing the
runtime (OpenVINO/ONNX) matters more than micro-optimizing Python.

## Tuning honestly

Seven OpenVINO configurations (latency/throughput hints, 1/2 threads, 1/2 streams,
pin to one or two cores) move batch-1 throughput only between 8.07 and 8.51 tok/s;
the best is **+2.6%** over the default. Two streams lose throughput. This is
reported as a small win, not a headline.

## Limitations

- The VM is shared; repeated runs of the same kernel vary by up to ~30%, so kernel
  numbers use a trimmed mean of 100 launches and three independent processes
  (median). End-to-end numbers are 10 runs after 2 warmups.
- PPL uses 16 x 512 wikitext-2 windows (8176 scored tokens), not the full split.
- Only 32 new tokens are decoded; long-context behavior is not measured.
- Only Qwen2.5-0.5B-Instruct is covered (depth over breadth).
- `perf` is unavailable on this kernel; profiling uses torch profiler and
  OpenVINO `benchmark_app` counters.
