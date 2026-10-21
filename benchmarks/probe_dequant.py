"""Config sweep for the dequant GEMM to locate the decode/prefill crossover.

    PYTHONPATH=src python benchmarks/probe_dequant.py
"""

from __future__ import annotations

import torch

from tritonforge import env, kernels, quant, reference
from tritonforge.bench import bandwidth_gbps, time_ms
from tritonforge.kernels.dequant_gemm import _CONFIGS

GROUP = 128

SHAPES = [
    (1, 4864, 896),
    (1, 896, 4864),
    (16, 4864, 896),
    (32, 4864, 896),
    (64, 4864, 896),
    (128, 4864, 896),
    (512, 4864, 896),
    (1024, 4864, 896),
]


def main() -> None:
    env.require_cuda()
    dtype = env.default_dtype()
    peak = env.peak_bandwidth_gbps()
    print(f"device {env.describe()} peak_bw={peak}")

    for m, n, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        weight = torch.randn(n, k)
        packed, scales, zeros, _ = quant.quantize_int4_group(weight, GROUP, dtype)
        packed = packed.cuda()
        scales = scales.cuda()
        zeros = zeros.cuda()
        ref = reference.quantized_linear_reference(x, packed, scales, zeros, None, GROUP)

        eager_ms = time_ms(
            lambda x=x, packed=packed, scales=scales, zeros=zeros: (
                reference.quantized_linear_reference(x, packed, scales, zeros, None, GROUP)
            )
        )
        results = []
        for cfg in _CONFIGS:
            c = dict(cfg.kwargs)
            c["num_warps"] = cfg.num_warps
            c["num_stages"] = cfg.num_stages
            try:
                out = kernels.quantized_linear(x, packed, scales, zeros, config=c)
                err = reference.max_abs_error(out, ref)
                ms = time_ms(
                    lambda c=c, x=x, packed=packed, scales=scales, zeros=zeros: (
                        kernels.quantized_linear(x, packed, scales, zeros, config=c)
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append((float("inf"), str(c), f"FAIL {exc}"))
                continue
            tag = (
                f"M{c['BLOCK_M']}N{c['BLOCK_N']}KH{c['BLOCK_KH']}"
                f"w{c['num_warps']}s{c['num_stages']}"
            )
            results.append((ms, tag, f"err={err:.4f}"))
        results.sort(key=lambda r: r[0])
        best = results[0]
        bytes_ = x.numel() * 2 + packed.numel() + scales.numel() * 2 + zeros.numel() * 2 + m * n * 2
        bw = bandwidth_gbps(bytes_, best[0])
        print(
            f"m={m:5d} n={n:5d} k={k:5d} | eager={eager_ms:8.4f}ms "
            f"best={best[0]:8.4f}ms ({best[1]}) speedup={eager_ms / best[0]:6.2f}x "
            f"bw={bw:6.1f}GB/s ({100 * bw / peak:5.1f}% peak)"
        )
        for ms, tag, note in results[:3]:
            print(f"      {tag:28s} {ms:8.4f}ms {note}")


if __name__ == "__main__":
    main()
