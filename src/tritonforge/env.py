"""Hardware/runtime detection so every benchmark adapts to the target GPU."""

from __future__ import annotations

import dataclasses

import torch

T4_PEAK_BW_GBPS = 320.0

_KNOWN_PEAK_BW = {
    "T4": 320.0,
    "L4": 300.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE-40GB": 1555.0,
    "RTX 3090": 936.2,
    "RTX 3080": 760.3,
    "RTX 4090": 1008.0,
    "RTX 4080": 716.8,
    "H100": 3350.0,
}


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    name: str
    cc: tuple[int, int]
    total_memory_gb: float
    sm_count: int
    is_t4: bool
    supports_bf16: bool
    peak_bw_gbps: float | None

    @property
    def cc_str(self) -> str:
        return f"{self.cc[0]}.{self.cc[1]}"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def require_cuda() -> None:
    if not cuda_available():
        raise RuntimeError("TritonForge kernels require a CUDA GPU; none is available.")


def _lookup_peak_bw(name: str) -> float | None:
    for key, value in _KNOWN_PEAK_BW.items():
        if key.lower() in name.lower():
            return value
    return None


def get_device_info(device: int | str | torch.device | None = None) -> DeviceInfo:
    if not cuda_available():
        return DeviceInfo(
            name="cpu",
            cc=(0, 0),
            total_memory_gb=0.0,
            sm_count=0,
            is_t4=False,
            supports_bf16=False,
            peak_bw_gbps=None,
        )

    index = 0 if device is None else torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    name = props.name
    return DeviceInfo(
        name=name,
        cc=cc,
        total_memory_gb=props.total_memory / (1024**3),
        sm_count=props.multi_processor_count,
        is_t4="T4" in name or cc == (7, 5),
        supports_bf16=cc[0] >= 8,
        peak_bw_gbps=_lookup_peak_bw(name),
    )


def default_dtype(device: int | str | torch.device | None = None) -> torch.dtype:
    info = get_device_info(device)
    return torch.bfloat16 if info.supports_bf16 else torch.float16


def peak_bandwidth_gbps(device: int | str | torch.device | None = None) -> float | None:
    return get_device_info(device).peak_bw_gbps


def describe(device: int | str | torch.device | None = None) -> str:
    info = get_device_info(device)
    bw = f"{info.peak_bw_gbps:.0f} GB/s" if info.peak_bw_gbps else "unknown"
    return (
        f"{info.name} | cc {info.cc_str} | {info.sm_count} SMs | "
        f"{info.total_memory_gb:.1f} GiB | peak BW {bw} | "
        f"bf16={info.supports_bf16}"
    )
