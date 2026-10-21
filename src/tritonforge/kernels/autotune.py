"""Default and autotune configuration helpers shared by all kernels."""

from __future__ import annotations

from typing import TypedDict


class ConfigDict(TypedDict, total=False):
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    BLOCK_KH: int
    BLOCK: int
    num_warps: int
    num_stages: int


DEFAULT_DEQUANT_CONFIG: ConfigDict = {
    "BLOCK_M": 16,
    "BLOCK_N": 128,
    "BLOCK_KH": 64,
    "num_warps": 4,
    "num_stages": 4,
}

DEFAULT_DEQUANT_CONFIG_PREFILL: ConfigDict = {
    "BLOCK_M": 32,
    "BLOCK_N": 128,
    "BLOCK_KH": 32,
    "num_warps": 4,
    "num_stages": 4,
}

DEFAULT_RMSNORM_CONFIG: ConfigDict = {
    "BLOCK": 1024,
    "num_warps": 4,
    "num_stages": 1,
}

DEFAULT_SWIGLU_CONFIG: ConfigDict = {
    "BLOCK": 512,
    "num_warps": 4,
    "num_stages": 1,
}


def default_dequant_config(m: int, block_m_hint: int | None = None) -> ConfigDict:
    """A single, hand-tuned baseline config used to measure autotune gain."""
    if m <= 16:
        return dict(DEFAULT_DEQUANT_CONFIG)
    return dict(DEFAULT_DEQUANT_CONFIG_PREFILL)


def resolve(config: ConfigDict, base: ConfigDict) -> ConfigDict:
    merged: ConfigDict = dict(base)
    merged.update(config)
    return merged


def grid_1d(n_elements: int, block: int) -> tuple[int]:
    return (-(-n_elements // block),)
