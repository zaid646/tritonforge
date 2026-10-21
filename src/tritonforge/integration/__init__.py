"""Model integration: swap HF linear/norm layers for TritonForge kernels."""

from __future__ import annotations

from .qwen import QuantLinear, patch_model  # noqa: F401

__all__ = ["QuantLinear", "patch_model"]
