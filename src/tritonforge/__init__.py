"""TritonForge: custom Triton kernels for quantized LLM inference."""

from __future__ import annotations

__version__ = "0.1.0"

from .env import default_dtype, describe, get_device_info  # noqa: F401
from .quant import (  # noqa: F401
    dequantize_int4,
    quantize_dequantize,
    quantize_int4_group,
    unpack_int4,
)
from .reference import (  # noqa: F401
    max_abs_error,
    max_rel_error,
    quantized_linear_reference,
    rmsnorm_residual_reference,
    swiglu_reference,
)
from .svd import (  # noqa: F401
    approximation_error,
    compression_ratio,
    eigen_decompose,
    svd_low_rank,
)

__all__ = [
    "__version__",
    "default_dtype",
    "describe",
    "get_device_info",
    "dequantize_int4",
    "quantize_dequantize",
    "quantize_int4_group",
    "unpack_int4",
    "max_abs_error",
    "max_rel_error",
    "quantized_linear_reference",
    "rmsnorm_residual_reference",
    "swiglu_reference",
    "approximation_error",
    "compression_ratio",
    "eigen_decompose",
    "svd_low_rank",
]
