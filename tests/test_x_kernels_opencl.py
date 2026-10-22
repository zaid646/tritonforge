"""CPU-only parity test for the fused kernels, using POCL instead of oneAPI.

CI builds the OpenCL C port in tests/opencl_kernels and runs this test on a
plain ubuntu-latest runner. The reported SYCL numbers come from the DPC++ build
in sycl/tfx_sycl.cpp; this test guards the same math in CI.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tests/opencl_kernels"
BIN = ROOT / "build/opencl_parity"


@pytest.fixture(scope="session")
def binary() -> pathlib.Path:
    if shutil.which("g++") is None or shutil.which("clinfo") is None:
        pytest.skip("g++/clinfo not available")
    proc = subprocess.run(["clinfo", "-l"], capture_output=True, text=True)
    if "Platform" not in proc.stdout:
        pytest.skip("no OpenCL platform (install pocl-opencl-icd)")
    BIN.parent.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(
        ["g++", "-O2", "-std=c++17", str(SRC / "parity.cpp"), "-o", str(BIN), "-lOpenCL"],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.fail(f"OpenCL parity harness failed to build:\n{build.stderr}")
    return BIN


def run(
    binary: pathlib.Path, kernel: str, m: int, n: int, wg: int, tensors: dict, tmp
) -> torch.Tensor:
    paths = {}
    for name, tensor in tensors.items():
        path = tmp / f"{name}.bin"
        tensor.detach().float().contiguous().numpy().tofile(path)
        paths[name] = path
    out_path = tmp / "out.bin"
    command = [
        str(binary),
        "--source",
        str(SRC / "tfx_kernels.cl"),
        "--kernel",
        kernel,
        "--m",
        str(m),
        "--n",
        str(n),
        "--wg",
        str(wg),
        "--in0",
        str(paths["in0"]),
        "--in1",
        str(paths["in1"]),
        "--out",
        str(out_path),
    ]
    if "in2" in paths:
        command += ["--in2", str(paths["in2"])]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"OpenCL kernel failed:\n{proc.stdout}\n{proc.stderr}")
    return torch.from_numpy(np.fromfile(out_path, dtype=np.float32).reshape(m, n).copy())


@pytest.mark.parametrize("m,n,wg", [(3, 895, 128), (1, 896, 256), (17, 1000, 128)])
def test_swiglu_matches_pytorch(binary, m, n, wg, tmp_path):
    torch.manual_seed(0)
    gate = torch.empty(m, n).uniform_(-1, 1)
    up = torch.empty(m, n).uniform_(-1, 1)
    out = run(binary, "swiglu", m, n, wg, {"in0": gate, "in1": up}, tmp_path)
    expected = (gate * torch.sigmoid(gate)) * up
    assert (out - expected).abs().max().item() <= 2e-5


@pytest.mark.parametrize("m,n,wg", [(7, 1000, 128), (1, 896, 256), (3, 895, 64)])
def test_rmsnorm_residual_matches_pytorch(binary, m, n, wg, tmp_path):
    torch.manual_seed(1)
    x = torch.empty(m, n).uniform_(-1, 1)
    residual = torch.empty(m, n).uniform_(-1, 1)
    weight = torch.empty(n).uniform_(0.5, 1.0)
    out = run(
        binary,
        "rmsnorm_residual",
        m,
        n,
        wg,
        {"in0": x, "in1": residual, "in2": weight},
        tmp_path,
    )
    summed = x + residual
    variance = summed.pow(2).mean(-1, keepdim=True)
    expected = summed * torch.rsqrt(variance + 1e-6) * weight
    assert (out - expected).abs().max().item() <= 2e-5
