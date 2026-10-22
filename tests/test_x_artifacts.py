"""Schema and honesty checks for the committed TritonForge-X artifacts.

These tests do not download a model or run a benchmark. They make sure the JSON
that feeds the report is internally consistent and that every row cites the same
measured CPU.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/x"


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{path} not collected")
    return json.loads(path.read_text())


def test_hardware_is_the_intel_cpu():
    hw = load("hardware.json")
    cpu = hw["cpu"]
    assert "Intel" in cpu["model_name"]
    assert cpu["logical_cpus"] >= 1
    assert cpu["cores_per_socket"] is not None
    assert cpu["threads_per_core"] is not None
    flags = hw["flags"]["present"]
    assert "avx2" in flags
    assert "avx512f" not in flags
    assert "amx_tile" not in flags


def test_models_and_commands():
    models = load("models.json")
    rows = {row["variant"]: row for row in models["variants"]}
    for variant in ("fp16", "int8", "int4_g128", "int4_awq_g128"):
        assert variant in rows
        assert rows[variant]["ir_weights_mb"] > 0
    assert "--group-size 128" in rows["int4_g128"]["command"]
    assert rows["int4_awq_g128"]["data_aware"] is True
    assert "--awq" in rows["int4_awq_g128"]["command"]
    assert "--dataset" in rows["int4_awq_g128"]["command"]
    assert rows["fp16"]["ir_weights_mb"] > rows["int8"]["ir_weights_mb"]
    assert rows["int8"]["ir_weights_mb"] > rows["int4_g128"]["ir_weights_mb"]


def test_validation_coverage_and_direction():
    validation = load("validation.json")
    assert validation["prompts"] >= 50
    assert validation["wikitext"]["tokens_scored"] >= 4096
    variants = {row["variant"]: row for row in validation["variants"]}
    assert set(variants) >= {"fp16", "int8", "int4_g128", "int4_awq_g128"}
    for row in variants.values():
        assert 0.0 <= row["top1_agreement_pct"] <= 100.0
        assert row["max_abs_err"] >= 0.0
        assert row["ppl"] > 1.0
    assert variants["fp16"]["top1_agreement_pct"] >= 99.0
    assert variants["int8"]["top1_agreement_pct"] >= 90.0
    assert variants["int8"]["ppl_delta_pct"] <= 1.0
    assert variants["int4_awq_g128"]["ppl_delta_pct"] <= variants["int4_g128"]["ppl_delta_pct"]


def test_bench_rows_are_complete():
    hardware = load("hardware.json")["cpu"]["model_name"]
    files = sorted((RESULTS / "bench").glob("*_b*.json"))
    if not files:
        pytest.skip("bench results not collected")
    for path in files:
        row = json.loads(path.read_text())
        assert row["hardware_cpu"] == hardware
        assert row["runs"] >= 10
        assert len(row["runs_detail"]) == row["runs"]
        assert row["tokens_per_s_mean"] > 0
        assert row["ttft_ms_mean"] > 0
        assert row["itl_ms_mean"] > 0
        assert row["peak_rss_mib"] > 0
        assert row["tokens_per_s_p95"] >= row["tokens_per_s_mean"] or row["runs"] < 5


def test_sycl_parity_and_bandwidth():
    sycl = load("sycl.json")
    assert sycl["all_validation_passed"] is True
    assert sycl["stream"]["peak_gbps"] > 0
    for op in ("swiglu", "rmsnorm_residual"):
        best = sycl["best"][op]
        assert best["gbps"] > 0
        assert 0 < best["pct_measured_peak_bw"] <= 100.0
        assert best["work_group"] in (64, 128, 256, 512)
    for row in sycl["validation"]:
        assert row["max_abs_err"] <= row["tol_max_abs"]
        assert row["max_rel_err"] <= row["tol_max_rel"]
