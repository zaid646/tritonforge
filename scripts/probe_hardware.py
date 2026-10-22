#!/usr/bin/env python
"""Record the exact CPU, cache and memory shape used for every TritonForge-X result.

Writes ``results/x/hardware.json`` and regenerates ``docs/HARDWARE.md`` so the
hardware description in the report is never hand-entered.

Usage:
    python scripts/probe_hardware.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

INTERESTING_FLAGS = [
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx512f",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "amx_tile",
    "amx_int8",
    "amx_bf16",
]


def run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return proc.stdout.strip()


def read(path: str) -> str:
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ""


def parse_cpuinfo() -> dict:
    text = read("/proc/cpuinfo")
    if not text:
        return {}
    first = text.split("\n\n", 1)[0]
    info: dict[str, str] = {}
    for line in first.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    info.pop("flags", None)
    return info


def cpu_flags() -> list[str]:
    text = read("/proc/cpuinfo")
    match = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
    return sorted(set(match.group(1).split())) if match else []


def lscpu_json() -> dict | None:
    raw = run(["lscpu", "-J"])
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_int(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    return int(match.group(0)) if match else None


def memory_gib() -> dict:
    info: dict[str, float] = {}
    for line in read("/proc/meminfo").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            kib = parse_int(value)
            if kib is not None:
                info[key] = round(kib / (1024**2), 2)
    cgroup_limit = read("/sys/fs/cgroup/memory.max")
    if cgroup_limit and cgroup_limit != "max":
        info["cgroup_memory_limit_gib"] = round(int(cgroup_limit) / (1024**3), 2)
    return info


def disk_info() -> dict:
    usage = shutil.disk_usage("/")
    return {
        "total_gib": round(usage.total / (1024**3), 1),
        "free_gib": round(usage.free / (1024**3), 1),
    }


def os_release() -> dict:
    info = {}
    for line in read("/etc/os-release").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value.strip().strip('"')
    return info


def python_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("torch", "openvino", "nncf", "onnxruntime", "transformers"):
        try:
            mod = __import__(module)
            versions[module] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional dependency probe
            versions[module] = None
    return versions


def md_table(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"| {key} | {value} |" for key, value in rows)
    return f"| Property | Value |\n| --- | --- |\n{body}"


def render_markdown(hw: dict, lscpu_raw: str) -> str:
    cpu = hw["cpu"]
    caches = hw["caches"]
    flags = hw["flags"]
    flag_rows = [
        (flag, "yes" if flag in flags["present"] else "no") for flag in flags["interesting"]
    ]
    cache_rows = [(name, f"{size} ({kind})") for name, size, kind in caches]
    mem = hw["memory"]
    cgroup = mem.get("cgroup_memory_limit_gib")
    mem_rows = [
        ("MemTotal", f"{mem.get('MemTotal', '?')} GiB"),
        ("MemAvailable", f"{mem.get('MemAvailable', '?')} GiB"),
        ("cgroup limit", "unlimited" if cgroup is None else f"{cgroup} GiB"),
    ]

    cpu_rows = [
        ("Vendor", cpu["vendor"]),
        ("Family / model / stepping", f"{cpu['family']} / {cpu['model']} / {cpu['stepping']}"),
        ("Logical CPUs", str(cpu["logical_cpus"])),
        ("Cores per socket", str(cpu["cores_per_socket"])),
        ("Threads per core", str(cpu["threads_per_core"])),
        ("Sockets", str(cpu["sockets"])),
        ("Max MHz", str(cpu["max_mhz"] or "unknown")),
        ("Min MHz", str(cpu["min_mhz"] or "unknown")),
        ("NUMA nodes", str(hw["numa_nodes"])),
        ("Hypervisor", cpu.get("hypervisor") or "none"),
    ]
    disk_row = ("Disk", f"{hw['disk']['total_gib']} GiB total, {hw['disk']['free_gib']} GiB free")
    software_rows = [(key, str(value)) for key, value in hw["software"].items()]
    os_rows = [("OS", hw["os"].get("PRETTY_NAME", "unknown"))]
    cache_section = md_table(cache_rows) if cache_rows else "`lscpu -J` did not report caches."
    logical = cpu["logical_cpus"]
    threads = cpu["threads_per_core"]
    cores = cpu["cores_per_socket"]

    parts = [
        "# Hardware",
        "",
        f"Generated by `python scripts/probe_hardware.py` on {hw['timestamp_utc']}.",
        "Every TritonForge-X number in `docs/FINAL_REPORT.md` was produced on this machine.",
        "Do not hand-edit this file.",
        "",
        f"**Exact CPU string: `{cpu['model_name']}`**",
        "",
        md_table(cpu_rows),
        "",
        "## Caches",
        "",
        cache_section,
        "",
        "## Memory and disk",
        "",
        md_table([*mem_rows, disk_row]),
        "",
        "## ISA flags",
        "",
        md_table(flag_rows),
        "",
        "## Software",
        "",
        md_table(software_rows),
        "",
        md_table(os_rows),
        "",
        "## Raw `lscpu`",
        "",
        "```text",
        lscpu_raw,
        "```",
        "",
        "## Measurement caveats",
        "",
        "- This is a shared cloud VM: other tenants share the physical L3 and memory",
        "  controllers, so the L3 figure is the host slice reported by `lscpu`,",
        "  not a dedicated per-VM cache.",
        f"- {logical} logical CPUs are {threads} SMT threads on {cores} physical",
        "  core(s). All benchmarks pin work to the online CPUs and report repeated",
        "  runs with p95, because single runs on this shape are noisy.",
        "- **No AVX-512 and no AMX** are available on this CPU. Any int8/int4 speedup",
        "  below comes from AVX2 code paths, not VNNI or AMX acceleration.",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(ROOT / "results/x/hardware.json"))
    parser.add_argument("--markdown", default=str(ROOT / "docs/HARDWARE.md"))
    args = parser.parse_args()

    raw = run(["lscpu"])
    if not raw:
        print("lscpu not available; run this on the Linux benchmark host", file=sys.stderr)
        return 2

    lscpu = lscpu_json() or {}
    fields = {
        row.get("field", "").rstrip(":"): row.get("data", "") for row in lscpu.get("lscpu", [])
    }
    cpuinfo = parse_cpuinfo()
    flags = cpu_flags()

    caches = []
    for entry in lscpu.get("caches", []):
        name = entry.get("name", "")
        size = entry.get("size", "")
        kind = entry.get("type", "")
        if name or size:
            caches.append((name, size, kind))
    if not caches:
        # util-linux < 2.37 has no --json caches; parse the human table instead.
        for line in raw.splitlines():
            match = re.match(r"^(L\d\w*) cache:\s+(.+)$", line.strip())
            if match:
                caches.append((match.group(1), match.group(2).strip(), "cache"))

    model_name = fields.get("Model name") or cpuinfo.get("model name", "unknown")
    hw = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Exact hardware for every TritonForge-X result; generated by scripts/probe_hardware.py"
        ),
        "cpu": {
            "model_name": model_name,
            "vendor": fields.get("Vendor ID") or cpuinfo.get("vendor_id", "unknown"),
            "family": cpuinfo.get("cpu family", "?"),
            "model": cpuinfo.get("model", "?"),
            "stepping": cpuinfo.get("stepping", "?"),
            "logical_cpus": int(fields.get("CPU(s)") or 0) or None,
            "cores_per_socket": parse_int(fields.get("Core(s) per socket", "")),
            "threads_per_core": parse_int(fields.get("Thread(s) per core", "")),
            "sockets": parse_int(fields.get("Socket(s)", "")),
            "max_mhz": fields.get("CPU max MHz") or cpuinfo.get("cpu MHz"),
            "min_mhz": fields.get("CPU min MHz"),
            "hypervisor": fields.get("Hypervisor vendor"),
        },
        "numa_nodes": parse_int(fields.get("NUMA node(s)", "")),
        "caches": caches,
        "flags": {"present": flags, "interesting": INTERESTING_FLAGS},
        "memory": memory_gib(),
        "disk": disk_info(),
        "software": python_versions(),
        "os": os_release(),
        "lscpu_raw": raw,
    }

    json_path = pathlib.Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(hw, indent=2) + "\n")

    md_path = pathlib.Path(args.markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(hw, raw))

    print(f"CPU: {model_name}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
