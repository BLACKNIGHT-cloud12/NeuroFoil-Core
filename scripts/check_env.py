#!/usr/bin/env python3
"""
Milestone 0.1: Cross-Platform (Windows + WSL/Linux) Environment Inspector.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_PYTHON_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
    "matplotlib",
    "pytest",
    "pyyaml",
    "streamlit",
]

CFD_BINARIES = [
    "simpleFoam",
    "pimpleFoam",
    "foamRun",
    "blockMesh",
    "checkMesh",
    "snappyHexMesh",
    "foamToVTK",
    "gmsh",
]


def get_memory_info_gb() -> dict[str, float | None]:
    mem_info: dict[str, float | None] = {"total_gb": None, "available_gb": None}
    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            mem_info["total_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            mem_info["available_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass
    return mem_info


def inspect_wsl_from_windows() -> dict[str, Any]:
    wsl_path = shutil.which("wsl")
    if not wsl_path:
        return {"wsl_installed": False}

    info: dict[str, Any] = {"wsl_installed": True, "distros": [], "wsl_openfoam_probe": []}
    try:
        proc = subprocess.run(["wsl", "-l", "-v"], capture_output=True, timeout=10, check=False)
        raw_bytes = proc.stdout or b""
        text = (
            raw_bytes.decode("utf-16-le", errors="ignore")
            if b"\x00" in raw_bytes
            else raw_bytes.decode("utf-8", errors="ignore")
        )
        info["distros"] = [line.strip() for line in text.splitlines() if line.strip()]
    except Exception as exc:
        info["distros_error"] = str(exc)

    try:
        probe_cmd = (
            "ls -d /opt/*openfoam* /opt/*OpenFOAM* /usr/lib/openfoam/* 2>/dev/null || true; "
            "which simpleFoam blockMesh gmsh 2>/dev/null || true"
        )
        proc_probe = subprocess.run(
            ["wsl", "bash", "-ic", probe_cmd],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        out = (proc_probe.stdout or "").strip()
        info["wsl_openfoam_probe"] = out.splitlines() if out else []
    except Exception as exc:
        info["wsl_openfoam_probe_error"] = str(exc)

    return info


def check_python_packages() -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    for pkg in REQUIRED_PYTHON_PACKAGES:
        try:
            results[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            results[pkg] = None
    return results


def check_cfd_binaries() -> dict[str, dict[str, Any]]:
    return {
        cmd: {"available": shutil.which(cmd) is not None, "path": shutil.which(cmd)}
        for cmd in CFD_BINARIES
    }


def main() -> None:
    report = {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_logical_cores": os.cpu_count(),
            "memory": get_memory_info_gb(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "windows_wsl_bridge": inspect_wsl_from_windows(),
        "cfd_binaries_current_shell": check_cfd_binaries(),
        "python_packages": check_python_packages(),
    }
    print("=" * 72)
    print("ML-ACCELERATED AERODYNAMIC PLATFORM — ENVIRONMENT AUDIT (MILESTONE 0.1)")
    print("=" * 72)
    print(json.dumps(report, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()