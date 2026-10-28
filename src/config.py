"""
Configuration loader, Windows-to-WSL path utilities, and solver invocation.

The OpenFOAM pipeline runs through WSL on Windows and natively on Linux;
`run_solver_command` picks the right subprocess invocation for the host OS.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_cfd.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config {path}, got {type(data)}")
    return data


def windows_to_wsl_path(path: Path | str) -> str:
    """Translate a Windows path to its WSL /mnt/... form; POSIX paths pass through."""
    p_str = Path(path).resolve().as_posix()
    if len(p_str) >= 2 and p_str[1] == ":":
        drive = p_str[0].lower()
        rest = p_str[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p_str


def run_solver_command(cmd: str, wsl_distro: str, timeout: int) -> subprocess.CompletedProcess:
    """
    Execute an OpenFOAM bash command string.

    Windows: run inside the configured WSL distro.
    Linux:   run natively (the distro setting is ignored).
    """
    if platform.system() == "Windows":
        argv = ["wsl", "-d", wsl_distro, "bash", "-c", cmd]
    else:
        argv = ["bash", "-c", cmd]
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )