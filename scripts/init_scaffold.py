#!/usr/bin/env python3
"""
Milestone 0.2: Repository Scaffold Initializer & WSL OpenFOAM v2606 Bridge Check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIRECTORIES = [
    "configs",
    "data/benchmarks",
    "data/raw_cases",
    "data/processed",
    "src/geometry",
    "src/cfd",
    "src/dataset",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/optimization",
    "src/reporting",
    "tests",
    "scripts",
    "app",
    "reports/figures",
    "docs",
]

PACKAGE_INIT_DIRS = [
    "src",
    "src/geometry",
    "src/cfd",
    "src/dataset",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/optimization",
    "src/reporting",
]


def test_wsl_openfoam_bridge() -> dict[str, str | int]:
    """Verify that Windows Python can invoke OpenFOAM v2606 inside Ubuntu-24.04."""
    cmd = (
        "source /usr/lib/openfoam/openfoam2606/etc/bashrc && "
        "echo WM_PROJECT_VERSION=$WM_PROJECT_VERSION && "
        "which simpleFoam blockMesh checkMesh"
    )
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu-24.04", "bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def main() -> None:
    root = PROJECT_ROOT
    for rel_dir in DIRECTORIES:
        (root / rel_dir).mkdir(parents=True, exist_ok=True)

    for pkg_dir in PACKAGE_INIT_DIRS:
        init_file = root / pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n', encoding="utf-8")

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# ML-Accelerated Aerodynamic Design and Validation Platform\n\n"
            "Automated OpenFOAM-to-SciML workflow for 2D airfoil performance prediction "
            "and CFD-verified design exploration.\n",
            encoding="utf-8",
        )

    from src.config import load_config, windows_to_wsl_path

    cfg = load_config()
    wsl_cwd = windows_to_wsl_path(root)
    bridge = test_wsl_openfoam_bridge()

    print("=" * 72)
    print("MILESTONE 0.2 — REPOSITORY SCAFFOLD & WSL BRIDGE VERIFICATION")
    print("=" * 72)
    print(f"Created directories      : {len(DIRECTORIES)} folders initialized")
    print(f"Loaded config project    : {cfg['project']['name']}")
    print(f"Solver & Turbulence      : {cfg['solver']['executable']} ({cfg['solver']['turbulence_model']})")
    print(f"WSL mapped project root  : {wsl_cwd}")
    print(f"WSL bridge return code   : {bridge['returncode']}")
    print("WSL bridge output:")
    for line in str(bridge["stdout"]).splitlines():
        print(f"  {line}")
    if bridge["stderr"]:
        print("WSL bridge stderr:")
        print(f"  {bridge['stderr']}")
    print("=" * 72)


if __name__ == "__main__":
    main()