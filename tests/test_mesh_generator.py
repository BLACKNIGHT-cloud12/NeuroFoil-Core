"""
Unit and WSL OpenFOAM integration tests for the 2D 6-block C-grid mesh generator.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import pytest

from src.config import PROJECT_ROOT, load_config
from src.cfd.mesh_generator import (
    generate_blockmesh_dict,
    run_blockmesh_and_checkmesh,
)
from src.geometry.naca import generate_naca4_airfoil


def test_generate_blockmesh_dict_structure(tmp_path: Path) -> None:
    airfoil = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    meta = generate_blockmesh_dict(
        airfoil=airfoil,
        case_dir=tmp_path,
        level="coarse",
        n_airfoil_side=60,
        n_wake=40,
        n_normal=45,
    )
    bm_path = Path(meta["dict_path"])
    assert bm_path.exists()
    text = bm_path.read_text(encoding="utf-8")
    assert "inlet" in text
    assert "outlet" in text
    assert "airfoil" in text
    assert "frontAndBack" in text
    assert meta["expected_cells"] == 2 * (meta["n_nose"] + meta["n_aft"] + 40) * 45


def test_wsl_blockmesh_and_checkmesh_integration() -> None:
    """
    Generate a medium 2D C-grid for NACA 0012 inside data/raw_cases/_mesh_sanity_check
    and verify OpenFOAM v2606 blockMesh + checkMesh pass with zero errors.
    """
    cfg = load_config()
    mesh_cfg = cfg["mesh"]
    med = mesh_cfg["levels"]["medium"]

    case_dir = PROJECT_ROOT / "data" / "raw_cases" / "_mesh_sanity_check"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    airfoil = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    meta = generate_blockmesh_dict(
        airfoil=airfoil,
        case_dir=case_dir,
        level="medium",
        farfield_radius=float(mesh_cfg["farfield_radius_chords"]),
        wake_length=float(mesh_cfg["wake_length_chords"]),
        span_thickness=float(mesh_cfg["span_thickness_m"]),
        n_airfoil_side=int(med["n_airfoil_side"]),
        n_wake=int(med["n_wake"]),
        n_normal=int(med["n_normal"]),
    )

    stats = run_blockmesh_and_checkmesh(
        case_dir=case_dir,
        meta=meta,
        wsl_distro=cfg["wsl_environment"]["distro"],
        openfoam_bashrc=cfg["wsl_environment"]["openfoam_bashrc"],
    )

    print(
        f"\n[checkMesh Medium NACA0012] cells={stats.n_cells}, "
        f"max_non_ortho={stats.max_non_orthogonality_deg:.2f} deg, "
        f"avg_non_ortho={stats.avg_non_orthogonality_deg:.2f} deg, "
        f"max_skewness={stats.max_skewness:.3f}, mesh_ok={stats.mesh_ok}"
    )

    assert stats.mesh_ok is True
    assert stats.n_cells == meta["expected_cells"]
    assert stats.max_non_orthogonality_deg < 68.0
    assert stats.max_skewness < 4.0