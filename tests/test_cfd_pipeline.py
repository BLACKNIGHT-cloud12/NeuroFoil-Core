"""
Unit tests for Milestone 1.3 CFD case builder, convergence evaluator, and extractor.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.config import load_config
from src.geometry.naca import generate_naca4_airfoil
from src.cfd.case_builder import OperatingPoint, build_openfoam_case
from src.cfd.convergence import (
    evaluate_convergence,
    parse_force_coeffs_file,
    parse_solver_log,
)
from src.cfd.extractor import extract_surface_cp


def test_operating_point_physics() -> None:
    op = OperatingPoint(alpha_deg=4.0, reynolds=6.0e6, u_inf=1.0, chord_m=1.0)
    assert op.alpha_deg == 4.0
    assert op.reynolds == 6.0e6
    assert op.nu == pytest.approx(1.666667e-7, rel=1e-4)

    ux, uy, uz = op.velocity_vector
    assert ux == pytest.approx(math.cos(math.radians(4.0)), rel=1e-5)
    assert uy == pytest.approx(math.sin(math.radians(4.0)), rel=1e-5)
    assert uz == 0.0

    lx, ly, lz = op.lift_dir
    dx, dy, dz = op.drag_dir
    # Dot product must be zero (perpendicular)
    assert (lx * dx + ly * dy + lz * dz) == pytest.approx(0.0, abs=1e-7)

    k_inf, omega_inf, nut_inf = op.turbulence_inlet_values(intensity=0.001, eddy_viscosity_ratio=0.1)
    assert k_inf == pytest.approx(1.5e-6)
    assert nut_inf == pytest.approx(0.1 * op.nu)
    assert omega_inf == pytest.approx(k_inf / nut_inf)


def test_build_openfoam_case_dictionaries(tmp_path: Path) -> None:
    cfg = load_config()
    op = OperatingPoint(alpha_deg=2.0, reynolds=3.0e6)
    build_openfoam_case(tmp_path, op, cfg)

    assert (tmp_path / "0" / "U").exists()
    assert (tmp_path / "0" / "p").exists()
    assert (tmp_path / "0" / "k").exists()
    assert (tmp_path / "0" / "omega").exists()
    assert (tmp_path / "0" / "nut").exists()
    assert (tmp_path / "constant" / "transportProperties").exists()
    assert (tmp_path / "constant" / "turbulenceProperties").exists()
    assert (tmp_path / "system" / "controlDict").exists()
    assert (tmp_path / "system" / "fvSchemes").exists()
    assert (tmp_path / "system" / "fvSolution").exists()

    u_txt = (tmp_path / "0" / "U").read_text(encoding="utf-8")
    assert "inlet" in u_txt and "outlet" in u_txt and "airfoil" in u_txt

    cd_txt = (tmp_path / "system" / "controlDict").read_text(encoding="utf-8")
    assert "forceCoeffs" in cd_txt
    assert "simpleFoam" in cd_txt


def test_convergence_evaluator_synthetic_data(tmp_path: Path) -> None:
    pp_dir = tmp_path / "postProcessing" / "forceCoeffs" / "0"
    pp_dir.mkdir(parents=True, exist_ok=True)
    coeff_file = pp_dir / "coefficient.dat"

    # Generate synthetic converging force coefficient history
    n_iters = 300
    times = np.arange(1, n_iters + 1)
    # Asymptotically converging to CL=0.45, CD=0.008, Cm=-0.001
    cl = 0.45 + 0.1 * np.exp(-times / 30.0) + 1e-5 * np.random.randn(n_iters)
    cd = 0.008 + 0.005 * np.exp(-times / 30.0) + 1e-6 * np.random.randn(n_iters)
    cm = -0.001 + 0.002 * np.exp(-times / 30.0) + 1e-5 * np.random.randn(n_iters)

    lines = ["# Time Cd Cs Cl CmRoll CmPitch CmYaw"]
    for t, d, l, m in zip(times, cd, cl, cm):
        lines.append(f"{int(t)} {d:.6f} 0.0 {l:.6f} 0.0 {m:.6f} 0.0")
    coeff_file.write_text("\n".join(lines), encoding="utf-8")

    log_file = tmp_path / "log.simpleFoam"
    log_lines = [f"Time = {n_iters}"]
    log_lines.append("Solving for p, Initial residual = 1e-6, Final residual = 1e-7, No Iterations 2")
    log_file.write_text("\n".join(log_lines), encoding="utf-8")

    res = evaluate_convergence(tmp_path, window_size=50)
    assert res.status == "converged"
    assert res.iterations_completed == n_iters
    assert res.cl_mean == pytest.approx(0.45, abs=1e-3)
    assert res.cd_mean == pytest.approx(0.008, abs=1e-3)
    assert res.cl_std < 5e-4
    assert res.cd_std < 2e-4


def test_extractor_missing_data(tmp_path: Path) -> None:
    airfoil = generate_naca4_airfoil("0012", num_points=51)
    cp_data = extract_surface_cp(tmp_path, airfoil)
    assert cp_data["available"] is False
    assert len(cp_data["x_grid"]) == 0
