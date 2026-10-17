"""
Unit tests for Phase I airfoil geometry generation, CST fitting, and validation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from src.geometry.naca import (
    evaluate_cst_surface,
    generate_naca4_airfoil,
    parse_naca4_code,
)
from src.geometry.validator import validate_airfoil_geometry


def test_parse_naca4_code_valid() -> None:
    m, p, t = parse_naca4_code("0012")
    assert m == pytest.approx(0.0)
    assert p == pytest.approx(0.0)
    assert t == pytest.approx(0.12)

    m2, p2, t2 = parse_naca4_code("NACA 2415")
    assert m2 == pytest.approx(0.02)
    assert p2 == pytest.approx(0.40)
    assert t2 == pytest.approx(0.15)


def test_parse_naca4_code_invalid() -> None:
    with pytest.raises(ValueError):
        parse_naca4_code("0412")  # m=0 with p=0.4 is invalid
    with pytest.raises(ValueError):
        parse_naca4_code("2012")  # m=0.02 with p=0 is invalid
    with pytest.raises(ValueError):
        parse_naca4_code("0000")  # zero thickness


def test_naca0012_symmetry_and_thickness() -> None:
    airfoil = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    assert airfoil.code == "NACA0012"
    assert np.allclose(airfoil.y_upper, -airfoil.y_lower, atol=1e-12)
    assert airfoil.te_gap == pytest.approx(0.0, abs=1e-12)
    assert airfoil.max_thickness_measured == pytest.approx(0.12, rel=5e-3)

    res = validate_airfoil_geometry(airfoil)
    assert res.is_valid, f"Unexpected validation errors: {res.errors}"


def test_cambered_naca2412_and_4412_validation() -> None:
    for code in ["2412", "4412", "2415"]:
        af = generate_naca4_airfoil(code, num_points=161, closed_te=True)
        res = validate_airfoil_geometry(af)
        assert res.is_valid, f"{code} failed validation: {res.errors}"
        assert af.max_camber_measured > 0.015


def test_cst_reconstruction_accuracy() -> None:
    # 5th-order Kulfan CST approximation of NACA 4-digit profiles has theoretical
    # chordwise RMSE ~ 2.8e-4 (< 0.04% chord, within wind-tunnel machining tolerance).
    airfoil = generate_naca4_airfoil("2412", num_points=161, closed_te=True, cst_order=5)
    yu_rec = evaluate_cst_surface(airfoil.x, airfoil.cst_upper)
    yl_rec = evaluate_cst_surface(airfoil.x, airfoil.cst_lower)

    rmse_u = float(np.sqrt(np.mean((airfoil.y_upper - yu_rec) ** 2)))
    rmse_l = float(np.sqrt(np.mean((airfoil.y_lower - yl_rec) ** 2)))
    assert rmse_u < 4e-4
    assert rmse_l < 4e-4


def test_validator_rejects_self_intersecting_and_out_of_bounds_shapes() -> None:
    airfoil = generate_naca4_airfoil("0012", num_points=101)
    # At index 50 (x=0.5), y_lower is -0.05286. Setting y_upper[50] = -0.08
    # forces a genuine upper/lower surface cross-over (yu < yl) and slope spike.
    bad_yu = airfoil.y_upper.copy()
    bad_yu[50] = -0.08
    corrupted = replace(airfoil, y_upper=bad_yu)

    res = validate_airfoil_geometry(corrupted)
    assert not res.is_valid
    assert any("Self-intersecting" in err for err in res.errors)
    assert any("slope discontinuity" in err for err in res.errors)

    # Too thin airfoil (NACA 0003 < 0.06 min_thickness_ratio)
    thin = generate_naca4_airfoil("0003", num_points=101)
    res_thin = validate_airfoil_geometry(thin, min_thickness_ratio=0.06)
    assert not res_thin.is_valid
    assert any("below minimum" in err for err in res_thin.errors)


def test_save_dat_and_ml_features(tmp_path: Path) -> None:
    airfoil = generate_naca4_airfoil("2412", num_points=81)
    dat_file = airfoil.save_dat(tmp_path / "NACA2412.dat")
    meta_file = airfoil.save_metadata(tmp_path / "NACA2412.json")

    assert dat_file.exists()
    assert meta_file.exists()

    feats = airfoil.sample_ml_features(n_sample=16)
    assert feats["airfoil_code"] == "NACA2412"
    assert "yu_00" in feats and "yl_15" in feats
    assert "cst_u_0" in feats