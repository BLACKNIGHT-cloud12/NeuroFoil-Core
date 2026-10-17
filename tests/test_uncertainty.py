"""
Unit tests for Aerodynamic Uncertainty Quantification and OOD Guardrails.
"""

import pytest
import numpy as np

from src.models.uncertainty import AerodynamicUncertaintyGuard, UncertaintyReport
from src.models.inference import AeroSurrogateEngine


def test_uncertainty_in_domain():
    """Verify that typical in-domain design has Certified status and tight intervals."""
    guard = AerodynamicUncertaintyGuard()
    beta = np.linspace(0, np.pi, 161)
    x_grid = 0.5 * (1.0 - np.cos(beta))
    cp_upper = -0.8 * np.ones_like(x_grid)
    cp_lower = 0.2 * np.ones_like(x_grid)

    report = guard.evaluate(
        m=0.02,
        p=0.40,
        t=0.08,
        alpha_deg=4.0,
        cl=0.45,
        cd=0.008,
        l_over_d=56.25,
        x_grid=x_grid,
        cp_upper=cp_upper,
        cp_lower=cp_lower,
    )

    assert isinstance(report, UncertaintyReport)
    assert report.status == "Certified (In-Domain)"
    assert not report.is_ood
    assert report.confidence_score > 75.0
    assert report.mahalanobis_dist < 2.5
    # CI intervals must bracket the mean
    assert report.cl_ci95[0] < 0.45 < report.cl_ci95[1]
    assert report.cd_ci95[0] < 0.008 < report.cd_ci95[1]
    # Cp CI low < mean < high
    assert np.all(report.cp_upper_ci95_low < cp_upper)
    assert np.all(report.cp_upper_ci95_high > cp_upper)


def test_uncertainty_ood_detection():
    """Verify that extreme out-of-domain geometry triggers OOD classification."""
    guard = AerodynamicUncertaintyGuard()
    beta = np.linspace(0, np.pi, 161)
    x_grid = 0.5 * (1.0 - np.cos(beta))
    cp_upper = -0.5 * np.ones_like(x_grid)
    cp_lower = 0.1 * np.ones_like(x_grid)

    # Angle of attack 22° is outside [-6, 14], thickness 0.25 is outside [0.06, 0.18]
    report = guard.evaluate(
        m=0.08,
        p=0.80,
        t=0.25,
        alpha_deg=22.0,
        cl=0.20,
        cd=0.05,
        l_over_d=4.0,
        x_grid=x_grid,
        cp_upper=cp_upper,
        cp_lower=cp_lower,
    )

    assert report.is_ood
    assert "Out-of-Distribution" in report.status
    assert len(report.warning_messages) >= 2
    assert report.confidence_score < 40.0


def test_engine_inference_with_uq_and_structures():
    """Verify that AeroSurrogateEngine returns UQ report and wingbox metrics."""
    engine = AeroSurrogateEngine()
    res = engine.predict(m=0.02, p=0.40, t=0.12, alpha_deg=4.0)

    assert "uq" in res
    assert isinstance(res["uq"], UncertaintyReport)
    assert "wingbox" in res
    assert res["wingbox"] is not None
    assert res["wingbox"].h_front_spar > 0.08
    assert res["wingbox"].fuel_tank_area > 0.03


def test_constrained_optimizer_with_structural_limits():
    """Verify that inverse design satisfies spar and fuel area constraints."""
    engine = AeroSurrogateEngine()
    opt = engine.optimize_airfoil(
        cl_target=0.50,
        t_min=0.10,
        t_max=0.16,
        min_front_spar=0.09,
        min_rear_spar=0.05,
        min_fuel_area=0.04,
        x0=(0.02, 0.40, 0.12, 4.0),
    )

    assert opt["success"]
    assert opt["structural_constraints_met"]
    wb = opt["opt_pred"]["wingbox"]
    assert wb.h_front_spar >= 0.09 - 1e-3
    assert wb.h_rear_spar >= 0.05 - 1e-3
    assert wb.fuel_tank_area >= 0.04 - 1e-3
