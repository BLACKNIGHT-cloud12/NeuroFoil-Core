"""
Unit tests for Aerodynamic Uncertainty Quantification and OOD Guardrails.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.uncertainty import AerodynamicUncertaintyGuard, UncertaintyReport
from src.models.inference import AeroSurrogateEngine


def _write_openfoam_calibration(path, features: np.ndarray) -> None:
    frame = pd.DataFrame(
        features,
        columns=["m", "p", "t", "alpha_deg", "reynolds"],
    )
    frame["data_source"] = "openfoam"
    frame["status"] = "converged"
    frame["is_synthetic"] = False
    frame.to_csv(path, index=False)


@pytest.fixture
def uq_inputs():
    beta = np.linspace(0, np.pi, 161)
    x_grid = 0.5 * (1.0 - np.cos(beta))
    return {
        "m": 0.02,
        "p": 0.40,
        "t": 0.12,
        "alpha_deg": 4.0,
        "cl": 0.45,
        "cd": 0.008,
        "l_over_d": 56.25,
        "x_grid": x_grid,
        "cp_upper": -0.8 * np.ones_like(x_grid),
        "cp_lower": 0.2 * np.ones_like(x_grid),
        "reynolds": 3.5e6,
    }


def test_uncertainty_in_domain(tmp_path, uq_inputs):
    """A calibrated in-domain design reports an in-domain surrogate estimate."""
    dataset = tmp_path / "calibration.csv"
    rng = np.random.default_rng(7)
    half = rng.uniform(
        low=[0.001, 0.01, 0.001, 0.1, 1.0e4],
        high=[0.015, 0.20, 0.04, 4.0, 2.0e6],
        size=(20, 5),
    )
    center = np.array([
        uq_inputs["m"],
        uq_inputs["p"],
        uq_inputs["t"],
        uq_inputs["alpha_deg"],
        3.5e6,
    ])
    features = np.vstack([center + half, center - half])
    _write_openfoam_calibration(dataset, features)
    guard = AerodynamicUncertaintyGuard(dataset_path=dataset)
    report = guard.evaluate(**uq_inputs)

    assert isinstance(report, UncertaintyReport)
    assert report.status == "In-Domain Surrogate Estimate"
    assert not report.is_ood
    assert report.covariance_provenance == "empirical"
    assert report.cl_ci95[0] < uq_inputs["cl"] < report.cl_ci95[1]
    assert report.cd_ci95[0] < uq_inputs["cd"] < report.cd_ci95[1]
    assert np.all(report.cp_upper_ci95_low < uq_inputs["cp_upper"])
    assert np.all(report.cp_upper_ci95_high > uq_inputs["cp_upper"])


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
    assert report.status == "Outside Calibrated Domain"
    assert len(report.warning_messages) >= 2
    assert report.confidence_score < 40.0


def test_p_bound_marks_query_ood(uq_inputs):
    """The declared camber-position bound is enforced independently of distance."""
    guard = AerodynamicUncertaintyGuard(dataset_path="/definitely/missing.csv")
    report = guard.evaluate(**{**uq_inputs, "p": 0.9})

    assert report.is_ood
    assert report.status == "Outside Calibrated Domain"
    assert any("Camber position" in warning for warning in report.warning_messages)


def test_stall_warning_and_classification_share_threshold(tmp_path, uq_inputs):
    guard = AerodynamicUncertaintyGuard(dataset_path=tmp_path / "missing.csv")
    report = guard.evaluate(**{
        **uq_inputs,
        "m": 0.0275,
        "p": 0.327,
        "t": 0.08,
        "alpha_deg": 11.0,
    })

    has_stall_warning = any("stall risk" in warning for warning in report.warning_messages)
    assert has_stall_warning
    assert report.status == "Elevated Uncertainty (Adverse Gradient / Separation Risk)"


def test_fallback_covariance_is_disclosed(tmp_path, uq_inputs):
    guard = AerodynamicUncertaintyGuard(dataset_path=tmp_path / "missing.csv")
    report = guard.evaluate(**uq_inputs)

    assert report.covariance_provenance == "fallback"
    assert report.calibration_error is None
    assert any("fallback analytic covariance" in warning for warning in report.warning_messages)


def test_calibration_failure_is_reported_without_traceback(tmp_path, uq_inputs):
    broken = tmp_path / "broken.csv"
    broken.write_text("m,p,t\n0.02,0.4,0.12\n", encoding="utf-8")
    guard = AerodynamicUncertaintyGuard(dataset_path=broken)
    report = guard.evaluate(**uq_inputs)

    assert report.covariance_provenance == "fallback"
    assert report.calibration_error is not None
    assert "ValueError" in report.calibration_error
    assert "Traceback" not in report.calibration_error
    assert any("Calibration failed" in warning for warning in report.warning_messages)


def test_empirical_covariance_provenance(tmp_path, uq_inputs):
    dataset = tmp_path / "calibration.csv"
    rng = np.random.default_rng(7)
    features = np.column_stack([
        rng.uniform(0.0, 0.06, 20),
        rng.uniform(0.0, 0.60, 20),
        rng.uniform(0.06, 0.18, 20),
        rng.uniform(-6.0, 14.0, 20),
        rng.uniform(1.0e6, 6.0e6, 20),
    ])
    _write_openfoam_calibration(dataset, features)
    guard = AerodynamicUncertaintyGuard(dataset_path=dataset)
    report = guard.evaluate(**uq_inputs)

    assert report.covariance_provenance == "empirical"
    assert report.calibration_error is None
    assert not any("fallback analytic covariance" in warning for warning in report.warning_messages)


def test_empirical_multi_reynolds_distance_is_finite(tmp_path, uq_inputs):
    dataset = tmp_path / "multi_re.csv"
    rng = np.random.default_rng(19)
    features = np.column_stack([
        rng.uniform(0.0, 0.06, 80),
        rng.uniform(0.0, 0.60, 80),
        rng.uniform(0.06, 0.18, 80),
        rng.uniform(-6.0, 14.0, 80),
        rng.uniform(1.0e6, 6.0e6, 80),
    ])
    _write_openfoam_calibration(dataset, features)
    guard = AerodynamicUncertaintyGuard(dataset_path=dataset)

    low_re = guard.evaluate(**{**uq_inputs, "reynolds": 1.2e6})
    high_re = guard.evaluate(**{**uq_inputs, "reynolds": 5.8e6})

    assert np.isfinite(low_re.mahalanobis_dist)
    assert np.isfinite(high_re.mahalanobis_dist)
    assert low_re.mahalanobis_dist < 10.0
    assert high_re.mahalanobis_dist < 10.0


def test_synthetic_calibration_is_rejected(tmp_path, uq_inputs):
    dataset = tmp_path / "synthetic.csv"
    rng = np.random.default_rng(23)
    features = np.column_stack([
        rng.uniform(0.0, 0.06, 20),
        rng.uniform(0.0, 0.60, 20),
        rng.uniform(0.06, 0.18, 20),
        rng.uniform(-6.0, 14.0, 20),
        rng.uniform(1.0e6, 6.0e6, 20),
    ])
    frame = pd.DataFrame(
        features,
        columns=["m", "p", "t", "alpha_deg", "reynolds"],
    )
    frame["data_source"] = "synthetic_analytic"
    frame["status"] = "converged"
    frame["is_synthetic"] = True
    frame.to_csv(dataset, index=False)

    report = AerodynamicUncertaintyGuard(dataset_path=dataset).evaluate(**uq_inputs)

    assert report.covariance_provenance == "fallback"
    assert report.calibration_error is not None
    assert "openfoam" in report.calibration_error.lower()


def test_cm_conformal_interval_is_exact(tmp_path, uq_inputs):
    guard = AerodynamicUncertaintyGuard(
        dataset_path=tmp_path / "missing.csv",
        conformal={"q_cm": 0.01},
    )
    report = guard.evaluate(**uq_inputs, cm=-0.08)

    assert report.cm_std == pytest.approx(0.01 / 1.96)
    assert report.cm_ci95 == pytest.approx((-0.09, -0.07))


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
