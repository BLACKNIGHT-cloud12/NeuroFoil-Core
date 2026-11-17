"""
Uncertainty Quantification (UQ) and Out-of-Distribution (OOD) Guardrails.

Provides engineering-oriented epistemic uncertainty bounds, Mahalanobis envelope
distance metrics, and conformal prediction confidence intervals for aerodynamic
scalar coefficients and continuous surface pressure fields Cp(x).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np

from src.config import PROJECT_ROOT


STALL_STATUS_THRESHOLD = 0.5


@dataclass(frozen=True)
class UncertaintyReport:
    """Aerodynamic uncertainty and calibrated-domain assessment report."""

    status: str
    confidence_score: float  # 0.0 to 100.0%
    mahalanobis_dist: float
    is_ood: bool
    warning_messages: tuple[str, ...]
    covariance_provenance: str
    calibration_error: str | None

    # Scalar 95% confidence intervals
    cl_std: float
    cl_ci95: tuple[float, float]
    cd_std: float
    cd_ci95: tuple[float, float]
    cm_std: float
    cm_ci95: tuple[float, float]
    ld_std: float
    ld_ci95: tuple[float, float]

    # Surface Cp(x) standard error and 95% bands
    cp_upper_std: np.ndarray
    cp_upper_ci95_low: np.ndarray
    cp_upper_ci95_high: np.ndarray
    cp_lower_std: np.ndarray
    cp_lower_ci95_low: np.ndarray
    cp_lower_ci95_high: np.ndarray


class AerodynamicUncertaintyGuard:
    """
    Epistemic and aleatoric uncertainty quantifier with Mahalanobis OOD detection.

    When split-conformal calibration parameters are supplied (exported by the
    training notebook), 95% intervals are the calibrated conformal quantiles of
    held-out absolute residuals. Otherwise a conservative analytic fallback is
    used and the report should be treated as indicative only.
    """

    # Certified training domain bounds from the CFD campaign dataset
    BOUNDS = {
        "m": (0.00, 0.06),
        "p": (0.00, 0.60),
        "t": (0.06, 0.18),
        "alpha": (-6.0, 14.0),
        "reynolds": (1.0e6, 6.0e6),
    }

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        conformal: dict | None = None,
    ):
        if dataset_path is None:
            dataset_path = PROJECT_ROOT / "data" / "processed" / "airfoil_cfd_dataset.csv"

        self.dataset_path = Path(dataset_path)
        self.conformal = conformal or {}
        self.covariance_provenance = "fallback"
        self.calibration_error: str | None = None
        self.mean_vec, self.scale_vec, self.inv_cov = self._calibrate_distribution()

    @staticmethod
    def _format_calibration_error(exc: Exception) -> str:
        """Return a bounded, single-line calibration error without a traceback."""
        detail = " ".join(str(exc).split()) or "no error detail"
        detail = detail[:240]
        return f"{type(exc).__name__}: {detail}"

    def _calibrate_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the empirical feature centroid, per-feature scale, and inverse
        correlation matrix. Standardization keeps Reynolds number from making
        the covariance numerically ill-conditioned relative to geometry inputs.
        """
        if self.dataset_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(self.dataset_path)
                if "status" in df.columns:
                    df = df.loc[df["status"].eq("converged")].copy()
                if "data_source" not in df.columns:
                    raise ValueError(
                        "Empirical covariance requires an explicit data_source column."
                    )
                if df.empty or not df["data_source"].astype(str).str.lower().eq("openfoam").all():
                    raise ValueError(
                        "Empirical covariance requires converged data_source='openfoam' rows only."
                    )
                if "is_synthetic" in df.columns:
                    synthetic = df["is_synthetic"].astype(str).str.strip().str.lower()
                    if synthetic.isin({"true", "1", "yes", "y"}).any():
                        raise ValueError("Synthetic rows cannot calibrate the empirical covariance.")
                feature_cols = ["m", "p", "t", "alpha_deg"]
                if "reynolds" in df.columns and df["reynolds"].nunique() > 1:
                    feature_cols.append("reynolds")
                self.maha_features = feature_cols
                features = df[feature_cols].to_numpy(dtype=float)
                if len(features) < 2 or not np.all(np.isfinite(features)):
                    raise ValueError("Calibration data must contain at least two finite rows.")
                mean_vec = np.mean(features, axis=0)
                scale_vec = np.std(features, axis=0, ddof=0)
                if np.any(scale_vec <= 0.0):
                    raise ValueError("Calibration features must have non-zero variance.")
                normalized = (features - mean_vec) / scale_vec
                cov = np.atleast_2d(np.cov(normalized, rowvar=False))
                cov += np.eye(cov.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(cov)
                self.covariance_provenance = "empirical"
                return mean_vec, scale_vec, inv_cov
            except Exception as exc:
                self.calibration_error = self._format_calibration_error(exc)

        # Robust analytic fallback based on uniform Latin Hypercube / grid sampling
        self.maha_features = ["m", "p", "t", "alpha_deg"]
        mean_vec = np.array([0.0275, 0.3270, 0.0800, 3.9320])
        scale_vec = np.array([0.020, 0.196, 0.014, 6.036])
        inv_cov = np.eye(len(mean_vec))
        return mean_vec, scale_vec, inv_cov

    def evaluate(
        self,
        m: float,
        p: float,
        t: float,
        alpha_deg: float,
        cl: float,
        cd: float,
        l_over_d: float,
        x_grid: np.ndarray,
        cp_upper: np.ndarray,
        cp_lower: np.ndarray,
        reynolds: float = 6.0e6,
        cm: float = 0.0,
    ) -> UncertaintyReport:
        """
        Evaluate uncertainty, confidence intervals, and OOD status for a design query.
        """
        values = {"m": m, "p": p, "t": t, "alpha_deg": alpha_deg, "reynolds": reynolds}
        query = np.array([values[f] for f in self.maha_features], dtype=float)
        diff = (query - self.mean_vec) / self.scale_vec
        d_mahalanobis = float(np.sqrt(max(0.0, diff @ self.inv_cov @ diff)))

        # 1. Envelope and separation risk checks
        warnings: list[str] = []
        is_ood = False

        if self.covariance_provenance == "fallback":
            warning = "Using fallback analytic covariance; empirical manifold calibration is unavailable."
            if self.calibration_error is not None:
                warning += f" Calibration failed ({self.calibration_error})."
            warnings.append(warning)

        if m < self.BOUNDS["m"][0] or m > self.BOUNDS["m"][1]:
            warnings.append(f"Camber m={m:.3f} outside training envelope [0.00, 0.06].")
            is_ood = True
        if p < self.BOUNDS["p"][0] or p > self.BOUNDS["p"][1]:
            warnings.append(f"Camber position p={p:.3f} outside training envelope [0.00, 0.60].")
            is_ood = True
        if t < self.BOUNDS["t"][0] or t > self.BOUNDS["t"][1]:
            warnings.append(f"Thickness t={t:.3f} outside training envelope [0.06, 0.18].")
            is_ood = True
        if alpha_deg < self.BOUNDS["alpha"][0] or alpha_deg > self.BOUNDS["alpha"][1]:
            warnings.append(f"Angle of attack α={alpha_deg:.1f}° outside training envelope [-6°, 14°].")
            is_ood = True
        if reynolds < self.BOUNDS["reynolds"][0] or reynolds > self.BOUNDS["reynolds"][1]:
            warnings.append(f"Reynolds number Re={reynolds:.1e} outside training envelope [1e6, 6e6].")
            is_ood = True

        # Nonlinear stall proximity penalty
        stall_risk = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, abs(alpha_deg) - 10.5))))
        if stall_risk > STALL_STATUS_THRESHOLD:
            warnings.append(f"High separation/stall risk regime at α={alpha_deg:.1f}°.")

        # Determine calibrated-domain classification
        if is_ood or d_mahalanobis > 3.8:
            status = "Outside Calibrated Domain"
            is_ood = True
        elif d_mahalanobis > 2.5 or stall_risk > STALL_STATUS_THRESHOLD:
            status = "Elevated Uncertainty (Adverse Gradient / Separation Risk)"
        else:
            status = "In-Domain Surrogate Estimate"

        # Confidence score 0 - 100%
        conf_factor = math.exp(-0.18 * (d_mahalanobis**1.6)) * (1.0 - 0.45 * stall_risk)
        confidence_score = float(np.clip(conf_factor * 100.0, 5.0, 99.2))

        # 2. Scalar Standard Errors
        # Analytic baseline grows with OOD distance and stall risk.
        uq_multiplier = math.sqrt(1.0 + 0.25 * (d_mahalanobis**2) + 4.0 * (stall_risk**2))
        if {"q_cl", "q_cd"}.issubset(self.conformal):
            # Split-conformal 95% quantiles of held-out absolute residuals
            q_cl = float(self.conformal["q_cl"])
            q_cd = float(self.conformal["q_cd"])
            cl_std = q_cl / 1.96
            cd_std = q_cd / 1.96
        else:
            # Analytic fallback: baseline epistemic std grown with OOD distance and stall risk
            sigma_cl_base = 0.012
            sigma_cd_base = 0.00045
            cl_std = float(sigma_cl_base * uq_multiplier)
            cd_std = float(sigma_cd_base * uq_multiplier)

        if "q_cm" in self.conformal:
            q_cm = float(self.conformal["q_cm"])
            cm_std = q_cm / 1.96
            cm_ci95 = (float(cm - q_cm), float(cm + q_cm))
        else:
            sigma_cm_base = 0.006
            cm_std = float(sigma_cm_base * uq_multiplier)
            cm_ci95 = (float(cm - 1.96 * cm_std), float(cm + 1.96 * cm_std))

        # First-order uncertainty propagation for L/D: (L/D) * sqrt((s_L/L)^2 + (s_D/D)^2)
        rel_cl_var = (cl_std / max(abs(cl), 0.05)) ** 2
        rel_cd_var = (cd_std / max(cd, 1e-5)) ** 2
        ld_std = float(abs(l_over_d) * math.sqrt(rel_cl_var + rel_cd_var))

        cl_ci95 = (float(cl - 1.96 * cl_std), float(cl + 1.96 * cl_std))
        cd_ci95 = (float(max(1e-5, cd - 1.96 * cd_std)), float(cd + 1.96 * cd_std))
        ld_ci95 = (float(l_over_d - 1.96 * ld_std), float(l_over_d + 1.96 * ld_std))

        # 3. Continuous 1D Surface Pressure Cp(x) Uncertainty Bands
        x_safe = np.clip(x_grid, 1e-4, 1.0 - 1e-4)

        if {"q_cp_upper", "q_cp_lower"}.issubset(self.conformal):
            # Split-conformal per-point 95% quantiles over the chord
            q_cp_u = np.asarray(self.conformal["q_cp_upper"], dtype=float)
            q_cp_l = np.asarray(self.conformal["q_cp_lower"], dtype=float)
            cp_upper_std = q_cp_u / 1.96
            cp_lower_std = q_cp_l / 1.96
            cp_upper_ci95_low = cp_upper - q_cp_u
            cp_upper_ci95_high = cp_upper + q_cp_u
            cp_lower_ci95_low = cp_lower - q_cp_l
            cp_lower_ci95_high = cp_lower + q_cp_l
        else:
            # Suction peak high sensitivity near leading edge:
            le_factor = np.exp(-x_safe / 0.12) * (1.0 + 0.1 * abs(alpha_deg))
            # Adverse pressure recovery uncertainty near trailing edge:
            te_factor = np.clip((x_safe - 0.65) / 0.35, 0.0, 1.0) ** 2

            # Upper surface has higher epistemic variance under adverse gradients
            base_cp_u_std = 0.035 + 0.12 * le_factor + 0.08 * te_factor * (1.0 + stall_risk)
            cp_upper_std = base_cp_u_std * math.sqrt(1.0 + 0.20 * (d_mahalanobis**2))

            # Lower surface is mostly favorable pressure gradient, lower variance
            base_cp_l_std = 0.020 + 0.05 * le_factor + 0.03 * te_factor
            cp_lower_std = base_cp_l_std * math.sqrt(1.0 + 0.15 * (d_mahalanobis**2))

            cp_upper_ci95_low = cp_upper - 1.96 * cp_upper_std
            cp_upper_ci95_high = cp_upper + 1.96 * cp_upper_std
            cp_lower_ci95_low = cp_lower - 1.96 * cp_lower_std
            cp_lower_ci95_high = cp_lower + 1.96 * cp_lower_std

        return UncertaintyReport(
            status=status,
            confidence_score=confidence_score,
            mahalanobis_dist=d_mahalanobis,
            is_ood=is_ood,
            warning_messages=tuple(warnings),
            covariance_provenance=self.covariance_provenance,
            calibration_error=self.calibration_error,
            cl_std=cl_std,
            cl_ci95=cl_ci95,
            cd_std=cd_std,
            cd_ci95=cd_ci95,
            cm_std=cm_std,
            cm_ci95=cm_ci95,
            ld_std=ld_std,
            ld_ci95=ld_ci95,
            cp_upper_std=cp_upper_std,
            cp_upper_ci95_low=cp_upper_ci95_low,
            cp_upper_ci95_high=cp_upper_ci95_high,
            cp_lower_std=cp_lower_std,
            cp_lower_ci95_low=cp_lower_ci95_low,
            cp_lower_ci95_high=cp_lower_ci95_high,
        )
