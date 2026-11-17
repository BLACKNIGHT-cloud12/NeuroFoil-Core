"""
Neural Surrogate Inference Engine (pure NumPy runtime).

Loads the PyTorch-trained surrogate weights exported by the Kaggle training
notebook (models/surrogate_weights.npz + models/surrogate_meta.json) and runs
sub-millisecond forward passes without requiring PyTorch at runtime.

Two backends:
- "neural": trained residual MLP (scalar coefficients) + field decoder
  (surface Cp), loaded from exported weights.
- "analytic_fallback": closed-form thin-airfoil model used only when trained
  weights are absent (e.g. fresh checkout before the first training run).

The active backend is reported in every predict() result under
"model_backend" so downstream users know which model produced the numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from src.config import PROJECT_ROOT
from src.geometry.naca import AirfoilGeometry, generate_naca4_airfoil
from src.geometry.structures import compute_wingbox_properties, WingboxProperties
from src.models.uncertainty import AerodynamicUncertaintyGuard, UncertaintyReport

WEIGHTS_FILE = "surrogate_weights.npz"
META_FILE = "surrogate_meta.json"

# Canonical feature order; must match the training notebook.
DEFAULT_FEATURE_ORDER = [
    "m", "p", "t",
    "cst_u_0", "cst_u_1", "cst_u_2", "cst_u_3", "cst_u_4", "cst_u_5",
    "cst_l_0", "cst_l_1", "cst_l_2", "cst_l_3", "cst_l_4", "cst_l_5",
    "alpha_deg", "reynolds",
]
TARGET_ORDER = ["cl", "cd", "cm"]


def _gelu(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _batchnorm(x: np.ndarray, mean: np.ndarray, var: np.ndarray,
               gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class AeroSurrogateEngine:
    """Trained surrogate inference engine with UQ and aero-structural coupling."""

    def __init__(self, models_dir: Path | str = PROJECT_ROOT / "models"):
        self.models_dir = Path(models_dir)
        self.backend = "analytic_fallback"
        self.model_provenance = "analytic_fallback"
        self.meta: dict[str, Any] = {}

        # Standardization coefficients
        _SCALER_FILES = ("scaler_x_mean.npy", "scaler_x_scale.npy",
                         "scaler_y_mean.npy", "scaler_y_scale.npy")
        self.has_scalers = all((self.models_dir / f).exists() for f in _SCALER_FILES)
        if self.has_scalers:
            self.x_mean = np.load(self.models_dir / "scaler_x_mean.npy")
            self.x_scale = np.load(self.models_dir / "scaler_x_scale.npy")
            self.y_mean = np.load(self.models_dir / "scaler_y_mean.npy")
            self.y_scale = np.load(self.models_dir / "scaler_y_scale.npy")
            if self.x_mean.shape != self.x_scale.shape:
                raise ValueError("Input scaler mean/scale shapes do not match.")
            if self.y_mean.shape != self.y_scale.shape:
                raise ValueError("Output scaler mean/scale shapes do not match.")
            if np.any(self.x_scale <= 0.0) or np.any(self.y_scale <= 0.0):
                raise ValueError("Scaler scale arrays must contain only positive values.")
        else:
            self.x_mean = self.x_scale = self.y_mean = self.y_scale = None

        # The field decoder is trained on standardized Cp targets. Both field
        # scaler files are therefore required before its output can be exposed
        # as physical Cp; an incomplete pair is an invalid model bundle.
        self.field_mean = None
        self.field_scale = None
        _fmean = self.models_dir / "scaler_field_mean.npy"
        _fscale = self.models_dir / "scaler_field_scale.npy"
        if _fmean.exists() != _fscale.exists():
            raise ValueError(
                "Incomplete Cp field scaler: expected both scaler_field_mean.npy "
                "and scaler_field_scale.npy."
            )
        if _fmean.exists():
            self.field_mean = np.load(_fmean)
            self.field_scale = np.load(_fscale)
            if self.field_mean.shape != self.field_scale.shape:
                raise ValueError("Cp field scaler mean/scale shapes do not match.")
            if np.any(self.field_scale <= 0.0):
                raise ValueError("Cp field scaler scale values must be positive.")

        # Trained neural weights (exported by the training notebook)
        weights_path = self.models_dir / WEIGHTS_FILE
        meta_path = self.models_dir / META_FILE
        if weights_path.exists():
            self.weights = dict(np.load(weights_path, allow_pickle=False))
            if meta_path.exists():
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_order = list(self.meta.get("feature_order", DEFAULT_FEATURE_ORDER))
            if self.has_scalers:
                self.backend = "neural"
                self.model_provenance = str(
                    self.meta.get("data_source", "unverified_legacy_artifact")
                )

        if self.backend == "neural":
            if self.x_mean.shape != (len(self.feature_order),):
                raise ValueError(
                    "Input scaler length does not match the model feature order "
                    f"({self.x_mean.shape} vs {len(self.feature_order)})."
                )
            if self.y_mean.shape != (len(TARGET_ORDER),):
                raise ValueError(
                    "Output scaler length does not match the target order "
                    f"({self.y_mean.shape} vs {len(TARGET_ORDER)})."
                )
            field_width = int(self.weights["f_l4_b"].shape[0])
            if self.field_mean is not None and self.field_mean.shape != (field_width,):
                raise ValueError(
                    "Cp field scaler length does not match the decoder output "
                    f"({self.field_mean.shape} vs {field_width})."
                )

        conformal = self.meta.get("conformal") if self.backend == "neural" else None
        self.uq_guard = AerodynamicUncertaintyGuard(conformal=conformal)
        # Legacy bundles omitted the Cp target scaler. Keep calibrated scalar
        # intervals, but drop incompatible raw-unit Cp quantiles while an
        # analytic Cp fallback is used.
        scalar_conformal = None
        if conformal:
            scalar_conformal = {
                key: value for key, value in conformal.items()
                if key not in {"q_cp_upper", "q_cp_lower"}
            }
        self.field_fallback_uq_guard = AerodynamicUncertaintyGuard(conformal=scalar_conformal)
        # Fully analytic predictions — including the neural-weights-but-bad-geometry
        # path — must never inherit neural conformal intervals or neural provenance.
        self.analytic_uq_guard = AerodynamicUncertaintyGuard(
            dataset_path=self.models_dir / "__analytic_no_calibration__"
        )

    # ------------------------------------------------------------------
    # Neural forward passes (numpy)
    # ------------------------------------------------------------------
    def _forward_scalars(self, x: np.ndarray) -> np.ndarray:
        """Residual MLP forward pass; input (B, F) standardized, output (B, 3)."""
        w = self.weights
        h = _gelu(_batchnorm(
            _linear(x, w["s_in_w"], w["s_in_b"]),
            w["s_in_bn_m"], w["s_in_bn_v"], w["s_in_bn_g"], w["s_in_bn_b"],
        ))
        for r in ("s_r1", "s_r2"):
            z = _linear(h, w[f"{r}_fc_w"], w[f"{r}_fc_b"])
            z = _gelu(_batchnorm(z, w[f"{r}_bn1_m"], w[f"{r}_bn1_v"], w[f"{r}_bn1_g"], w[f"{r}_bn1_b"]))
            z = _linear(z, w[f"{r}_fc2_w"], w[f"{r}_fc2_b"])
            z = _batchnorm(z, w[f"{r}_bn2_m"], w[f"{r}_bn2_v"], w[f"{r}_bn2_g"], w[f"{r}_bn2_b"])
            h = _gelu(h + z)
        h = _gelu(_linear(h, w["s_h1_w"], w["s_h1_b"]))
        return _linear(h, w["s_out_w"], w["s_out_b"])

    def _forward_field(self, x: np.ndarray) -> np.ndarray:
        """Field decoder forward pass; output (B, 322) = [cp_upper(161), cp_lower(161)]."""
        w = self.weights
        h = _gelu(_linear(x, w["f_l1_w"], w["f_l1_b"]))
        h = _gelu(_linear(h, w["f_l2_w"], w["f_l2_b"]))
        h = _gelu(_linear(h, w["f_l3_w"], w["f_l3_b"]))
        return _linear(h, w["f_l4_w"], w["f_l4_b"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        m: float,
        p: float,
        t: float,
        alpha_deg: float,
        reynolds: float = 6.0e6,
        compute_structures: bool = True,
    ) -> dict[str, Any]:
        """
        Predict aerodynamic coefficients, surface Cp distribution, UQ intervals,
        and structural metrics with the active backend.
        """
        m_digit = int(round(m * 100))
        p_digit = int(round(p * 10)) if m > 0 else 0
        t_digit = int(round(t * 100))
        naca_code = f"{m_digit:1d}{p_digit:1d}{t_digit:02d}"

        try:
            airfoil = generate_naca4_airfoil(
                naca_code,
                num_points=161,
                closed_te=True,
                m=m,
                p=p,
                t=t,
            )
        except (TypeError, ValueError):
            airfoil = None

        if self.backend == "neural" and airfoil is not None:
            result = self._predict_neural(airfoil, m, p, t, alpha_deg, reynolds)
        else:
            result = self._predict_analytic(
                alpha_deg, m, p, t, reynolds, uq_guard=self.analytic_uq_guard
            )
            if self.backend == "neural":
                result["model_backend"] = "analytic_fallback"
                result["backend_note"] = "neural weights present but geometry construction failed"
                result["model_provenance"] = "analytic_fallback"
            if airfoil is not None:
                result["x_grid"] = airfoil.x

        wingbox = None
        if compute_structures and airfoil is not None:
            try:
                wingbox = compute_wingbox_properties(airfoil, x_front_spar=0.20, x_rear_spar=0.65)
            except Exception:
                wingbox = None

        result["wingbox"] = wingbox
        result.setdefault("model_backend", self.backend)
        result.setdefault("model_provenance", self.model_provenance)
        return result

    def _build_features(self, airfoil: AirfoilGeometry, alpha_deg: float, reynolds: float) -> np.ndarray:
        values = {
            "m": airfoil.m,
            "p": airfoil.p,
            "t": airfoil.t,
            **{f"cst_u_{i}": float(v) for i, v in enumerate(airfoil.cst_upper)},
            **{f"cst_l_{i}": float(v) for i, v in enumerate(airfoil.cst_lower)},
            "alpha_deg": float(alpha_deg),
            "reynolds": float(reynolds),
        }
        return np.array([[values[f] for f in self.feature_order]], dtype=float)

    @staticmethod
    def _analytic_cp(
        x_grid: np.ndarray,
        cl: float,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a conservative analytic Cp field for legacy model bundles."""
        x_safe = np.clip(x_grid, 1e-4, 1.0 - 1e-4)
        v_gamma = 2.0 * cl * np.sqrt((1.0 - x_safe) / x_safe) / math.pi
        v_thick = 1.0 + 5.0 * t * (
            0.2969 / (2.0 * np.sqrt(x_safe)) - 0.1260 - 0.7032 * x_safe
        )
        v_thick = np.clip(v_thick, 0.0, 2.5)
        cp_upper = np.clip(1.0 - (v_thick + 0.5 * v_gamma) ** 2, -4.5, 1.0)
        cp_lower = np.clip(1.0 - (v_thick - 0.5 * v_gamma) ** 2, -1.0, 1.0)
        return cp_upper, cp_lower, cp_lower - cp_upper

    def _predict_neural(
        self, airfoil: AirfoilGeometry, m: float, p: float, t: float,
        alpha_deg: float, reynolds: float,
    ) -> dict[str, Any]:
        feats = self._build_features(airfoil, alpha_deg, reynolds)
        x = (feats - self.x_mean) / self.x_scale

        y_norm = self._forward_scalars(x)
        y = y_norm * self.y_scale + self.y_mean
        cl, cd, cm = (float(v) for v in y[0])
        l_over_d = cl / max(cd, 1e-6)

        x_grid = airfoil.x
        field_backend = "neural"
        if self.field_mean is not None:
            field = self._forward_field(x)[0]
            field = field * self.field_scale + self.field_mean
            n_pts = field.shape[0] // 2
            if n_pts != len(x_grid) or field.shape[0] % 2:
                raise ValueError(
                    "Cp decoder output does not match the runtime chord grid "
                    f"({field.shape[0]} values for {len(x_grid)} points)."
                )
            cp_upper = field[:n_pts]
            cp_lower = field[n_pts:]
            delta_cp = cp_lower - cp_upper
            uq_guard = self.uq_guard
        else:
            # Older bundles trained this decoder on standardized Cp but did not
            # export its target scaler. Never present those normalized values as
            # physical pressure coefficients.
            cp_upper, cp_lower, delta_cp = self._analytic_cp(x_grid, cl, t)
            field_backend = "analytic_fallback_missing_field_scaler"
            uq_guard = self.field_fallback_uq_guard

        uq_report = uq_guard.evaluate(
            m=m, p=p, t=t, alpha_deg=alpha_deg, reynolds=reynolds,
            cl=cl, cd=cd, cm=cm, l_over_d=l_over_d,
            x_grid=x_grid, cp_upper=cp_upper, cp_lower=cp_lower,
        )

        return {
            "cl": cl, "cd": cd, "cm": cm, "lift_to_drag": l_over_d,
            "x_grid": x_grid, "cp_upper": cp_upper, "cp_lower": cp_lower,
            "delta_cp": delta_cp, "uq": uq_report,
            "field_backend": field_backend,
        }

    def _predict_analytic(
        self,
        alpha_deg: float,
        m: float,
        p: float,
        t: float,
        reynolds: float,
        uq_guard: AerodynamicUncertaintyGuard | None = None,
    ) -> dict[str, Any]:
        """Closed-form thin-airfoil fallback (used when trained weights are absent)."""
        a_rad = math.radians(alpha_deg)

        alpha_0_deg = -m * 75.0
        cl_linear = 2.0 * math.pi * math.radians(alpha_deg - alpha_0_deg)
        thickness_factor = 0.92 - 0.25 * t
        stall_factor = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, (abs(alpha_deg) - 12.0) / 1.5))))
        cl = float(cl_linear * thickness_factor * stall_factor)

        cf = 0.074 / (reynolds**0.2)
        cd0 = float(2.0 * cf * (1.0 + 2.0 * t + 60.0 * (t**4)))
        k_induced = 0.0055 + 0.002 * (t / 0.12)
        cd = float(cd0 + k_induced * (cl - 0.1 * m) ** 2 + 0.001 * (1.0 - stall_factor))

        cm = float(-0.25 * math.pi * m * 0.95 - 0.005 * a_rad)
        l_over_d = float(cl / max(cd, 1e-6))

        beta = np.linspace(0.0, np.pi, 161)
        x_grid = 0.5 * (1.0 - np.cos(beta))
        cp_upper, cp_lower, delta_cp = self._analytic_cp(x_grid, cl, t)

        guard = uq_guard if uq_guard is not None else self.uq_guard
        uq_report = guard.evaluate(
            m=m, p=p, t=t, alpha_deg=alpha_deg, reynolds=reynolds,
            cl=cl, cd=cd, cm=cm, l_over_d=l_over_d,
            x_grid=x_grid, cp_upper=cp_upper, cp_lower=cp_lower,
        )

        return {
            "cl": cl, "cd": cd, "cm": cm, "lift_to_drag": l_over_d,
            "x_grid": x_grid, "cp_upper": cp_upper, "cp_lower": cp_lower,
            "delta_cp": delta_cp, "uq": uq_report,
            "field_backend": "analytic_fallback",
        }

    def optimize_airfoil(
        self,
        cl_target: float = 0.50,
        t_min: float = 0.10,
        t_max: float = 0.16,
        m_max: float = 0.04,
        min_front_spar: float = 0.08,
        min_rear_spar: float = 0.04,
        min_fuel_area: float = 0.035,
        alpha_bounds: tuple[float, float] = (1.0, 7.0),
        x0: tuple[float, float, float, float] = (0.0, 0.2, 0.12, 4.0),
    ) -> dict[str, Any]:
        """
        Run constrained SLSQP inverse design optimization to maximize L/D
        under coupled aerodynamic and structural wingbox constraints.
        """
        from functools import lru_cache
        from scipy.optimize import minimize
        import time

        def objective(x):
            m, p, t, alpha = x
            res = self.predict(m, p, t, alpha, compute_structures=False)
            return -(res["cl"] / max(res["cd"], 1e-5))

        def ineq_lift(x):
            m, p, t, alpha = x
            res = self.predict(m, p, t, alpha, compute_structures=False)
            return res["cl"] - cl_target

        @lru_cache(maxsize=256)
        def structural_properties(m, p, t):
            airfoil = generate_naca4_airfoil(m=m, p=p, t=t)
            return compute_wingbox_properties(
                airfoil,
                x_front_spar=0.20,
                x_rear_spar=0.65,
            )

        def wingbox_for(x):
            m, p, t, _alpha = (float(value) for value in x)
            return structural_properties(m, p, t)

        def ineq_front_spar(x):
            return wingbox_for(x).h_front_spar - min_front_spar

        def ineq_rear_spar(x):
            return wingbox_for(x).h_rear_spar - min_rear_spar

        def ineq_fuel_area(x):
            return wingbox_for(x).fuel_tank_area - min_fuel_area

        if not 0.0 <= m_max <= 0.09:
            raise ValueError("m_max must lie in [0.0, 0.09].")
        if not 0.0 < t_min <= t_max:
            raise ValueError("Thickness bounds must satisfy 0 < t_min <= t_max.")
        if alpha_bounds[0] > alpha_bounds[1]:
            raise ValueError("alpha_bounds must be ordered from low to high.")

        x0_array = np.asarray(x0, dtype=float)
        if x0_array.shape != (4,) or not np.all(np.isfinite(x0_array)):
            raise ValueError("x0 must contain four finite values (m, p, t, alpha).")

        bounds = [
            (0.00, m_max),
            (0.20, 0.50),
            (t_min, t_max),
            alpha_bounds,
        ]
        x0_array = np.array([
            np.clip(value, lower, upper)
            for value, (lower, upper) in zip(x0_array, bounds, strict=True)
        ])
        constraints = [
            {"type": "ineq", "fun": ineq_lift},
            {"type": "ineq", "fun": ineq_front_spar},
            {"type": "ineq", "fun": ineq_rear_spar},
            {"type": "ineq", "fun": ineq_fuel_area},
        ]

        t0 = time.perf_counter()
        opt_res = minimize(
            objective,
            x0_array,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-5},
        )
        elapsed = time.perf_counter() - t0

        opt_m, opt_p, opt_t, opt_alpha = opt_res.x
        baseline_pred = self.predict(
            x0_array[0], x0_array[1], x0_array[2], x0_array[3],
            compute_structures=True,
        )
        opt_pred = self.predict(opt_m, opt_p, opt_t, opt_alpha, compute_structures=True)

        wingbox_constraints_present = wb_opt is not None
        struct_ok = wingbox_constraints_present and (
            wb_opt.h_front_spar >= (min_front_spar - 1e-4)
            and wb_opt.h_rear_spar >= (min_rear_spar - 1e-4)
            and wb_opt.fuel_tank_area >= (min_fuel_area - 1e-4)
        )
        lift_ok = opt_pred["cl"] >= (cl_target - 1e-4)
        constraints_met = bool(struct_ok and lift_ok)

        return {
            "success": bool(opt_res.success and constraints_met),
            "solver_success": bool(opt_res.success),
            "solver_message": str(opt_res.message),
            "elapsed_seconds": elapsed,
            "iterations": opt_res.nit,
            "baseline_params": {
                "m": float(x0_array[0]),
                "p": float(x0_array[1]),
                "t": float(x0_array[2]),
                "alpha": float(x0_array[3]),
            },
            "opt_params": {"m": opt_m, "p": opt_p, "t": opt_t, "alpha": opt_alpha},
            "baseline_pred": baseline_pred,
            "opt_pred": opt_pred,
            "ld_improvement_pct": ((opt_pred["lift_to_drag"] / baseline_pred["lift_to_drag"]) - 1.0) * 100.0,
            "structural_constraints_met": bool(struct_ok),
            "lift_constraint_met": bool(lift_ok),
            "constraints_met": constraints_met,
            "min_front_spar": min_front_spar,
            "min_rear_spar": min_rear_spar,
            "min_fuel_area": min_fuel_area,
        }
