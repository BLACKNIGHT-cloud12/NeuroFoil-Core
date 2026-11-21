"""NeuroFoil desktop backend — FastAPI wrapper around AeroSurrogateEngine.

Fully local: binds to 127.0.0.1 only, serves the SPA from ./ui, and wraps the
shipped real-CFD model bundle. No cloud calls, no telemetry.

Run:  python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
(spawned automatically by the Electron main process)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import AeroSurrogateEngine  # noqa: E402

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(title="NeuroFoil Flight Deck", docs_url=None, redoc_url=None)
engine = AeroSurrogateEngine()
FINGERPRINT = f"{engine.model_provenance}:{engine.meta.get('trained_at', 'no-artifact-metadata')}"

# Observed coefficient ranges from the converged campaign rows — the readout
# range bars are scaled against these so "where am I" is dataset-honest.
try:
    import pandas as pd

    _conv = pd.read_csv(PROJECT_ROOT / "data/processed/airfoil_cfd_dataset.csv")
    _conv = _conv[_conv["status"] == "converged"]
    DATASET_RANGES = {
        key: [float(_conv[key].min()), float(_conv[key].max())]
        for key in ("cl", "cd", "cm")
        if key in _conv.columns
    }
    # L/D is only aerodynamically meaningful with positive lift
    _pos = _conv[_conv["lift_to_drag"] > 0]
    if len(_pos):
        DATASET_RANGES["lift_to_drag"] = [
            float(_pos["lift_to_drag"].min()), float(_pos["lift_to_drag"].max())]
except Exception:  # noqa: BLE001
    DATASET_RANGES = {}


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series(value: Any) -> list[float | None]:
    if value is None:
        return []
    return [_num(v) for v in np.asarray(value).ravel().tolist()]


def _uq_payload(uq: Any) -> dict[str, Any]:
    return {
        "status": str(getattr(uq, "status", "")),
        "is_ood": bool(getattr(uq, "is_ood", False)),
        "confidence_score": _num(getattr(uq, "confidence_score", None)),
        "mahalanobis_dist": _num(getattr(uq, "mahalanobis_dist", None)),
        "cl_ci95": [_num(v) for v in (getattr(uq, "cl_ci95", None) or (None, None))],
        "cd_ci95": [_num(v) for v in (getattr(uq, "cd_ci95", None) or (None, None))],
        "cm_ci95": [_num(v) for v in (getattr(uq, "cm_ci95", None) or (None, None))],
        "cp_upper_ci95": _series(getattr(uq, "cp_upper_ci95_low", None)),
        "cp_upper_ci95_high": _series(getattr(uq, "cp_upper_ci95_high", None)),
        "cp_lower_ci95": _series(getattr(uq, "cp_lower_ci95_low", None)),
        "cp_lower_ci95_high": _series(getattr(uq, "cp_lower_ci95_high", None)),
        "warnings": [str(w) for w in (getattr(uq, "warning_messages", None) or ())],
    }


def _wingbox_payload(wb: Any) -> dict[str, Any] | None:
    if wb is None:
        return None
    out: dict[str, Any] = {}
    for key, value in getattr(wb, "__dict__", {}).items():
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                out[key] = [[_num(a), _num(b)] for a, b in value]
            else:
                out[key] = _series(value)
        elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            out[key] = _num(value)
        else:
            out[key] = value
    return out


def _prediction_payload(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "cl": _num(res.get("cl")),
        "cd": _num(res.get("cd")),
        "cm": _num(res.get("cm")),
        "lift_to_drag": _num(res.get("lift_to_drag")),
        "cp_upper": [_num(v) for v in res.get("cp_upper", [])],
        "cp_lower": [_num(v) for v in res.get("cp_lower", [])],
        "delta_cp": [_num(v) for v in res.get("delta_cp", [])],
        "uq": _uq_payload(res.get("uq")),
        "wingbox": _wingbox_payload(res.get("wingbox")),
        "model_backend": res.get("model_backend"),
        "model_provenance": res.get("model_provenance"),
        "field_backend": res.get("field_backend"),
        "backend_note": res.get("backend_note"),
    }


class PredictRequest(BaseModel):
    m: float = Field(ge=0.0, le=0.10)
    p: float = Field(ge=0.0, le=0.9)
    t: float = Field(ge=0.04, le=0.24)
    alpha_deg: float = Field(ge=-8.0, le=16.0)
    reynolds: float = Field(ge=1e5, le=2e7)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "model_backend": engine.backend,
        "model_provenance": engine.model_provenance,
        "artifact_fingerprint": FINGERPRINT,
        "calibration_n": engine.meta.get("conformal", {}).get("calibration_n"),
        "field_calibration_n": engine.meta.get("conformal", {}).get(
            "field_calibration_n"),
        "trained_at": engine.meta.get("trained_at"),
        "ranges": DATASET_RANGES,
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        res = engine.predict(
            req.m, req.p, req.t, req.alpha_deg, req.reynolds,
            compute_structures=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return _prediction_payload(res)


class PolarRequest(BaseModel):
    m: float = Field(ge=0.0, le=0.10)
    p: float = Field(ge=0.0, le=0.9)
    t: float = Field(ge=0.04, le=0.24)
    reynolds: float = Field(ge=1e5, le=2e7)


@app.post("/api/polar")
def polar(req: PolarRequest) -> dict[str, Any]:
    import numpy as np

    rows = []
    for sweep_alpha in np.linspace(-6.0, 16.0, 45):
        try:
            res = engine.predict(
                req.m, req.p, req.t, float(sweep_alpha), req.reynolds,
                compute_structures=False,
            )
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "alpha": float(sweep_alpha),
            "cl": _num(res.get("cl")),
            "cd": _num(res.get("cd")),
            "cm": _num(res.get("cm")),
            "lift_to_drag": _num(res.get("lift_to_drag")),
        })
    return {"rows": rows}


class OptimizeRequest(BaseModel):
    cl_target: float = 0.50
    t_min: float = 0.10
    t_max: float = 0.16
    m_max: float = 0.04
    min_front_spar: float = 0.08
    min_rear_spar: float = 0.04
    min_fuel_area: float = 0.035
    alpha_low: float = 1.0
    alpha_high: float = 7.0


@app.post("/api/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    try:
        res = engine.optimize_airfoil(
            cl_target=req.cl_target,
            t_min=req.t_min, t_max=req.t_max, m_max=req.m_max,
            min_front_spar=req.min_front_spar,
            min_rear_spar=req.min_rear_spar,
            min_fuel_area=req.min_fuel_area,
            alpha_bounds=(req.alpha_low, req.alpha_high),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {
        "success": bool(res.get("success")),
        "solver_message": str(res.get("solver_message", "")),
        "baseline_params": res.get("baseline_params"),
        "opt_params": res.get("opt_params") or res.get("optimized_params"),
        "elapsed_seconds": _num(res.get("elapsed_seconds")),
        "iterations": res.get("iterations"),
        "constraints_met": res.get("constraints_met"),
        "raw": {k: v for k, v in res.items()
                if not isinstance(v, dict) and not hasattr(v, "predict")},
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
