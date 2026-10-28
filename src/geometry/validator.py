"""
Airfoil Geometry Validation for CFD Meshing and ML Integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.geometry.naca import AirfoilGeometry


@dataclass(frozen=True)
class GeometryValidationResult:
    """Validation report for an airfoil geometry."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    max_thickness_ratio: float
    max_camber_ratio: float
    te_gap: float
    max_second_diff: float


def validate_airfoil_geometry(
    airfoil: AirfoilGeometry,
    min_thickness_ratio: float = 0.06,
    max_thickness_ratio: float = 0.18,
    max_camber_ratio: float = 0.06,
    te_closure_tol: float = 1e-6,
    max_slope_jump: float = 0.5,
) -> GeometryValidationResult:
    """Validate an AirfoilGeometry instance against geometric and meshing constraints."""
    errors: list[str] = []
    warnings: list[str] = []

    x = airfoil.x
    yu = airfoil.y_upper
    yl = airfoil.y_lower

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(yu)) and np.all(np.isfinite(yl))):
        errors.append("Coordinate arrays contain NaN or infinite values.")
        return GeometryValidationResult(
            is_valid=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            max_thickness_ratio=0.0,
            max_camber_ratio=0.0,
            te_gap=float("nan"),
            max_second_diff=float("nan"),
        )

    if abs(x[0] - 0.0) > 1e-7 or abs(x[-1] - 1.0) > 1e-7:
        errors.append(f"Chord endpoints must be [0.0, 1.0], got [{x[0]:.6e}, {x[-1]:.6e}].")
    if np.any(np.diff(x) <= 0.0):
        errors.append("Chordwise x coordinates are not strictly monotonically increasing.")

    if abs(yu[0]) > 1e-6 or abs(yl[0]) > 1e-6:
        errors.append(f"Leading edge is not pinned at y=0: yu(0)={yu[0]:.3e}, yl(0)={yl[0]:.3e}.")

    te_gap = float(abs(yu[-1] - yl[-1]))
    if airfoil.closed_te and te_gap > te_closure_tol:
        errors.append(f"Trailing edge is not closed: gap={te_gap:.3e} > tol={te_closure_tol:.3e}.")

    interior_thickness = yu[1:-1] - yl[1:-1]
    if np.any(interior_thickness <= 0.0):
        min_thick = float(np.min(interior_thickness))
        errors.append(
            f"Self-intersecting or zero-thickness interior detected: min(yu - yl) = {min_thick:.3e}."
        )

    measured_t = airfoil.max_thickness_measured
    measured_m = airfoil.max_camber_measured

    if measured_t < min_thickness_ratio - 1e-3:
        errors.append(
            f"Measured thickness ratio {measured_t:.4f} is below minimum {min_thickness_ratio:.4f}."
        )
    if measured_t > max_thickness_ratio + 1e-3:
        errors.append(
            f"Measured thickness ratio {measured_t:.4f} exceeds maximum {max_thickness_ratio:.4f}."
        )
    if measured_m > max_camber_ratio + 1e-3:
        errors.append(
            f"Measured camber ratio {measured_m:.4f} exceeds maximum {max_camber_ratio:.4f}."
        )

    mask_smooth = x > 0.02
    dx = np.diff(x[mask_smooth])
    dyu_dx = np.diff(yu[mask_smooth]) / dx
    dyl_dx = np.diff(yl[mask_smooth]) / dx

    max_jump_u = float(np.max(np.abs(np.diff(dyu_dx)))) if len(dyu_dx) > 1 else 0.0
    max_jump_l = float(np.abs(np.diff(dyl_dx)).max()) if len(dyl_dx) > 1 else 0.0
    max_second_diff = max(max_jump_u, max_jump_l)

    if max_second_diff > max_slope_jump:
        errors.append(
            f"Surface slope discontinuity detected (max slope jump = {max_second_diff:.3f} > {max_slope_jump:.3f})."
        )

    return GeometryValidationResult(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        max_thickness_ratio=measured_t,
        max_camber_ratio=measured_m,
        te_gap=te_gap,
        max_second_diff=max_second_diff,
    )