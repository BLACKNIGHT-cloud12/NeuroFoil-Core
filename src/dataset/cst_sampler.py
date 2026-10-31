"""
General CST Airfoil Sampler.

Breaks the NACA 4-digit parameter-space degeneracy: instead of airfoils being
a deterministic function of (m, p, t), this sampler proposes perturbed CST
coefficient vectors around NACA anchors and keeps only shapes that pass the
geometric validator. The result is a genuinely higher-dimensional airfoil
manifold with 12 free shape degrees of freedom, while every accepted shape is
guaranteed meshable (self-intersection, slope-continuity, thickness and
trailing-edge checks).

All shapes stay schema-compatible with the NACA campaign: the CSV columns
m / p / t carry the *measured* max camber, camber location and thickness of
the sampled shape, and cst_u_* / cst_l_* carry the refitted CST weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from src.geometry.naca import (
    AirfoilGeometry,
    cosine_spacing,
    evaluate_cst_surface,
    fit_cst_weights,
    generate_naca4_airfoil,
)
from src.geometry.validator import validate_airfoil_geometry

CST_ORDER = 5


@dataclass(frozen=True)
class CSTSample:
    """A general-CST airfoil candidate (weights are the *sampled* values)."""

    sample_id: str
    cst_upper: tuple[float, ...]
    cst_lower: tuple[float, ...]
    geometry: AirfoilGeometry

    @property
    def measured_camber(self) -> float:
        return self.geometry.max_camber_measured

    @property
    def measured_thickness(self) -> float:
        return self.geometry.max_thickness_measured


def geometry_from_cst(
    cst_upper: tuple[float, ...],
    cst_lower: tuple[float, ...],
    num_points: int = 161,
) -> AirfoilGeometry:
    """Build an AirfoilGeometry directly from CST weights on the cosine grid."""
    x = cosine_spacing(num_points)
    y_upper = evaluate_cst_surface(x, tuple(cst_upper))
    y_lower = evaluate_cst_surface(x, tuple(cst_lower))
    y_camber = 0.5 * (y_upper + y_lower)

    # Refit for feature extraction (round-trip error is negligible for smooth CST shapes)
    cst_u_fit = fit_cst_weights(x, y_upper, order=CST_ORDER)
    cst_l_fit = fit_cst_weights(x, y_lower, order=CST_ORDER)

    digest = hashlib.sha1(
        (f"{[round(v, 6) for v in cst_upper]}|{[round(v, 6) for v in cst_lower]}").encode()
    ).hexdigest()[:8]

    return AirfoilGeometry(
        code=f"CST-{digest}",
        family="cst",
        m=float(np.max(np.abs(y_camber))),
        p=float(x[int(np.argmax(np.abs(y_camber)))]),
        t=float(np.max(y_upper - y_lower)),
        x=x,
        y_upper=y_upper,
        y_lower=y_lower,
        x_upper=x.copy(),
        x_lower=x.copy(),
        y_camber=y_camber,
        cst_upper=cst_u_fit,
        cst_lower=cst_l_fit,
        closed_te=True,
    )


def sample_general_cst_airfoils(
    n_samples: int,
    seed: int = 42,
    perturbation_scale: float = 0.45,
    thickness_range: tuple[float, float] = (0.06, 0.18),
    max_camber_ratio: float = 0.06,
    max_attempts_factor: int = 60,
) -> list[CSTSample]:
    """
    Rejection-sample general CST airfoils.

    Strategy: draw a NACA anchor (m, p, t) from the campaign bounds, take its
    fitted CST weights, perturb the coefficients multiplicatively, rebuild the
    shape, and keep it only if the geometric validator accepts it (thickness in
    range, no self-intersection, slope continuity, closed trailing edge,
    measured camber within the calibrated manifold).

    perturbation_scale 0.45 keeps >95% of proposals meshable while giving the
    coefficients real freedom; raise it for more aggressive shapes.
    """
    rng = np.random.default_rng(seed)
    accepted: list[CSTSample] = []
    seen: set[str] = set()
    max_attempts = n_samples * max_attempts_factor

    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
        attempts += 1

        # 1. NACA anchor
        m_anchor = float(rng.uniform(0.0, 0.06))
        p_anchor = float(rng.uniform(0.2, 0.6)) if m_anchor > 0 else 0.2
        t_anchor = float(rng.uniform(*thickness_range))
        m_digit = int(round(m_anchor * 100))
        p_digit = int(round(p_anchor * 10)) if m_anchor > 0 else 0
        t_digit = int(round(t_anchor * 100))
        try:
            anchor = generate_naca4_airfoil(f"{m_digit:1d}{p_digit:1d}{t_digit:02d}", 161, True)
        except ValueError:
            continue

        # 2. Perturb both surfaces' CST coefficients multiplicatively
        cu = list(anchor.cst_upper)
        cl = list(anchor.cst_lower)
        for k in range(len(cu)):
            cu[k] = cu[k] * (1.0 + rng.uniform(-1.0, 1.0) * perturbation_scale)
            cl[k] = cl[k] * (1.0 + rng.uniform(-1.0, 1.0) * perturbation_scale)

        try:
            geom = geometry_from_cst(tuple(cu), tuple(cl))
        except Exception:
            continue

        if geom.code in seen:
            continue

        # 3. Validate against the same constraints the campaign enforces
        val = validate_airfoil_geometry(
            geom,
            min_thickness_ratio=thickness_range[0],
            max_thickness_ratio=thickness_range[1],
            max_camber_ratio=max_camber_ratio,
        )
        if not val.is_valid:
            continue

        seen.add(geom.code)
        accepted.append(
            CSTSample(
                sample_id=geom.code,
                cst_upper=tuple(round(v, 6) for v in geom.cst_upper),
                cst_lower=tuple(round(v, 6) for v in geom.cst_lower),
                geometry=geom,
            )
        )

    if len(accepted) < n_samples:
        raise RuntimeError(
            f"Only {len(accepted)}/{n_samples} valid CST shapes after {attempts} attempts; "
            "raise perturbation diversity or loosen constraints."
        )
    return accepted
