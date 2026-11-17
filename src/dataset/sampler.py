"""
Airfoil Geometry and Operating Condition Sampler for CFD Dataset Generation.

Generates Latin Hypercube and canonical aerodynamic parameter grids across
the NACA 4-digit design space and angle-of-attack range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class AirfoilSample:
    """Design space sample defining geometry parameters and operating conditions.

    For NACA-mode cases, `naca_code` defines the geometry. For general-CST
    cases, provide `cst_upper`/`cst_lower` instead and set `naca_code` to the
    shape's CST hash id (e.g. "CST-1a2b3c4d"); m/p/t then carry the *measured*
    camber/camber-position/thickness summaries.
    """

    case_id: str
    naca_code: str
    m: float
    p: float
    t: float
    alpha_deg: float
    reynolds: float
    cst_upper: tuple[float, ...] | None = None
    cst_lower: tuple[float, ...] | None = None


def naca_code_from_params(m: float, p: float, t: float) -> str:
    """Format (m, p, t) into standard 4-digit NACA string (e.g. 2412)."""
    m_int = int(round(m * 100))
    p_int = int(round(p * 10))
    t_int = int(round(t * 100))

    # Clamp bounds
    m_int = max(0, min(9, m_int))
    p_int = max(0, min(9, p_int))
    t_int = max(6, min(24, t_int))

    if m_int == 0 or p_int == 0:
        m_int, p_int = 0, 0

    return f"{m_int:1d}{p_int:1d}{t_int:02d}"


def sample_latin_hypercube(
    n_samples: int = 30,
    alpha_min: float = -4.0,
    alpha_max: float = 12.0,
    reynolds: float | None = None,
    seed: int = 42,
    *,
    reynolds_min: float = 1.0e6,
    reynolds_max: float = 6.0e6,
) -> list[AirfoilSample]:
    """
    Generate space-filling Latin Hypercube Samples across NACA thickness,
    camber, camber location, angle of attack, and (by default) Reynolds number.

    When ``reynolds`` is ``None``, Reynolds number is sampled log-uniformly
    between ``reynolds_min`` and ``reynolds_max`` using a fifth LHS dimension.
    Passing a finite positive ``reynolds`` retains the legacy fixed-Reynolds
    behavior and uses the original four-dimensional design.
    """
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
    if not all(
        isinstance(bound, (int, float, np.integer, np.floating))
        and not isinstance(bound, (bool, np.bool_))
        for bound in (alpha_min, alpha_max)
    ) or not np.isfinite(alpha_min) or not np.isfinite(alpha_max) or alpha_min >= alpha_max:
        raise ValueError(
            f"Angle-of-attack bounds must be finite with alpha_min < alpha_max, got "
            f"[{alpha_min}, {alpha_max}]."
        )

    sample_reynolds = reynolds is None
    if sample_reynolds:
        if not all(
            isinstance(bound, (int, float, np.integer, np.floating))
            and not isinstance(bound, (bool, np.bool_))
            for bound in (reynolds_min, reynolds_max)
        ):
            raise ValueError(
                "Reynolds bounds must be finite and positive with reynolds_min < reynolds_max, "
                f"got [{reynolds_min}, {reynolds_max}]."
            )
        if (
            not np.isfinite(reynolds_min)
            or not np.isfinite(reynolds_max)
            or reynolds_min <= 0.0
            or reynolds_min >= reynolds_max
        ):
            raise ValueError(
                "Reynolds bounds must be finite and positive with reynolds_min < reynolds_max, "
                f"got [{reynolds_min}, {reynolds_max}]."
            )
    elif (
        not isinstance(reynolds, (int, float, np.integer, np.floating))
        or isinstance(reynolds, (bool, np.bool_))
        or not np.isfinite(reynolds)
        or reynolds <= 0.0
    ):
        raise ValueError(f"Fixed reynolds must be finite and positive, got {reynolds!r}.")

    dimensions = 5 if sample_reynolds else 4
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    unit_samples = sampler.random(n=int(n_samples))

    # Bounds: [m_fraction, p_fraction, t_fraction, alpha_deg, log(Re)]
    # m: [0.0, 0.06], p: [0.2, 0.6], t: [0.06, 0.18], alpha: [alpha_min, alpha_max]
    l_bounds = [0.0, 0.20, 0.06, alpha_min]
    u_bounds = [0.06, 0.60, 0.18, alpha_max]
    scaled = qmc.scale(unit_samples[:, :4], l_bounds, u_bounds)

    if sample_reynolds:
        log_re_min = np.log(float(reynolds_min))
        log_re_max = np.log(float(reynolds_max))
        reynolds_values = np.exp(log_re_min + unit_samples[:, 4] * (log_re_max - log_re_min))
    else:
        reynolds_values = np.full(int(n_samples), float(reynolds), dtype=float)

    samples: list[AirfoilSample] = []
    for i, (row, reynolds_value) in enumerate(zip(scaled, reynolds_values, strict=True)):
        m_raw, p_raw, t_raw, alpha_raw = row
        # Keep the display code for readability, but preserve the continuous LHS geometry.
        code = naca_code_from_params(m_raw, p_raw, t_raw)
        alpha_val = round(float(alpha_raw), 2)
        case_id = f"NACA{code}_a{alpha_val:+.2f}_case{i:03d}"

        samples.append(
            AirfoilSample(
                case_id=case_id,
                naca_code=code,
                m=float(m_raw),
                p=float(p_raw),
                t=float(t_raw),
                alpha_deg=alpha_val,
                reynolds=float(reynolds_value),
            )
        )
    return samples


def generate_canonical_benchmark_cases(
    codes: Sequence[str] = ("0012", "2412", "4412", "0015", "2415"),
    alphas: Sequence[float] = (-4.0, 0.0, 4.0, 8.0, 12.0),
    reynolds: float = 6.0e6,
) -> list[AirfoilSample]:
    """Generate a clean polar grid across classic NACA benchmark airfoils."""
    samples: list[AirfoilSample] = []
    idx = 0
    for code in codes:
        clean = code.replace("NACA", "").replace("-", "").strip()
        m_val = int(clean[0]) / 100.0
        p_val = int(clean[1]) / 10.0
        t_val = int(clean[2:4]) / 100.0
        for alpha in alphas:
            case_id = f"NACA{clean}_a{alpha:+.2f}_bmk{idx:03d}"
            samples.append(
                AirfoilSample(
                    case_id=case_id,
                    naca_code=clean,
                    m=m_val,
                    p=p_val,
                    t=t_val,
                    alpha_deg=float(alpha),
                    reynolds=reynolds,
                )
            )
            idx += 1
    return samples
