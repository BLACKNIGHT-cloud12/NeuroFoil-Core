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
    """Design space sample defining geometry parameters and operating conditions."""

    case_id: str
    naca_code: str
    m: float
    p: float
    t: float
    alpha_deg: float
    reynolds: float


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
    reynolds: float = 6.0e6,
    seed: int = 42,
) -> list[AirfoilSample]:
    """
    Generate space-filling Latin Hypercube Samples across NACA thickness,
    camber, camber location, and angle of attack.
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Bounds: [m_fraction, p_fraction, t_fraction, alpha_deg]
    # m: [0.0, 0.06], p: [0.2, 0.6], t: [0.06, 0.18], alpha: [alpha_min, alpha_max]
    l_bounds = [0.0, 0.20, 0.06, alpha_min]
    u_bounds = [0.06, 0.60, 0.18, alpha_max]
    scaled = qmc.scale(unit_samples, l_bounds, u_bounds)

    samples: list[AirfoilSample] = []
    for i, row in enumerate(scaled):
        m_raw, p_raw, t_raw, alpha_raw = row
        # Discretize into valid NACA 4-digit code
        code = naca_code_from_params(m_raw, p_raw, t_raw)
        m_val = int(code[0]) / 100.0
        p_val = int(code[1]) / 10.0
        t_val = int(code[2:4]) / 100.0
        alpha_val = round(float(alpha_raw), 2)
        case_id = f"NACA{code}_a{alpha_val:+.2f}_case{i:03d}"

        samples.append(
            AirfoilSample(
                case_id=case_id,
                naca_code=code,
                m=m_val,
                p=p_val,
                t=t_val,
                alpha_deg=alpha_val,
                reynolds=reynolds,
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
