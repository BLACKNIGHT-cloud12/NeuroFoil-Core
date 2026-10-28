"""
Parameterized 2D Airfoil Geometry Generation and Representation.

Supports:
- NACA 4-digit series with cosine chordwise spacing and closed trailing edge
  (NASA Langley Turbulence Modeling Resource formulation, a4 = -0.1036).
- Class-Shape Transformation (CST / Kulfan) polynomial representation for
  extensibility to general airfoil shapes.
- Standardized fixed-grid surface interpolation for machine-learning inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass(frozen=True)
class AirfoilGeometry:
    """Container for a normalized 2D airfoil geometry (unit chord c = 1.0)."""

    code: str
    family: str
    m: float
    p: float
    t: float
    x: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    x_upper: np.ndarray
    x_lower: np.ndarray
    y_camber: np.ndarray
    cst_upper: tuple[float, ...]
    cst_lower: tuple[float, ...]
    closed_te: bool = True

    @property
    def max_thickness_measured(self) -> float:
        """Measured maximum vertical thickness across the chord."""
        return float(np.max(self.y_upper - self.y_lower))

    @property
    def max_camber_measured(self) -> float:
        """Measured maximum camber across the chord."""
        return float(np.max(np.abs(self.y_camber)))

    @property
    def te_gap(self) -> float:
        """Trailing-edge vertical gap at x/c = 1."""
        return float(abs(self.y_upper[-1] - self.y_lower[-1]))

    def to_closed_contour(self) -> np.ndarray:
        """
        Return a single counter-clockwise (N_total, 2) coordinate array:
        Trailing Edge (1,0) -> Upper Surface -> Leading Edge (0,0) -> Lower Surface -> Trailing Edge (1,0).
        """
        upper_te_to_le = np.column_stack([self.x_upper[::-1], self.y_upper[::-1]])
        lower_le_to_te = np.column_stack([self.x_lower[1:], self.y_lower[1:]])
        return np.vstack([upper_te_to_le, lower_le_to_te])

    def sample_ml_features(self, n_sample: int = 32) -> dict[str, Any]:
        """
        Extract a consistent ML feature dictionary combining parametric, CST,
        and interpolated upper/lower surface coordinates on an interior cosine grid.
        """
        beta = np.linspace(0.0, np.pi, n_sample + 2)[1:-1]
        x_grid = 0.5 * (1.0 - np.cos(beta))
        yu_interp = np.interp(x_grid, self.x, self.y_upper)
        yl_interp = np.interp(x_grid, self.x, self.y_lower)

        features: dict[str, Any] = {
            "airfoil_code": self.code,
            "m": float(self.m),
            "p": float(self.p),
            "t": float(self.t),
        }
        for i, val in enumerate(self.cst_upper):
            features[f"cst_u_{i}"] = float(val)
        for i, val in enumerate(self.cst_lower):
            features[f"cst_l_{i}"] = float(val)
        for i, val in enumerate(yu_interp):
            features[f"yu_{i:02d}"] = float(val)
        for i, val in enumerate(yl_interp):
            features[f"yl_{i:02d}"] = float(val)
        return features

    def save_dat(self, output_path: Path | str) -> Path:
        """Save airfoil coordinates in standard Selig .dat format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        contour = self.to_closed_contour()
        lines = [f"# {self.code} (family={self.family}, m={self.m:.4f}, p={self.p:.4f}, t={self.t:.4f})"]
        for pt in contour:
            lines.append(f"{pt[0]:.8f}  {pt[1]:.8f}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def save_metadata(self, output_path: Path | str) -> Path:
        """Save scalar geometry parameters and CST weights to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "code": self.code,
            "family": self.family,
            "m": self.m,
            "p": self.p,
            "t": self.t,
            "closed_te": self.closed_te,
            "num_points_per_surface": int(len(self.x)),
            "max_thickness_measured": self.max_thickness_measured,
            "max_camber_measured": self.max_camber_measured,
            "te_gap": self.te_gap,
            "cst_upper": list(self.cst_upper),
            "cst_lower": list(self.cst_lower),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def parse_naca4_code(code: str) -> tuple[float, float, float]:
    """
    Parse a 4-digit string (e.g. '0012', '2412', 'NACA4415') into (m, p, t).
    - m: maximum camber as fraction of chord (first digit / 100)
    - p: position of maximum camber in tenths of chord (second digit / 10)
    - t: maximum thickness as fraction of chord (last two digits / 100)
    """
    cleaned = code.strip().upper().replace("NACA", "").replace("-", "").replace(" ", "")
    if len(cleaned) != 4 or not cleaned.isdigit():
        raise ValueError(f"Invalid NACA 4-digit code: '{code}'. Expected 4 digits like '0012' or '2412'.")

    m_digit = int(cleaned[0])
    p_digit = int(cleaned[1])
    t_digits = int(cleaned[2:4])

    m = m_digit / 100.0
    p = p_digit / 10.0
    t = t_digits / 100.0

    # Symmetric airfoils must have both m == 0 and p == 0
    if (m == 0.0 and p != 0.0) or (m != 0.0 and p == 0.0):
        raise ValueError(
            f"Inconsistent camber parameters in NACA {cleaned}: m={m}, p={p}. "
            "Both m and p must be zero for symmetric airfoils, or both non-zero for cambered airfoils."
        )
    if t <= 0.0:
        raise ValueError(f"Thickness ratio t must be positive, got t={t} in '{code}'.")

    return m, p, t


def cosine_spacing(num_points: int = 161) -> np.ndarray:
    """Generate cosine-spaced chordwise coordinates x/c in [0.0, 1.0]."""
    if num_points < 21:
        raise ValueError(f"num_points must be at least 21 for adequate LE resolution, got {num_points}")
    beta = np.linspace(0.0, np.pi, num_points)
    x = 0.5 * (1.0 - np.cos(beta))
    x[0] = 0.0
    x[-1] = 1.0
    return x


def fit_cst_weights(x: np.ndarray, y_surface: np.ndarray, order: int = 5) -> tuple[float, ...]:
    """
    Fit Kulfan Class-Shape Transformation (CST) Bernstein polynomial weights
    to an airfoil surface y(x) with class exponents N1 = 0.5, N2 = 1.0.
    """
    mask = (x > 1e-8) & (x < 1.0 - 1e-8)
    xi = x[mask]
    yi = y_surface[mask]

    class_fn = np.sqrt(xi) * (1.0 - xi)
    basis = np.zeros((len(xi), order + 1), dtype=float)
    for r in range(order + 1):
        binom = comb(order, r)
        basis[:, r] = class_fn * binom * (xi**r) * ((1.0 - xi) ** (order - r))

    weights, _, _, _ = np.linalg.lstsq(basis, yi, rcond=None)
    return tuple(round(float(w), 6) for w in weights)


def evaluate_cst_surface(x: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """Reconstruct surface y(x) from Kulfan CST Bernstein weights."""
    order = len(weights) - 1
    class_fn = np.sqrt(np.clip(x, 0.0, 1.0)) * (1.0 - np.clip(x, 0.0, 1.0))
    y = np.zeros_like(x, dtype=float)
    for r, w in enumerate(weights):
        binom = comb(order, r)
        y += w * class_fn * binom * (x**r) * ((1.0 - x) ** (order - r))
    y[0] = 0.0
    y[-1] = 0.0
    return y


def generate_naca4_airfoil(
    code: str = "0012",
    num_points: int = 161,
    closed_te: bool = True,
    cst_order: int = 5,
) -> AirfoilGeometry:
    """
    Generate a parameterized NACA 4-digit airfoil with cosine chordwise spacing.

    Uses the standard NACA thickness distribution:
      y_t = 5 * t * (a0*sqrt(x) + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
    where a4 = -0.1036 for a closed trailing edge (NASA TMR standard)
    or a4 = -0.1015 for an open blunt trailing edge.
    """
    m, p, t = parse_naca4_code(code)
    clean_code = f"NACA{int(round(m * 100)):1d}{int(round(p * 10)):1d}{int(round(t * 100)):02d}"

    x = cosine_spacing(num_points)
    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036 if closed_te else -0.1015

    y_t = 5.0 * t * (a0 * np.sqrt(x) + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4))
    if closed_te:
        y_t[-1] = 0.0

    y_c = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0.0 and p > 0.0:
        front = x <= p
        back = ~front

        y_c[front] = (m / (p**2)) * (2.0 * p * x[front] - x[front] ** 2)
        dyc_dx[front] = (2.0 * m / (p**2)) * (p - x[front])

        y_c[back] = (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * x[back] - x[back] ** 2)
        dyc_dx[back] = (2.0 * m / ((1.0 - p) ** 2)) * (p - x[back])

    theta = np.arctan(dyc_dx)

    # Compute displaced upper/lower coordinates, then resample onto common chordwise x grid
    xu_raw = x - y_t * np.sin(theta)
    yu_raw = y_c + y_t * np.cos(theta)
    xl_raw = x + y_t * np.sin(theta)
    yl_raw = y_c - y_t * np.cos(theta)

    # Pin exact leading and trailing edges
    xu_raw[0], yu_raw[0] = 0.0, 0.0
    xl_raw[0], yl_raw[0] = 0.0, 0.0
    if closed_te:
        xu_raw[-1], yu_raw[-1] = 1.0, 0.0
        xl_raw[-1], yl_raw[-1] = 1.0, 0.0

    # Interpolate onto the monotonic cosine x grid for clean structured C-grid meshing
    y_upper = np.interp(x, xu_raw, yu_raw)
    y_lower = np.interp(x, xl_raw, yl_raw)
    y_upper[0], y_lower[0] = 0.0, 0.0
    if closed_te:
        y_upper[-1], y_lower[-1] = 0.0, 0.0

    cst_u = fit_cst_weights(x, y_upper, order=cst_order)
    cst_l = fit_cst_weights(x, y_lower, order=cst_order)

    return AirfoilGeometry(
        code=clean_code,
        family="naca4",
        m=m,
        p=p,
        t=t,
        x=x,
        y_upper=y_upper,
        y_lower=y_lower,
        x_upper=x.copy(),
        x_lower=x.copy(),
        y_camber=y_c,
        cst_upper=cst_u,
        cst_lower=cst_l,
        closed_te=closed_te,
    )