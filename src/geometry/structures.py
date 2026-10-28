"""
Aero-Structural Wingbox and Cross-Sectional Mechanics Analysis.

Provides structural evaluation of airfoil cross-sections for MDAO:
- Front and rear spar heights (web depths)
- Total and inter-spar fuel tank cross-sectional areas
- Centroidal neutral axis location (y_bar)
- Area moment of inertia (I_xx) for wing bending stiffness
- Structural section modulus and bending efficiency index
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.geometry.naca import AirfoilGeometry


@dataclass(frozen=True)
class WingboxProperties:
    """Aero-structural cross-sectional metrics for an airfoil."""

    x_front_spar: float
    x_rear_spar: float
    h_front_spar: float
    h_rear_spar: float
    total_area: float
    fuel_tank_area: float
    fuel_area_fraction: float
    y_centroid: float
    i_xx: float
    section_modulus_upper: float
    section_modulus_lower: float
    min_section_modulus: float
    bending_efficiency: float

    # Spar coordinates for visualization
    front_spar_coords: tuple[tuple[float, float], tuple[float, float]]
    rear_spar_coords: tuple[tuple[float, float], tuple[float, float]]


def compute_wingbox_properties(
    airfoil: AirfoilGeometry,
    x_front_spar: float = 0.20,
    x_rear_spar: float = 0.65,
) -> WingboxProperties:
    """
    Compute rigorous aero-structural properties of an airfoil cross-section.

    Parameters
    ----------
    airfoil : AirfoilGeometry
        Airfoil geometry instance with x, y_upper, y_lower.
    x_front_spar : float, default=0.20
        Non-dimensional chordwise station of the front spar (typically 0.15 - 0.25).
    x_rear_spar : float, default=0.65
        Non-dimensional chordwise station of the rear spar (typically 0.60 - 0.75).

    Returns
    -------
    WingboxProperties
        Dataclass containing geometric depths, enclosed areas, and stiffness moments.
    """
    if not (0.05 <= x_front_spar < x_rear_spar <= 0.95):
        raise ValueError(f"Invalid spar stations: front={x_front_spar}, rear={x_rear_spar}")

    x = np.asarray(airfoil.x, dtype=float)
    yu = np.asarray(airfoil.y_upper, dtype=float)
    yl = np.asarray(airfoil.y_lower, dtype=float)

    # 1. Interpolate spar heights at specified chordwise stations
    yu_front = float(np.interp(x_front_spar, x, yu))
    yl_front = float(np.interp(x_front_spar, x, yl))
    h_front = max(0.0, yu_front - yl_front)

    yu_rear = float(np.interp(x_rear_spar, x, yu))
    yl_rear = float(np.interp(x_rear_spar, x, yl))
    h_rear = max(0.0, yu_rear - yl_rear)

    # 2. Total cross-sectional area: A_total = \int (yu - yl) dx
    thickness_dist = np.maximum(0.0, yu - yl)
    total_area = float(np.trapezoid(thickness_dist, x))

    # 3. Inter-spar fuel tank area: A_fuel = \int_{x_front}^{x_rear} (yu - yl) dx
    mask_tank = (x >= x_front_spar) & (x <= x_rear_spar)
    x_tank = np.concatenate([[x_front_spar], x[mask_tank], [x_rear_spar]])
    x_tank = np.unique(x_tank)
    yu_tank = np.interp(x_tank, x, yu)
    yl_tank = np.interp(x_tank, x, yl)
    fuel_area = float(np.trapezoid(np.maximum(0.0, yu_tank - yl_tank), x_tank))
    fuel_fraction = fuel_area / max(total_area, 1e-9)

    # 4. Neutral Axis (Centroid) vertical coordinate y_bar:
    # \bar{y} = (1/2 \int (yu^2 - yl^2) dx) / A_total
    first_moment_y = 0.5 * np.trapezoid(yu**2 - yl**2, x)
    y_bar = float(first_moment_y / max(total_area, 1e-9))

    # 5. Second Moment of Area (Moment of Inertia) I_xx about neutral axis:
    # I_xx = \int_0^1 [(yu - y_bar)^3 - (yl - y_bar)^3] / 3 dx
    d_ixx = ((yu - y_bar) ** 3 - (yl - y_bar) ** 3) / 3.0
    i_xx = float(np.trapezoid(np.maximum(0.0, d_ixx), x))

    # 6. Section Modulus Z = I_xx / y_max (resistance to extreme fiber bending stress)
    c_upper = max(float(np.max(yu)) - y_bar, 1e-6)
    c_lower = max(y_bar - float(np.min(yl)), 1e-6)
    z_upper = i_xx / c_upper
    z_lower = i_xx / c_lower
    z_min = min(z_upper, z_lower)

    # 7. Non-dimensional Structural Bending Efficiency: \eta = I_xx / (A_total^2)
    bending_eff = float(i_xx / max(total_area**2, 1e-9))

    return WingboxProperties(
        x_front_spar=x_front_spar,
        x_rear_spar=x_rear_spar,
        h_front_spar=h_front,
        h_rear_spar=h_rear,
        total_area=total_area,
        fuel_tank_area=fuel_area,
        fuel_area_fraction=fuel_fraction,
        y_centroid=y_bar,
        i_xx=i_xx,
        section_modulus_upper=z_upper,
        section_modulus_lower=z_lower,
        min_section_modulus=z_min,
        bending_efficiency=bending_eff,
        front_spar_coords=((x_front_spar, yl_front), (x_front_spar, yu_front)),
        rear_spar_coords=((x_rear_spar, yl_rear), (x_rear_spar, yu_rear)),
    )
