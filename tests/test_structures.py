"""
Unit tests for Aero-Structural Wingbox Mechanics.
"""

import pytest
import numpy as np

from src.geometry.naca import generate_naca4_airfoil
from src.geometry.structures import compute_wingbox_properties, WingboxProperties


def test_wingbox_symmetric_naca0012():
    """Verify structural properties of standard NACA 0012."""
    af = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    wb = compute_wingbox_properties(af, x_front_spar=0.20, x_rear_spar=0.65)

    assert isinstance(wb, WingboxProperties)
    # Symmetric airfoil must have centroid on chordline y=0
    assert abs(wb.y_centroid) < 1e-4

    # Spar web heights: at x=0.20 thickness is ~0.115, at x=0.65 thickness is ~0.082
    assert 0.110 < wb.h_front_spar < 0.120
    assert 0.075 < wb.h_rear_spar < 0.085

    # Enclosed areas
    assert 0.075 < wb.total_area < 0.090
    assert 0.040 < wb.fuel_tank_area < 0.055
    assert 0.50 < wb.fuel_area_fraction < 0.65

    # Moment of inertia and section modulus
    assert wb.i_xx > 0.0
    assert wb.min_section_modulus > 0.0
    assert wb.bending_efficiency > 0.0


def test_wingbox_cambered_naca2412():
    """Verify that positive camber shifts the neutral axis upward."""
    af_sym = generate_naca4_airfoil("0012", num_points=161)
    af_cam = generate_naca4_airfoil("2412", num_points=161)

    wb_sym = compute_wingbox_properties(af_sym)
    wb_cam = compute_wingbox_properties(af_cam)

    # Positive camber must shift neutral axis upward (y_centroid > 0)
    assert wb_cam.y_centroid > wb_sym.y_centroid
    assert wb_cam.y_centroid > 0.005


def test_invalid_spar_stations():
    """Verify error handling on invalid spar placement."""
    af = generate_naca4_airfoil("0012")
    with pytest.raises(ValueError):
        compute_wingbox_properties(af, x_front_spar=0.70, x_rear_spar=0.30)
