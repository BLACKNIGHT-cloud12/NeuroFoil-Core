"""
Surface Pressure Coefficient (Cp) and Aerodynamic Field Extractor (Milestone 1.3).

Extracts surface pressure distributions from OpenFOAM polyMesh face geometry
and boundary field values (e.g. <time>/p) or postProcessing surface samples,
resampling onto the airfoil's chordwise grid for ML feature extraction and validation.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import numpy as np

from src.geometry.naca import AirfoilGeometry


def parse_poly_mesh_points(points_file: Path) -> np.ndarray:
    """Parse constant/polyMesh/points into (N_points, 3) float array."""
    text = points_file.read_text(encoding="utf-8", errors="ignore")
    # Find bracketed coordinates: (x y z)
    coords = re.findall(
        r"\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)",
        text,
    )
    return np.array([[float(c[0]), float(c[1]), float(c[2])] for c in coords], dtype=float)


def parse_poly_mesh_boundary(boundary_file: Path, patch_name: str = "airfoil") -> tuple[int, int]:
    """Parse startFace and nFaces for the given patch from constant/polyMesh/boundary."""
    text = boundary_file.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"{patch_name}\s*\{{[^}}]*?nFaces\s+(\d+);[^}}]*?startFace\s+(\d+);"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Fallback order search
        pattern_rev = rf"{patch_name}\s*\{{[^}}]*?startFace\s+(\d+);[^}}]*?nFaces\s+(\d+);"
        match_rev = re.search(pattern_rev, text, re.DOTALL)
        if match_rev:
            return int(match_rev.group(1)), int(match_rev.group(2))
        raise ValueError(f"Could not find patch '{patch_name}' in {boundary_file}")
    n_faces = int(match.group(1))
    start_face = int(match.group(2))
    return start_face, n_faces


def parse_poly_mesh_faces_slice(faces_file: Path, start_face: int, n_faces: int) -> list[list[int]]:
    """Parse a slice of faces [start_face, start_face + n_faces) from constant/polyMesh/faces."""
    lines = faces_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    face_lines: list[str] = []
    in_faces = False

    for line in lines:
        stripped = line.strip()
        if stripped == "(":
            in_faces = True
            continue
        elif stripped == ")":
            break
        elif in_faces and stripped:
            face_lines.append(stripped)

    selected = face_lines[start_face : start_face + n_faces]
    parsed_faces: list[list[int]] = []
    for f_str in selected:
        # Format e.g. 4(12 13 14 15)
        m = re.search(r"\(([^)]+)\)", f_str)
        if m:
            indices = [int(x) for x in m.group(1).split()]
            parsed_faces.append(indices)
    return parsed_faces


def parse_field_patch_values(p_file: Path, patch_name: str = "airfoil") -> np.ndarray:
    """Parse scalar field values (e.g. pressure p) for the given boundary patch."""
    text = p_file.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"{patch_name}\s*\{{[^}}]*?value\s+nonuniform\s+List<scalar>\s*\d+\s*\((.*?)\)\s*;"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        nums = [float(v) for v in match.group(1).split()]
        return np.array(nums, dtype=float)

    # Check for uniform value
    pattern_uniform = rf"{patch_name}\s*\{{[^}}]*?value\s+uniform\s+([0-9.eE+-]+);"
    match_uni = re.search(pattern_uniform, text, re.DOTALL)
    if match_uni:
        val = float(match_uni.group(1))
        return np.array([val], dtype=float)

    return np.array([], dtype=float)


def parse_internal_field(p_file: Path) -> np.ndarray:
    """Parse volScalarField internalField values."""
    text = p_file.read_text(encoding="utf-8", errors="ignore")
    pattern = r"internalField\s+nonuniform\s+List<scalar>\s*\d+\s*\((.*?)\)\s*;"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        nums = [float(v) for v in match.group(1).split()]
        return np.array(nums, dtype=float)
    return np.array([], dtype=float)


def parse_poly_mesh_owners_slice(owner_file: Path, start_face: int, n_faces: int) -> list[int]:
    """Parse cell owner IDs for a slice of faces from constant/polyMesh/owner."""
    lines = owner_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_b = False
    owners: list[int] = []
    for l in lines:
        s = l.strip()
        if s == "(":
            in_b = True
            continue
        elif s == ")":
            break
        elif in_b and s.isdigit():
            owners.append(int(s))
    return owners[start_face : start_face + n_faces]


def extract_surface_cp(
    case_dir: Path | str,
    airfoil: AirfoilGeometry,
    u_inf: float = 1.0,
    p_inf: float = 0.0,
    rho: float = 1.0,
) -> dict[str, Any]:
    """
    Extract surface pressure coefficient Cp(x) along upper and lower surfaces,
    interpolating onto the standard chordwise grid of the AirfoilGeometry.
    """
    case_path = Path(case_dir)
    empty_result: dict[str, Any] = {
        "available": False,
        "x_grid": [],
        "cp_upper": [],
        "cp_lower": [],
        "delta_cp": [],
        "cl_from_cp": 0.0,
    }

    # Locate mesh files
    poly_mesh_dir = case_path / "constant" / "polyMesh"
    boundary_file = poly_mesh_dir / "boundary"
    points_file = poly_mesh_dir / "points"
    faces_file = poly_mesh_dir / "faces"
    owner_file = poly_mesh_dir / "owner"

    if not (boundary_file.exists() and points_file.exists() and faces_file.exists()):
        return empty_result

    # Locate latest time directory containing 'p'
    time_dirs = []
    for d in case_path.iterdir():
        if d.is_dir() and re.match(r"^[0-9]+(?:\.[0-9]+)?$", d.name):
            if (d / "p").exists() and float(d.name) > 0:
                time_dirs.append(d)

    if not time_dirs:
        return empty_result

    time_dirs.sort(key=lambda p: float(p.name))
    latest_p_file = time_dirs[-1] / "p"

    try:
        start_face, n_faces = parse_poly_mesh_boundary(boundary_file, "airfoil")
        p_vals = parse_field_patch_values(latest_p_file, "airfoil")

        # Fallback to zeroGradient owner cell internalField value
        if len(p_vals) != n_faces and owner_file.exists():
            owners_slice = parse_poly_mesh_owners_slice(owner_file, start_face, n_faces)
            p_internal = parse_internal_field(latest_p_file)
            if len(owners_slice) == n_faces and len(p_internal) > 0:
                p_vals = p_internal[owners_slice]

        if len(p_vals) != n_faces:
            return empty_result

        points = parse_poly_mesh_points(points_file)
        faces = parse_poly_mesh_faces_slice(faces_file, start_face, n_faces)

        # Compute 2D face centroids (x, y)
        face_x = np.zeros(n_faces, dtype=float)
        face_y = np.zeros(n_faces, dtype=float)
        for i, face_v_indices in enumerate(faces):
            f_pts = points[face_v_indices]
            face_x[i] = float(np.mean(f_pts[:, 0]))
            face_y[i] = float(np.mean(f_pts[:, 1]))

        # Calculate kinematic Cp: Cp = (p - p_inf) / (0.5 * U_inf^2)
        dyn_p = 0.5 * (u_inf**2)
        cp_faces = (p_vals - p_inf) / max(dyn_p, 1e-15)

        # Split into upper surface (y >= 0) and lower surface (y < 0)
        upper_mask = face_y >= -1e-6
        lower_mask = ~upper_mask

        x_u = face_x[upper_mask]
        cp_u = cp_faces[upper_mask]
        x_l = face_x[lower_mask]
        cp_l = cp_faces[lower_mask]

        if len(x_u) < 2 or len(x_l) < 2:
            return empty_result

        # Sort along chordwise direction
        idx_u = np.argsort(x_u)
        x_u_sorted = x_u[idx_u]
        cp_u_sorted = cp_u[idx_u]

        idx_l = np.argsort(x_l)
        x_l_sorted = x_l[idx_l]
        cp_l_sorted = cp_l[idx_l]

        # Resample onto target airfoil x grid
        x_grid = airfoil.x
        cp_upper_interp = np.interp(x_grid, x_u_sorted, cp_u_sorted)
        cp_lower_interp = np.interp(x_grid, x_l_sorted, cp_l_sorted)
        delta_cp = cp_lower_interp - cp_upper_interp

        # Lift coefficient via pressure integration: CL_cp = int_0^1 (Cp_lower - Cp_upper) dx
        cl_from_cp = float(np.trapezoid(delta_cp, x_grid))

        return {
            "available": True,
            "x_grid": x_grid.tolist(),
            "cp_upper": [float(v) for v in cp_upper_interp],
            "cp_lower": [float(v) for v in cp_lower_interp],
            "delta_cp": [float(v) for v in delta_cp],
            "cl_from_cp": cl_from_cp,
        }

    except Exception as exc:
        return {**empty_result, "error": str(exc)}
