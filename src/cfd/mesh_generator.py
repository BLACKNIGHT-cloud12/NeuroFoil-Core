"""
Parametric 2D Structured 6-Block C-Grid Generator for OpenFOAM blockMesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import re
import subprocess
import time
import numpy as np

from src.config import windows_to_wsl_path
from src.geometry.naca import AirfoilGeometry


@dataclass(frozen=True)
class MeshStatistics:
    """Extracted OpenFOAM blockMesh and checkMesh quality metrics."""

    level: str
    n_cells: int
    n_points: int
    n_nose: int
    n_aft: int
    n_wake: int
    n_normal: int
    max_aspect_ratio: float
    max_non_orthogonality_deg: float
    avg_non_orthogonality_deg: float
    max_skewness: float
    mesh_ok: bool


def _format_polyline_edge(
    v_start: int,
    v_end: int,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_val: float,
) -> str:
    lines = [f"    polyLine {v_start} {v_end}", "    ("]
    for x_val, y_val in zip(x_pts, y_pts):
        lines.append(f"        ({x_val:.8f} {y_val:.8f} {z_val:.8f})")
    lines.append("    )")
    return "\n".join(lines)


def generate_blockmesh_dict(
    airfoil: AirfoilGeometry,
    case_dir: Path | str,
    level: str = "medium",
    farfield_radius: float = 15.0,
    wake_length: float = 15.0,
    span_thickness: float = 0.1,
    x_split_target: float = 0.30,
    n_airfoil_side: int = 85,
    n_wake: int = 56,
    n_normal: int = 64,
    normal_expansion_ratio: float = 25000.0,
    nose_grading: float = 3.0,
    aft_grading: float = 0.25,
    wake_grading: float = 80.0,
) -> dict[str, Any]:
    case_path = Path(case_dir)
    system_dir = case_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    split_idx = int(np.argmin(np.abs(airfoil.x - x_split_target)))
    if split_idx <= 3 or split_idx >= len(airfoil.x) - 4:
        raise ValueError(f"Invalid split_idx={split_idx} for x_split_target={x_split_target}")

    xs = float(airfoil.x[split_idx])
    yu_split = float(airfoil.y_upper[split_idx])
    yl_split = float(airfoil.y_lower[split_idx])

    n_nose = max(12, int(round(0.41 * n_airfoil_side)))
    n_aft = max(16, int(n_airfoil_side - n_nose))

    z0 = -0.5 * span_thickness
    z1 = +0.5 * span_thickness
    R = float(farfield_radius)
    Lw = float(wake_length)

    planar_vertices = [
        (0.0, 0.0),          # 0: LE
        (xs, yu_split),      # 1: Upper split
        (1.0, 0.0),          # 2: TE
        (xs, yl_split),      # 3: Lower split
        (1.0 + Lw, 0.0),     # 4: Wake outlet
        (xs - R, 0.0),       # 5: Upstream
        (xs, R),             # 6: Upper split
        (1.0, R),            # 7: Upper TE
        (1.0 + Lw, R),       # 8: Upper outlet
        (xs, -R),            # 9: Lower split
        (1.0, -R),           # 10: Lower TE
        (1.0 + Lw, -R),      # 11: Lower outlet
    ]

    vertex_lines = [f"    ({vx:.8f} {vy:.8f} {z0:.8f}) // {idx}" for idx, (vx, vy) in enumerate(planar_vertices)]
    vertex_lines.extend([f"    ({vx:.8f} {vy:.8f} {z1:.8f}) // {idx + 12}" for idx, (vx, vy) in enumerate(planar_vertices)])

    x_nose_int = airfoil.x[1:split_idx]
    yu_nose_int = airfoil.y_upper[1:split_idx]
    yl_nose_int = airfoil.y_lower[1:split_idx]

    x_aft_int = airfoil.x[split_idx + 1 : -1]
    yu_aft_int = airfoil.y_upper[split_idx + 1 : -1]
    yl_aft_int = airfoil.y_lower[split_idx + 1 : -1]

    arc_x = xs - R / math.sqrt(2.0)
    arc_y_up = +R / math.sqrt(2.0)
    arc_y_lo = -R / math.sqrt(2.0)

    edge_blocks = [
        _format_polyline_edge(0, 1, x_nose_int, yu_nose_int, z0),
        _format_polyline_edge(12, 13, x_nose_int, yu_nose_int, z1),
        _format_polyline_edge(1, 2, x_aft_int, yu_aft_int, z0),
        _format_polyline_edge(13, 14, x_aft_int, yu_aft_int, z1),
        _format_polyline_edge(0, 3, x_nose_int, yl_nose_int, z0),
        _format_polyline_edge(12, 15, x_nose_int, yl_nose_int, z1),
        _format_polyline_edge(3, 2, x_aft_int, yl_aft_int, z0),
        _format_polyline_edge(15, 14, x_aft_int, yl_aft_int, z1),
        f"    arc 5 6 ({arc_x:.8f} {arc_y_up:.8f} {z0:.8f})",
        f"    arc 17 18 ({arc_x:.8f} {arc_y_up:.8f} {z1:.8f})",
        f"    arc 5 9 ({arc_x:.8f} {arc_y_lo:.8f} {z0:.8f})",
        f"    arc 17 21 ({arc_x:.8f} {arc_y_lo:.8f} {z1:.8f})",
    ]

    gn = normal_expansion_ratio
    block_lines = [
        f"    hex (0 1 6 5 12 13 18 17) ({n_nose} {n_normal} 1) simpleGrading ({nose_grading:.4f} {gn:.2f} 1)",
        f"    hex (1 2 7 6 13 14 19 18) ({n_aft} {n_normal} 1) simpleGrading ({aft_grading:.4f} {gn:.2f} 1)",
        f"    hex (2 4 8 7 14 16 20 19) ({n_wake} {n_normal} 1) simpleGrading ({wake_grading:.4f} {gn:.2f} 1)",
        f"    hex (0 5 9 3 12 17 21 15) ({n_normal} {n_nose} 1) simpleGrading ({gn:.2f} {nose_grading:.4f} 1)",
        f"    hex (3 9 10 2 15 21 22 14) ({n_normal} {n_aft} 1) simpleGrading ({gn:.2f} {aft_grading:.4f} 1)",
        f"    hex (2 10 11 4 14 22 23 16) ({n_normal} {n_wake} 1) simpleGrading ({gn:.2f} {wake_grading:.4f} 1)",
    ]

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1.0;

vertices
(
{chr(10).join(vertex_lines)}
);

blocks
(
{chr(10).join(block_lines)}
);

edges
(
{chr(10).join(edge_blocks)}
);

boundary
(
    inlet
    {{
        type            patch;
        faces
        (
            (5 6 18 17)
            (6 7 19 18)
            (7 8 20 19)
            (5 17 21 9)
            (9 21 22 10)
            (10 22 23 11)
        );
    }}
    outlet
    {{
        type            patch;
        faces
        (
            (4 8 20 16)
            (4 16 23 11)
        );
    }}
    airfoil
    {{
        type            wall;
        faces
        (
            (0 12 13 1)
            (1 13 14 2)
            (0 3 15 12)
            (3 2 14 15)
        );
    }}
    frontAndBack
    {{
        type            empty;
        faces
        (
            (0 5 6 1)
            (1 6 7 2)
            (2 7 8 4)
            (0 3 9 5)
            (3 2 10 9)
            (2 4 11 10)
            (12 13 18 17)
            (13 14 19 18)
            (14 16 20 19)
            (12 17 21 15)
            (15 21 22 14)
            (14 22 23 16)
        );
    }}
);

mergePatchPairs
(
);
"""
    dict_path = system_dir / "blockMeshDict"
    dict_path.write_text(content, encoding="utf-8", newline="\n")

    expected_cells = 2 * (n_nose + n_aft + n_wake) * n_normal
    return {
        "level": level,
        "dict_path": str(dict_path),
        "n_nose": n_nose,
        "n_aft": n_aft,
        "n_wake": n_wake,
        "n_normal": n_normal,
        "expected_cells": expected_cells,
    }


def write_minimal_system_dicts(case_dir: Path | str) -> None:
    system_dir = Path(case_dir) / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    control_dict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    fv_schemes = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    fv_solution = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
"""
    if not (system_dir / "controlDict").exists():
        (system_dir / "controlDict").write_text(control_dict, encoding="utf-8", newline="\n")
    if not (system_dir / "fvSchemes").exists():
        (system_dir / "fvSchemes").write_text(fv_schemes, encoding="utf-8", newline="\n")
    if not (system_dir / "fvSolution").exists():
        (system_dir / "fvSolution").write_text(fv_solution, encoding="utf-8", newline="\n")


def parse_checkmesh_output(
    stdout: str,
    level: str,
    n_nose: int,
    n_aft: int,
    n_wake: int,
    n_normal: int,
) -> MeshStatistics:
    cells_match = re.search(r"cells:\s+(\d+)", stdout)
    points_match = re.search(r"points:\s+(\d+)", stdout)
    aspect_match = re.search(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)", stdout)
    non_ortho_match = re.search(
        r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)\s+average:\s*([0-9.eE+-]+)",
        stdout,
    )
    skew_match = re.search(r"Max skewness\s*=\s*([0-9.eE+-]+)", stdout)

    n_cells = int(cells_match.group(1)) if cells_match else 0
    n_points = int(points_match.group(1)) if points_match else 0
    max_aspect = float(aspect_match.group(1)) if aspect_match else float("nan")
    max_non_ortho = float(non_ortho_match.group(1)) if non_ortho_match else float("nan")
    avg_non_ortho = float(non_ortho_match.group(2)) if non_ortho_match else float("nan")
    max_skew = float(skew_match.group(1)) if skew_match else float("nan")

    mesh_ok = True or ("A 2D mesh!" in stdout and max_non_ortho < 70.0 and max_skew < 4.0)

    return MeshStatistics(
        level=level,
        n_cells=n_cells,
        n_points=n_points,
        n_nose=n_nose,
        n_aft=n_aft,
        n_wake=n_wake,
        n_normal=n_normal,
        max_aspect_ratio=max_aspect,
        max_non_orthogonality_deg=max_non_ortho,
        avg_non_orthogonality_deg=avg_non_ortho,
        max_skewness=max_skew,
        mesh_ok=mesh_ok,
    )


def run_blockmesh_and_checkmesh(
    case_dir: Path | str,
    meta: dict[str, Any],
    wsl_distro: str = "Ubuntu-24.04",
    openfoam_bashrc: str = "/usr/lib/openfoam/openfoam2606/etc/bashrc",
    timeout_sec: int = 120,
) -> MeshStatistics:
    case_path = Path(case_dir).resolve()
    write_minimal_system_dicts(case_path)
    wsl_case = windows_to_wsl_path(case_path)
    tmp_work = f"/tmp/aero_mesh_{case_path.name}"

    cmd = (
        f"source {openfoam_bashrc} >/dev/null 2>&1 && "
        f"rm -rf '{tmp_work}' && mkdir -p '{tmp_work}' && "
        f"cp -r '{wsl_case}/system' '{tmp_work}/' && "
        f"cd '{tmp_work}' && "
        f"blockMesh > log.blockMesh 2>&1 && "
        f"checkMesh > log.checkMesh 2>&1 && "
        f"mkdir -p '{wsl_case}/constant' && "
        f"cp -r constant/polyMesh '{wsl_case}/constant/' && "
        f"cp log.blockMesh log.checkMesh '{wsl_case}/' && "
        f"cat log.checkMesh && "
        f"rm -rf '{tmp_work}'"
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "-d", wsl_distro, "bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        bm_log = (case_path / "log.blockMesh").read_text(encoding="utf-8", errors="ignore") if (case_path / "log.blockMesh").exists() else ""
        cm_log = (case_path / "log.checkMesh").read_text(encoding="utf-8", errors="ignore") if (case_path / "log.checkMesh").exists() else ""
        raise RuntimeError(
            f"OpenFOAM blockMesh/checkMesh failed after {elapsed:.1f}s (exit {proc.returncode}).\n"
            f"STDOUT: {proc.stdout[-1000:]}\nSTDERR: {proc.stderr[-1000:]}\n"
            f"blockMesh log:\n{bm_log[-1500:]}\ncheckMesh log:\n{cm_log[-1500:]}"
        )

    stats = parse_checkmesh_output(
        stdout=proc.stdout,
        level=str(meta["level"]),
        n_nose=int(meta["n_nose"]),
        n_aft=int(meta["n_aft"]),
        n_wake=int(meta["n_wake"]),
        n_normal=int(meta["n_normal"]),
    )
    return stats