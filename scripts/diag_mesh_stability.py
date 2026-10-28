#!/usr/bin/env python3
"""Diagnostic bisection for the NACA0012 alpha=4 divergence: mesh vs numerics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import subprocess

import numpy as np

from src.config import load_config, windows_to_wsl_path
from src.geometry.naca import generate_naca4_airfoil
from src.cfd.mesh_generator import generate_blockmesh_dict, run_blockmesh_and_checkmesh
from src.cfd.case_builder import OperatingPoint, build_openfoam_case
from src.cfd.convergence import parse_force_coeffs_file

CFG = load_config()
MESH_CFG = CFG["mesh"]
MED = MESH_CFG["levels"]["medium"]
WSL = CFG["wsl_environment"]


def run_experiment(tag: str, case_dir: Path, first_cell: float | None,
                   span: float, relax: tuple[float, float, float],
                   expansion_ratio: float | None = None, iters: int = 300,
                   nut_bc: str | None = None) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    airfoil = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    kwargs = dict(
        airfoil=airfoil, case_dir=case_dir, level="medium",
        farfield_radius=float(MESH_CFG["farfield_radius_chords"]),
        wake_length=float(MESH_CFG["wake_length_chords"]),
        span_thickness=span,
        n_airfoil_side=int(MED["n_airfoil_side"]),
        n_wake=int(MED["n_wake"]),
        n_normal=int(MED["n_normal"]),
    )
    if expansion_ratio is not None:
        kwargs["normal_expansion_ratio"] = expansion_ratio
    else:
        kwargs["first_cell_height"] = first_cell
    meta = generate_blockmesh_dict(**kwargs)
    run_blockmesh_and_checkmesh(case_dir, meta, wsl_distro=WSL["distro"],
                                openfoam_bashrc=WSL["openfoam_bashrc"])

    op = OperatingPoint(alpha_deg=4.0, reynolds=6.0e6)
    cfg = dict(CFG)
    build_openfoam_case(case_dir, op, cfg)

    # Override relaxation factors + iterations
    fv_sol = case_dir / "system" / "fvSolution"
    txt = fv_sol.read_text()
    txt = txt.replace("p               0.3;", f"p               {relax[0]:.2f};")
    txt = txt.replace("U               0.7;", f"U               {relax[1]:.2f};")
    txt = txt.replace("k               0.7;", f"k               {relax[2]:.2f};")
    txt = txt.replace("omega           0.7;", f"omega           {relax[2]:.2f};")
    fv_sol.write_text(txt)
    ctrl = case_dir / "system" / "controlDict"
    txt = ctrl.read_text().replace("endTime         1500;", f"endTime         {iters};")
    txt = txt.replace("writeInterval   1500;", f"writeInterval   {iters};")
    ctrl.write_text(txt)

    if nut_bc is not None:
        nut_file = case_dir / "0" / "nut"
        txt = nut_file.read_text().replace("type            nutkWallFunction;", f"type            {nut_bc};")
        nut_file.write_text(txt)

    wsl_case = windows_to_wsl_path(case_dir)
    tmp = f"/tmp/aero_diag_{tag}"
    cmd = (
        f"source {WSL['openfoam_bashrc']} >/dev/null 2>&1 && "
        f"rm -rf '{tmp}' && mkdir -p '{tmp}' && "
        f"cp -r '{wsl_case}/0' '{wsl_case}/constant' '{wsl_case}/system' '{tmp}/' && "
        f"cd '{tmp}' && blockMesh > log.blockMesh 2>&1 && "
        f"simpleFoam > log.simpleFoam 2>&1; "
        f"cp log.simpleFoam '{wsl_case}/' && "
        f"mkdir -p '{wsl_case}/postProcessing' && cp -r postProcessing/* '{wsl_case}/postProcessing/' 2>/dev/null; "
        f"rm -rf '{tmp}'"
    )
    subprocess.run(["wsl", "-d", WSL["distro"], "bash", "-c", cmd],
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)

    dat = case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    if not dat.exists():
        print(f"{tag:<28} -> NO forceCoeffs OUTPUT (solver failed)")
        return
    coeffs = parse_force_coeffs_file(dat)
    cl, cd = np.asarray(coeffs["cl"]), np.asarray(coeffs["cd"])
    w = min(100, len(cl))
    print(
        f"{tag:<28} -> iters={len(cl):4d}  CL={cl[-w:].mean():+8.4f}±{cl[-w:].std():.4f}  "
        f"CD={cd[-w:].mean():+8.5f}±{cd[-w:].std():.5f}"
    )


if __name__ == "__main__":
    cases_dir = PROJECT_ROOT / "data" / "raw_cases" / "_diag"
    cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'experiment':<28}  result")
    run_experiment("y1_4e-6_heavyrelax_1500", cases_dir / "r2_relax", 4.0e-6, 0.1,
                   (0.1, 0.3, 0.5), iters=1500)
    run_experiment("y1_4e-6_spalding_1500", cases_dir / "r2_spalding", 4.0e-6, 0.1,
                   (0.3, 0.7, 0.7), iters=1500, nut_bc="nutUSpaldingWallFunction")
    run_experiment("oldmesh_heavyrelax_1500", cases_dir / "r2_old", None, 0.1,
                   (0.1, 0.3, 0.5), expansion_ratio=25000.0, iters=1500)
