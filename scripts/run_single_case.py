#!/usr/bin/env python3
"""
Milestone 1.3b: End-to-End Single Case CFD Runner.
Generates a NACA 0012 airfoil, builds a 6-block C-grid mesh, sets up OpenFOAM 0/,
constant/, and system/ directories for alpha = 4.0 deg, Re = 6e6, executes simpleFoam
inside WSL, evaluates convergence, and extracts aerodynamic coefficients.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, windows_to_wsl_path
from src.geometry.naca import generate_naca4_airfoil
from src.cfd.mesh_generator import generate_blockmesh_dict, run_blockmesh_and_checkmesh
from src.cfd.case_builder import OperatingPoint, build_openfoam_case
from src.cfd.convergence import evaluate_convergence
from src.cfd.extractor import extract_surface_cp


def main() -> None:
    print("=" * 72)
    print("MILESTONE 1.3b — RUNNING LIVE NAC0012 CFD SIMULATION (alpha=4.0 deg, Re=6e6)")
    print("=" * 72)

    cfg = load_config()
    mesh_cfg = cfg["mesh"]
    med = mesh_cfg["levels"]["medium"]
    eval_only = "--eval-only" in sys.argv

    case_dir = Path("data/raw_cases/NACA0012_alpha4_re6m")
    if not eval_only:
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate geometry
    print("[1/5] Generating NACA 0012 geometry...")
    airfoil = generate_naca4_airfoil("0012", num_points=161, closed_te=True)
    airfoil.save_dat(case_dir / "airfoil.dat")
    airfoil.save_metadata(case_dir / "airfoil_meta.json")

    if not eval_only:
        # 2. Generate mesh
        print("[2/5] Generating 6-block structured C-grid mesh...")
        meta = generate_blockmesh_dict(
            airfoil=airfoil,
            case_dir=case_dir,
            level="medium",
            farfield_radius=float(mesh_cfg["farfield_radius_chords"]),
            wake_length=float(mesh_cfg["wake_length_chords"]),
            span_thickness=float(mesh_cfg["span_thickness_m"]),
            first_cell_height=float(mesh_cfg.get("first_cell_height_m", 4.0e-6)),
            n_airfoil_side=int(med["n_airfoil_side"]),
            n_wake=int(med["n_wake"]),
            n_normal=int(med["n_normal"]),
        )
        stats = run_blockmesh_and_checkmesh(
            case_dir=case_dir,
            meta=meta,
            wsl_distro=cfg["wsl_environment"]["distro"],
            openfoam_bashrc=cfg["wsl_environment"]["openfoam_bashrc"],
        )
        print(f"      Mesh generated successfully: {stats.n_cells} cells, max non-ortho = {stats.max_non_orthogonality_deg:.2f}°")

        # 3. Build OpenFOAM case files (0/, constant/, system/)
        print("[3/5] Building OpenFOAM case dictionaries (alpha = 4.0°, Re = 6.0e6)...")
        op = OperatingPoint(alpha_deg=4.0, reynolds=6.0e6, u_inf=1.0, chord_m=1.0)
        build_openfoam_case(case_dir, op, cfg)

        # 4. Execute simpleFoam inside WSL (running in /tmp for speed, syncing back results)
        print("[4/5] Executing simpleFoam inside WSL (this will take ~30-60 seconds)...")
        wsl_case = windows_to_wsl_path(case_dir)
        tmp_work = f"/tmp/aero_run_{case_dir.name}"
        openfoam_bashrc = cfg["wsl_environment"]["openfoam_bashrc"]
        timeout_sec = 300

        cmd = (
            f"source {openfoam_bashrc} >/dev/null 2>&1 && "
            f"rm -rf '{tmp_work}' && mkdir -p '{tmp_work}' && "
            f"cp -r '{wsl_case}/0' '{wsl_case}/constant' '{wsl_case}/system' '{tmp_work}/' && "
            f"cd '{tmp_work}' && "
            f"blockMesh > log.blockMesh 2>&1 && "
            f"potentialFoam > log.potentialFoam 2>&1 && "
            f"simpleFoam > log.simpleFoam 2>&1 && "
            f"mkdir -p '{wsl_case}/postProcessing' && "
            f"cp -r postProcessing '{wsl_case}/' && "
            f"cp log.simpleFoam '{wsl_case}/' && "
            f"cp -r [1-9]* '{wsl_case}/' 2>/dev/null || true && "
            f"rm -rf '{tmp_work}'"
        )

        t0 = time.perf_counter()
        proc = subprocess.run(
            ["wsl", "-d", cfg["wsl_environment"]["distro"], "bash", "-c", cmd],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        wall_time = time.perf_counter() - t0

        if proc.returncode != 0:
            log_simple = (case_dir / "log.simpleFoam").read_text(encoding="utf-8", errors="ignore") if (case_dir / "log.simpleFoam").exists() else ""
            raise RuntimeError(
                f"simpleFoam failed after {wall_time:.1f}s (exit {proc.returncode}).\n"
                f"STDERR: {proc.stderr}\nlog.simpleFoam tail:\n{log_simple[-2000:]}"
            )
        print(f"      Simulation completed in {wall_time:.1f} seconds wall-clock time.")
    else:
        print("[2/5 - 4/5] Skipping mesh generation and CFD solver (--eval-only mode)...")

    # 5. Evaluate convergence and extract aerodynamic outputs
    print("[5/5] Evaluating convergence and extracting force coefficients & Cp(x)...")
    conv = evaluate_convergence(case_dir, window_size=100)
    cp_data = extract_surface_cp(case_dir, airfoil)

    print("=" * 72)
    print(r"SIMULATION RESULTS SUMMARY — NACA 0012 ($\alpha = 4.0^\circ$, $Re = 6.0 \times 10^6$)")
    print("=" * 72)
    print(f"Convergence Status     : {conv.status.upper()}")
    print(f"Iterations Completed   : {conv.iterations_completed}")
    print(f"Lift Coefficient (CL)  : {conv.cl_mean:.5f} (std = {conv.cl_std:.2e})")
    print(f"Drag Coefficient (CD)  : {conv.cd_mean:.5f} (std = {conv.cd_std:.2e})")
    print(f"Pitching Moment (Cm)   : {conv.cm_mean:.5f} (std = {conv.cm_std:.2e})")
    print(f"Lift-to-Drag Ratio     : {(conv.cl_mean / conv.cd_mean):.2f}")
    print(f"Surface Cp Extracted   : {cp_data['available']} ({len(cp_data.get('x_grid', []))} points)")
    if conv.reasons:
        print(f"Notes / Warnings       : {conv.reasons}")
    print("=" * 72)


if __name__ == "__main__":
    main()