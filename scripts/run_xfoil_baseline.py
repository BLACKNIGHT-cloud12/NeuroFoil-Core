#!/usr/bin/env python3
"""
XFOIL baseline runner.

Generates classical XFOIL polars for the same airfoils/conditions as the CFD
campaign, producing a baseline CSV (cl, cd, cm per case) for the paper's
comparison tables. XFOIL is transition-aware — exactly what fully-turbulent
RANS is missing — so this also anchors the viscous-drag discussion.

Two supported backends (auto-detected):
  1. the compiled `xfoil` python module (pip install xfoil, needs a Fortran
     toolchain — usually only older Python versions have wheels);
  2. the standalone `xfoil` executable on PATH (official XFOIL 6.99/7.x build),
     driven through a scripted input file.

Usage:
  python scripts/run_xfoil_baseline.py --codes 0012,2412 --alphas "-6,-2,2,6,10,14" --re 6e6
  python scripts/run_xfoil_baseline.py --from-dataset   # all airfoils in the campaign CSV
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.geometry.naca import generate_naca4_airfoil
from src.dataset.cst_sampler import geometry_from_cst

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "xfoil_baseline.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the XFOIL baseline polars.")
    p.add_argument("--codes", type=str, default="0012,2412,4412")
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14")
    p.add_argument("--re", type=float, default=6.0e6)
    p.add_argument("--mach", type=float, default=0.0)
    p.add_argument("--from-dataset", action="store_true",
                   help="Run every distinct airfoil in the campaign CSV instead")
    p.add_argument("--itrmax", type=int, default=100, help="Max xfoil iterations")
    return p.parse_args()


def _write_dat(geom, path: Path) -> None:
    geom.save_dat(path)


def _run_xfoil_exec(dat_path: Path, alphas: list[float], re: float, mach: float,
                    itrmax: int) -> pd.DataFrame:
    """Drive the standalone xfoil executable with a scripted session."""
    exe = shutil.which("xfoil") or shutil.which("xfoil.exe")
    if exe is None:
        raise FileNotFoundError(
            "XFOIL executable not on PATH. Install the official XFOIL build "
            "(https://web.mit.edu/drela/Public/web/xfoil/) or `pip install xfoil` "
            "with a Fortran toolchain."
        )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        polar_file = td_path / "polar.txt"
        script = (
            f"LOAD {dat_path.resolve().as_posix()}\n"
            f"OPER\n"
            f"VISC {re:.0f}\n"
            f"MACH {mach:.2f}\n"
            f"ITER {itrmax}\n"
            f"PACC\n{polar_file.resolve().as_posix()}\n\n"
            + "".join(f"ALFA {a:.2f}\n" for a in alphas)
            + "PACC\n\nQUIT\n"
        )
        script_file = td_path / "session.in"
        script_file.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [exe], stdin=open(script_file), capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0 or not polar_file.exists():
            raise RuntimeError(f"xfoil failed: {proc.stderr[-500:]}")

        rows = []
        for line in polar_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    vals = [float(v) for v in parts[:13]]
                except ValueError:
                    continue  # header
                rows.append(vals)
        if not rows:
            raise RuntimeError("xfoil produced an empty polar file")
        cols = ["alpha", "cd", "cl", "cm", "xcp_top", "xcp_bot",
                "Cpmin", "Chinge", "XCpmin", "XCpHinge", "bl1", "bl2", "bl3"][: len(rows[0])]
        return pd.DataFrame(rows, columns=cols)


def run_case(geom, alphas: list[float], re: float, mach: float, itrmax: int) -> pd.DataFrame:
    """Run one airfoil through the available backend."""
    try:
        from xfoil import XFoil  # compiled module fast path

        xf = XFoil()
        xf.Re = re
        xf.Mach = mach
        xf.max_iter = itrmax
        x, y = geom.to_closed_contour().T
        xf.airfoil = np.column_stack([x, y])
        records = []
        for a in alphas:
            cl, cd, cm, _ = xf.a(a)
            records.append((a, cd, cl, cm))
        return pd.DataFrame(records, columns=["alpha", "cd", "cl", "cm"])
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            dat = Path(td) / "airfoil.dat"
            _write_dat(geom, dat)
            return _run_xfoil_exec(dat, alphas, re, mach, itrmax)


def main() -> None:
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    # Build the (shape_id, geometry) work list
    shapes: list[tuple[str, object]] = []
    if args.from_dataset:
        import json

        csv = PROJECT_ROOT / "data" / "processed" / "airfoil_cfd_dataset.csv"
        df = pd.read_csv(csv)
        cst_cols_u = [f"cst_u_{i}" for i in range(6)]
        cst_cols_l = [f"cst_l_{i}" for i in range(6)]
        for sid, grp in df.groupby("naca_code"):
            row = grp.iloc[0]
            if str(sid).startswith("CST-"):
                geom = geometry_from_cst(
                    tuple(row[cst_cols_u]), tuple(row[cst_cols_l])
                )
            else:
                geom = generate_naca4_airfoil(str(sid), 161, True)
            shapes.append((str(sid), geom))
    else:
        for code in [c.strip() for c in args.codes.split(",") if c.strip()]:
            shapes.append((code, generate_naca4_airfoil(code, 161, True)))

    all_rows = []
    for i, (sid, geom) in enumerate(shapes, 1):
        print(f"[{i}/{len(shapes)}] XFOIL: {sid} ...", flush=True)
        try:
            dfp = run_case(geom, alphas, args.re, args.mach, args.itrmax)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        dfp.insert(0, "shape_id", sid)
        dfp.insert(1, "reynolds", args.re)
        all_rows.append(dfp)

    if not all_rows:
        print("No XFOIL results collected.")
        return
    out = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
