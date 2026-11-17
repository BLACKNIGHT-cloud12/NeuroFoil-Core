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
  python scripts/run_xfoil_baseline.py --from-dataset   # every (airfoil, Re, alpha) combination in the campaign CSV
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
                   help="Run every (airfoil, Reynolds, alpha) combination present in the campaign CSV instead")
    p.add_argument("--itrmax", type=int, default=100, help="Max xfoil iterations")
    return p.parse_args()


def _write_dat(geom, path: Path) -> None:
    geom.save_dat(path)


_CANON = {
    "alpha": "alpha",
    "alfa": "alpha",
    "cl": "cl",
    "cd": "cd",
    "cdp": "cdp",
    "cm": "cm",
    "top_xtr": "top_xtr",
    "topxtr": "top_xtr",
    "xtr_top": "top_xtr",
    "bot_xtr": "bot_xtr",
    "botxtr": "bot_xtr",
    "bottom_xtr": "bot_xtr",
    "xtr_bot": "bot_xtr",
    "cpmin": "cpmin",
    "xcpmin": "xcpmin",
    "chinge": "chinge",
}
_FALLBACK_POLAR_COLUMNS = ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]


def _canonical_polar_name(token: str) -> str:
    cleaned = token.strip().strip("#,:;|").lower().replace("-", "_")
    return _CANON.get(cleaned, cleaned)


def _polar_column_names(polar_path: Path) -> list[str] | None:
    """Detect XFOIL's column header, whether plain or comment-prefixed."""
    with polar_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            tokens = [_canonical_polar_name(token) for token in stripped.split()]
            if "alpha" in tokens and "cl" in tokens and "cd" in tokens:
                # Some builds prefix explanatory text before the actual header.
                return tokens[tokens.index("alpha"):]
    return None


def _polar_rows(polar_path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in polar_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(value.replace("D", "E").replace("d", "e")) for value in parts])
        except ValueError:
            continue
    return rows


def _polar_frame(polar_path: Path) -> pd.DataFrame:
    """Parse an XFOIL polar while keeping header and numeric widths consistent."""
    rows = _polar_rows(polar_path)
    if not rows:
        raise RuntimeError("xfoil produced an empty polar file")

    names = _polar_column_names(polar_path) or list(_FALLBACK_POLAR_COLUMNS)
    width = min(len(names), min(len(row) for row in rows))
    if width < 5:
        raise RuntimeError("xfoil polar has fewer than five usable columns")
    return pd.DataFrame([row[:width] for row in rows], columns=names[:width])


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
        with script_file.open(encoding="utf-8") as script_input:
            proc = subprocess.run(
                [exe], stdin=script_input, capture_output=True, text=True, timeout=300
            )
        if proc.returncode != 0 or not polar_file.exists():
            raise RuntimeError(f"xfoil failed: {proc.stderr[-500:]}")
        return _polar_frame(polar_file)


def run_case(geom, alphas: list[float], re: float, mach: float, itrmax: int) -> pd.DataFrame:
    """Run one airfoil through the available backend."""
    try:
        from xfoil import XFoil  # compiled module fast path
        from xfoil.model import Airfoil

        xf = XFoil()
        xf.Re = re
        xf.Mach = mach
        xf.max_iter = itrmax
        x, y = geom.to_closed_contour().T
        xf.airfoil = Airfoil(x, y)
        records = []
        for a in alphas:
            result = xf.a(a)
            if len(result) < 3:
                raise RuntimeError("compiled xfoil backend returned fewer than three coefficients")
            cl, cd, cm = result[:3]
            records.append((a, cl, cd, cm))
        return pd.DataFrame(records, columns=["alpha", "cl", "cd", "cm"])
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            dat = Path(td) / "airfoil.dat"
            _write_dat(geom, dat)
            return _run_xfoil_exec(dat, alphas, re, mach, itrmax)


def main() -> None:
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    # Build the (shape_id, geometry, reynolds, alphas) work list. Dataset mode
    # reproduces the campaign's operating conditions per geometry instead of
    # collapsing everything onto the CLI's single Reynolds/alpha grid.
    if args.from_dataset:
        csv = PROJECT_ROOT / "data" / "processed" / "airfoil_cfd_dataset.csv"
        df = pd.read_csv(csv)
        df = df.loc[df["naca_code"].notna()]
        cst_cols_u = [f"cst_u_{i}" for i in range(6)]
        cst_cols_l = [f"cst_l_{i}" for i in range(6)]
        work: list[tuple[str, object, float, list[float]]] = []
        for (sid, re_value), grp in df.groupby(["naca_code", "reynolds"]):
            alphas_ds = sorted(float(a) for a in grp["alpha_deg"].unique())
            geom_row = None
            if str(sid).startswith("CST-"):
                complete = grp.dropna(subset=cst_cols_u + cst_cols_l)
                if not complete.empty:
                    geom_row = complete.iloc[0]
            else:
                geom_row = grp.iloc[0]
            if geom_row is None:
                print(f"    SKIP {sid}: no row with complete CST coefficients")
                continue
            if str(sid).startswith("CST-"):
                geom = geometry_from_cst(
                    tuple(float(geom_row[col]) for col in cst_cols_u),
                    tuple(float(geom_row[col]) for col in cst_cols_l),
                )
            else:
                geom = generate_naca4_airfoil(str(sid), 161, True)
            work.append((str(sid), geom, float(re_value), alphas_ds))
    else:
        work = [
            (code, generate_naca4_airfoil(code, 161, True), args.re, alphas)
            for code in [c.strip() for c in args.codes.split(",") if c.strip()]
        ]

    all_rows = []
    for i, (sid, geom, re_value, alphas_item) in enumerate(work, 1):
        print(
            f"[{i}/{len(work)}] XFOIL: {sid} "
            f"(Re={re_value:.1e}, {len(alphas_item)} alphas) ...",
            flush=True,
        )
        try:
            dfp = run_case(geom, alphas_item, re_value, args.mach, args.itrmax)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        dfp.insert(0, "shape_id", sid)
        dfp.insert(1, "reynolds", re_value)
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
