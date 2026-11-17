"""Grid-convergence study: NACA 0012, alpha = 4 deg, Re = 6e6 across mesh levels.

Runs the exact campaign pipeline (same geometry, case builder, solver settings,
convergence evaluation, Cp extraction) at coarse/medium/fine mesh levels, one
case at a time, into data/mesh_study/<level>/ (separate from the campaign's
raw_cases so resume logic and the dataset CSV are untouched).

The medium-mesh reference result comes from the validated pipeline check
(CL 0.4392, CD 0.00948, 179 s). Levels are computed sequentially; pass the
levels to run, e.g.:

  python scripts/run_mesh_sensitivity.py --levels coarse,fine
  python scripts/run_mesh_sensitivity.py --levels coarse --timeout 1200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset.pipeline import run_single_cfd_sample
from src.dataset.sampler import AirfoilSample

STUDY_DIR = PROJECT_ROOT / "data" / "mesh_study"
REPORT_PATH = PROJECT_ROOT / "reports" / "mesh_study" / "mesh_sensitivity.json"

# Reference: medium-mesh pipeline check (same operating point, measured 2026-11-17).
MEDIUM_REFERENCE = {"cl": 0.439174, "cd": 0.009479, "cm": 0.002438,
                    "wall_time_s": 178.8, "status": "converged"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mesh sensitivity study (sequential).")
    p.add_argument("--levels", type=str, default="coarse,fine",
                   help="Comma-separated mesh levels to run (medium is the reference)")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-case solver timeout (s); fine mesh needs more than medium")
    p.add_argument("--max-iterations", type=int, default=0,
                   help="Override solver max_iterations for this study (0 = config "
                        "value 3000); fine mesh may need ~6000 to converge")
    return p.parse_args()


def count_cells(case_dir: Path) -> int | None:
    log = case_dir / "log.checkMesh"
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "cells:" in line:
            try:
                return int(line.split("cells:")[1].strip())
            except ValueError:
                return None
    return None


def run_level(level: str, cfg: dict, timeout: int,
              max_iterations: int = 0) -> dict:
    case_dir = STUDY_DIR / level
    case_dir.mkdir(parents=True, exist_ok=True)
    sample = AirfoilSample(
        case_id=f"NACA0012_a+4.0_Re6e6_{level}", naca_code="0012",
        m=0.0, p=0.0, t=0.12, alpha_deg=4.0, reynolds=6.0e6,
    )
    print(f"\n[{level}] solving in {case_dir} ...", flush=True)
    t0 = time.perf_counter()
    original_max_iter = cfg["solver"]["max_iterations"]
    if max_iterations:
        # In-memory override only; the case builder reads this config key.
        cfg["solver"]["max_iterations"] = int(max_iterations)
    try:
        result = run_single_cfd_sample(
            sample, case_dir, cfg, mesh_level=level, timeout_sec=timeout
        )
    finally:
        cfg["solver"]["max_iterations"] = original_max_iter
    elapsed = time.perf_counter() - t0
    record = dict(result.get("record") or {})
    row = {
        "level": level,
        "status": record.get("status", result.get("status", "unknown")),
        "cl": record.get("cl"),
        "cd": record.get("cd"),
        "cm": record.get("cm"),
        "wall_time_s": round(elapsed, 1),
        "cells": count_cells(case_dir),
        "max_iterations": max_iterations or cfg["solver"]["max_iterations"],
    }
    print(f"[{level}] {row['status']}  CL={row['cl']}  CD={row['cd']}  "
          f"CM={row['cm']}  ({row['wall_time_s']}s, {row['cells']} cells)", flush=True)
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config()
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for level in levels:  # strictly sequential, one core at a time
        rows.append(run_level(level, cfg, args.timeout, args.max_iterations))

    summary = {
        "case": "NACA0012, alpha=4.0 deg, Re=6e6, incompressible kOmegaSST",
        "medium_reference": MEDIUM_REFERENCE,
        "runs": rows,
        "note": "Same pipeline as the 504-case campaign; separate case dirs "
                "(data/mesh_study/) keep campaign resume logic untouched.",
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("MESH SENSITIVITY SUMMARY (medium reference: CL 0.4392, CD 0.00948)")
    for row in rows:
        dcl = None if row["cl"] is None else row["cl"] - MEDIUM_REFERENCE["cl"]
        dcd = None if row["cd"] is None else row["cd"] - MEDIUM_REFERENCE["cd"]
        print(f"  {row['level']:<7} cells={row['cells']}  CL={row['cl']} "
              f"(dCL={dcl:+.4f})  CD={row['cd']} (dCD={dcd:+.5f})  "
              f"{row['status']} ({row['wall_time_s']}s)")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
