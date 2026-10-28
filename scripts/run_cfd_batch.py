#!/usr/bin/env python3
"""
Automated Batch CFD Runner across Airfoils and Angles of Attack.
Runs OpenFOAM simpleFoam in WSL2 and compiles results into tabular & field datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import load_config
from src.dataset.sampler import AirfoilSample, generate_canonical_benchmark_cases, sample_latin_hypercube
from src.dataset.pipeline import run_single_cfd_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch OpenFOAM CFD simulations.")
    parser.add_argument("--mode", choices=["canonical", "lhs"], default="canonical", help="Sampling strategy")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of LHS samples (if mode=lhs)")
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="coarse", help="Grid refinement")
    parser.add_argument("--codes", type=str, default="0012,2412", help="Comma-separated NACA codes (if mode=canonical)")
    parser.add_argument("--alphas", type=str, default="0.0,4.0", help="Comma-separated alpha values (if mode=canonical)")
    parser.add_argument("--reynolds", type=float, default=6.0e6, help="Reynolds number")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    print("=" * 72)
    print("OPENFOAM AUTOMATED BATCH CFD RUNNER (PHASE II -> PHASE III)")
    print("=" * 72)
    print(f"Sampling Mode  : {args.mode.upper()}")
    print(f"Mesh Level     : {args.mesh_level.upper()}")
    print(f"Reynolds No.   : {args.reynolds:.1e}")

    # Build sample list
    if args.mode == "canonical":
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
        samples = generate_canonical_benchmark_cases(codes=codes, alphas=alphas, reynolds=args.reynolds)
    else:
        samples = sample_latin_hypercube(n_samples=args.n_samples, reynolds=args.reynolds)

    print(f"Total Cases to Run: {len(samples)}")
    print("=" * 72)

    raw_dir = PROJECT_ROOT / "data" / "raw_cases"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = processed_dir / "cfd_batch_results.csv"

    records = []
    t_start = time.perf_counter()

    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] Running {s.case_id} (NACA {s.naca_code}, alpha={s.alpha_deg:+.1f} deg)...")
        case_path = raw_dir / s.case_id
        t0 = time.perf_counter()
        res = run_single_cfd_sample(
            sample=s,
            output_case_dir=case_path,
            cfg=cfg,
            mesh_level=args.mesh_level,
            timeout_sec=300,
        )
        elapsed = time.perf_counter() - t0

        if res.get("success"):
            rec = res["record"]
            records.append(rec)
            cl = rec["cl"]
            cd = rec["cd"]
            ld = rec["lift_to_drag"]
            stat = rec["status"].upper()
            print(f"       Done in {elapsed:.1f}s | CL={cl:.4f} | CD={cd:.5f} | L/D={ld:.1f} | Status={stat}")
        else:
            print(f"       Failed in {elapsed:.1f}s | Status={res.get('status')}")

    total_time = time.perf_counter() - t_start
    print("=" * 72)
    print(f"BATCH SIMULATION COMPLETED ({len(records)}/{len(samples)} successful) in {total_time:.1f}s")
    print("=" * 72)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(summary_csv, index=False)
        print(f"Saved results to : {summary_csv}")
        print("\nSummary Table:")
        cols_to_show = ["case_id", "naca_code", "alpha_deg", "cl", "cd", "cm", "lift_to_drag", "status"]
        print(df[[c for c in cols_to_show if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
