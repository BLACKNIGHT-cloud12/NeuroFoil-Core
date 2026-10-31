#!/usr/bin/env python3
"""
Select the next active-learning CFD batch.

Prerequisites: models/ contains trained surrogate weights + conformal meta
(i.e., the Kaggle notebook has been run on the data collected so far).

Workflow per round:
  1. python scripts/train round done on Kaggle  (models/ updated)
  2. python scripts/select_active_batch.py --batch-size 150
  3. python scripts/run_dataset_campaign.py --manifest data/active_round<k>.json

The manifest contains only case_ids not already present in the dataset CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.active_learning import (
    ActiveLearningSelector,
    build_candidate_pool,
    write_manifest,
)

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "airfoil_cfd_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick the next active-learning CFD batch.")
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--pool-shapes", type=int, default=60, help="Candidate shapes in the pool")
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14")
    p.add_argument("--res", type=str, default="1e6,2e6,3e6,6e6")
    p.add_argument("--sampler", choices=["cst", "naca"], default="cst")
    p.add_argument("--round", type=int, default=2, help="Round index (for the manifest name)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default="",
                   help="Manifest output path (default data/active_round<k>.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    res = [float(r.strip()) for r in args.res.split(",") if r.strip()]

    print("Loading trained surrogate for acquisition scoring ...")
    selector = ActiveLearningSelector(models_dir=MODELS_DIR, dataset_csv=CSV_PATH)

    pool = build_candidate_pool(
        CSV_PATH, n_shapes=args.pool_shapes, alphas=alphas, reynolds=res,
        sampler=args.sampler, seed=args.seed,
    )
    print(f"Candidate pool: {len(pool)} unqueried cases "
          f"({args.pool_shapes} shapes x {len(alphas)} alphas x {len(res)} Re)")

    batch = selector.select_batch(pool, batch_size=args.batch_size)
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / f"active_round{args.round}.json"
    write_manifest(batch, out, round_index=args.round)

    scores = selector.score_candidates(batch)
    print(f"\nSelected {len(batch)} cases -> {out}")
    print("Top scores in batch:", [round(s, 2) for s in sorted(scores, reverse=True)[:5]])
    print("\nNext step:")
    print(f"  python scripts/run_dataset_campaign.py --manifest {out.name} --jobs 5")


if __name__ == "__main__":
    main()
