"""Remove rows from the campaign checkpoint so the next run retries them.

The campaign resume logic treats success=True rows (including solved-but-
unconverged cases with Cp data) as durable and skips them. To retry a class
of cases with a larger solver budget, first prune those rows from BOTH the
CSV and the Cp NPZ so checkpoint state stays consistent.

Examples:
  python scripts/prune_checkpoint_rows.py --status unconverged --dry-run
  python scripts/prune_checkpoint_rows.py --status unconverged,mesh_failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dataset_campaign import CSV_PATH, NPZ_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune campaign checkpoint rows for retry.")
    p.add_argument("--status", type=str, default="unconverged",
                   help="Comma-separated status values to remove (e.g. unconverged)")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--npz", type=Path, default=NPZ_PATH)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be removed without writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = {s.strip() for s in args.status.split(",") if s.strip()}
    if not targets:
        raise SystemExit("No status values given.")

    df = pd.read_csv(args.csv, dtype={"case_id": str, "naca_code": str, "airfoil_code": str})
    drop_mask = df["status"].isin(targets)
    dropped_ids = df.loc[drop_mask, "case_id"].tolist()
    print(f"CSV rows: {len(df)} | dropping {int(drop_mask.sum())} rows "
          f"with status in {sorted(targets)}")
    if dropped_ids:
        print(f"  e.g. {dropped_ids[:5]}")
    if args.dry_run or not dropped_ids:
        print("Dry run / nothing to drop — no files written.")
        return

    kept = df.loc[~drop_mask].copy()
    csv_tmp = args.csv.parent / (args.csv.name + ".tmp")
    kept.to_csv(csv_tmp, index=False)
    csv_tmp.replace(args.csv)
    print(f"CSV rewritten: {len(kept)} rows remain.")

    if not args.npz.exists():
        print("No NPZ present — nothing else to do.")
        return
    npz = np.load(args.npz, allow_pickle=False)
    case_ids = [str(c) for c in npz["case_ids"]]
    keep_idx = [i for i, cid in enumerate(case_ids) if cid not in set(dropped_ids)]
    removed_fields = len(case_ids) - len(keep_idx)
    if removed_fields == 0:
        print("NPZ has no fields for the dropped rows — left untouched.")
        return
    npz_tmp = args.npz.parent / (args.npz.name + ".tmp.npz")
    np.savez_compressed(
        npz_tmp,
        x_grid=npz["x_grid"].astype(np.float32),
        case_ids=np.array([case_ids[i] for i in keep_idx]),
        cp_upper=np.asarray(npz["cp_upper"])[keep_idx].astype(np.float32),
        cp_lower=np.asarray(npz["cp_lower"])[keep_idx].astype(np.float32),
        delta_cp=np.asarray(npz["delta_cp"])[keep_idx].astype(np.float32),
    )
    npz_tmp.replace(args.npz)
    print(f"NPZ rewritten: removed {removed_fields} Cp fields, "
          f"{len(keep_idx)} remain.")


if __name__ == "__main__":
    main()
