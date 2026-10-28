#!/usr/bin/env python3
"""
Real-CFD Dataset Campaign Driver.

Runs the OpenFOAM pipeline across a Latin-Hypercube sample of NACA 4-digit
airfoils, multiple Reynolds numbers, and a fixed alpha grid, and compiles an
honest dataset: every row keeps its true convergence status, and Cp fields are
stored only for cases where extraction succeeded.

Resume-safe: completed case_ids are skipped on re-run, and the CSV/NPZ are
rewritten incrementally. Heavy solver artifacts (time dirs, 0/, constant/,
system/) are deleted after each case; logs and postProcessing are kept.

Example:
    python scripts/run_dataset_campaign.py --airfoils 21 --mesh medium \
        --alphas "-6,-2,2,6,10,14" --res "1e6,2e6,3e6,6e6"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.config import load_config
from src.dataset.pipeline import run_single_cfd_sample
from src.dataset.sampler import AirfoilSample, naca_code_from_params

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "airfoil_cfd_dataset.csv"
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "cp_surface_distributions.npz"
RAW_DIR = PROJECT_ROOT / "data" / "raw_cases"

# Anchor airfoils always included (benchmark / literature cases)
ANCHOR_CODES = ("0012", "2412", "4412")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the real-CFD dataset campaign.")
    p.add_argument("--airfoils", type=int, default=21, help="Number of LHS-sampled airfoils")
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14", help="Comma-separated alphas (deg)")
    p.add_argument("--res", type=str, default="1e6,2e6,3e6,6e6", help="Comma-separated Reynolds numbers")
    p.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    p.add_argument("--timeout", type=int, default=600, help="Per-case solver timeout (s)")
    p.add_argument("--limit", type=int, default=0, help="Stop after N new cases (0 = no limit)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_airfoil_codes(n_airfoils: int, seed: int) -> list[str]:
    """LHS over (m, p, t), discretized to valid NACA 4-digit codes, deduplicated."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n_airfoils * 3)  # oversample: dedupe + validity culling
    scaled = qmc.scale(unit, [0.0, 0.2, 0.06], [0.06, 0.60, 0.18])

    codes: list[str] = []
    for m_raw, p_raw, t_raw in scaled:
        code = naca_code_from_params(float(m_raw), float(p_raw), float(t_raw))
        if code not in codes:
            codes.append(code)
        if len(codes) >= n_airfoils - len(ANCHOR_CODES):
            break

    all_codes = list(ANCHOR_CODES) + [c for c in codes if c not in ANCHOR_CODES]
    return all_codes


def cleanup_case_dir(case_dir: Path) -> None:
    """Keep logs, postProcessing and geometry metadata; delete heavy solver artifacts."""
    keep = {"postProcessing", "airfoil.dat", "airfoil_meta.json"}
    for item in case_dir.iterdir():
        if item.name in keep or item.name.startswith("log."):
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    cfg = load_config()

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    reynolds_list = [float(r.strip()) for r in args.res.split(",") if r.strip()]
    codes = sample_airfoil_codes(args.airfoils, args.seed)

    total = len(codes) * len(alphas) * len(reynolds_list)
    print("=" * 72)
    print("REAL-CFD DATASET CAMPAIGN")
    print("=" * 72)
    print(f"Airfoils ({len(codes)}) : {', '.join(codes)}")
    print(f"Alphas          : {alphas}")
    print(f"Reynolds        : {reynolds_list}")
    print(f"Mesh level      : {args.mesh_level}")
    print(f"Total cases     : {total}")

    # Resume state
    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()
    done_ids = set(existing["case_id"]) if len(existing) else set()
    print(f"Already done    : {len(done_ids)} (resume mode)" if done_ids else "Fresh start")

    # Cp store (resumed from existing NPZ if present)
    cp_store: dict[str, dict[str, np.ndarray]] = {}
    x_grid: np.ndarray | None = None
    if NPZ_PATH.exists():
        prev = np.load(NPZ_PATH, allow_pickle=False)
        x_grid = prev["x_grid"]
        for cid, cu, cl_, cd_ in zip(
            prev["case_ids"], prev["cp_upper"], prev["cp_lower"], prev["delta_cp"]
        ):
            cp_store[str(cid)] = {"cp_upper": cu, "cp_lower": cl_, "delta_cp": cd_}

    def save_csv(df_all: pd.DataFrame) -> None:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(CSV_PATH, index=False)

    def save_npz() -> None:
        if not cp_store or x_grid is None:
            return
        ids = list(cp_store.keys())
        NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            NPZ_PATH,
            x_grid=x_grid.astype(np.float32),
            cp_upper=np.array([cp_store[c]["cp_upper"] for c in ids], dtype=np.float32),
            cp_lower=np.array([cp_store[c]["cp_lower"] for c in ids], dtype=np.float32),
            delta_cp=np.array([cp_store[c]["delta_cp"] for c in ids], dtype=np.float32),
            case_ids=np.array(ids),
        )

    records: list[dict] = []
    new_count = 0
    t_campaign = time.perf_counter()

    for code in codes:
        for reynolds in reynolds_list:
            for alpha in alphas:
                case_id = f"NACA{code}_a{alpha:+.1f}_Re{reynolds:.0e}".replace("e+0", "e").replace("e+", "e")

                if case_id in done_ids:
                    continue
                if args.limit and new_count >= args.limit:
                    break

                sample = AirfoilSample(
                    case_id=case_id,
                    naca_code=code,
                    m=int(code[0]) / 100.0,
                    p=int(code[1]) / 10.0,
                    t=int(code[2:4]) / 100.0,
                    alpha_deg=alpha,
                    reynolds=reynolds,
                )
                case_dir = RAW_DIR / case_id
                t0 = time.perf_counter()
                print(f"[{new_count + 1}] {case_id} ...", flush=True)
                try:
                    result = run_single_cfd_sample(
                        sample, case_dir, cfg, mesh_level=args.mesh_level, timeout_sec=args.timeout
                    )
                except Exception as exc:
                    result = {"success": False, "status": f"error: {type(exc).__name__}"}

                elapsed = time.perf_counter() - t0

                if result.get("success"):
                    rec = dict(result["record"])
                    if result.get("cp_data", {}).get("available"):
                        cp_store[case_id] = {
                            "cp_upper": np.asarray(result["cp_data"]["cp_upper"], dtype=np.float32),
                            "cp_lower": np.asarray(result["cp_data"]["cp_lower"], dtype=np.float32),
                            "delta_cp": np.asarray(result["cp_data"]["delta_cp"], dtype=np.float32),
                        }
                        if x_grid is None:
                            x_grid = np.asarray(result["cp_data"]["x_grid"], dtype=np.float32)
                else:
                    rec = {
                        "case_id": case_id,
                        "naca_code": code,
                        "m": sample.m,
                        "p": sample.p,
                        "t": sample.t,
                        "alpha_deg": alpha,
                        "reynolds": reynolds,
                        "status": result.get("status", "solver_failed"),
                        "success": False,
                    }
                rec["wall_time_s"] = round(elapsed, 1)
                records.append(rec)
                new_count += 1

                df_all = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
                save_csv(df_all)
                if new_count % 20 == 0:
                    save_npz()
                cleanup_case_dir(case_dir)
                print(
                    f"    -> {rec.get('status', '?'):<12} CL={rec.get('cl', float('nan')):+.4f} "
                    f"CD={rec.get('cd', float('nan')):.5f} ({elapsed:.0f}s)",
                    flush=True,
                )
            if args.limit and new_count >= args.limit:
                break
        if args.limit and new_count >= args.limit:
            break

    save_npz()
    df_all = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if records else existing
    if len(df_all):
        save_csv(df_all)

    total_time = time.perf_counter() - t_campaign
    print("=" * 72)
    print(f"CAMPAIGN DONE: {new_count} new cases in {total_time / 3600:.2f} h")
    if len(df_all):
        vc = df_all["status"].value_counts()
        print("Status distribution:")
        print(vc.to_string())
        print(f"CSV : {CSV_PATH} ({len(df_all)} rows)")
        print(f"NPZ : {NPZ_PATH} ({len(cp_store)} Cp fields)")
    print("=" * 72)


if __name__ == "__main__":
    main()
