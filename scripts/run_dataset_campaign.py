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
        --alphas "-6,-2,2,6,10,14" --res "1e6,2e6,3e6,6e6" --jobs 3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the real-CFD dataset campaign.")
    p.add_argument("--airfoils", type=int, default=21, help="Number of LHS-sampled airfoils")
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14", help="Comma-separated alphas (deg)")
    p.add_argument("--res", type=str, default="1e6,2e6,3e6,6e6", help="Comma-separated Reynolds numbers")
    p.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    p.add_argument("--timeout", type=int, default=900, help="Per-case solver timeout (s)")
    p.add_argument("--jobs", type=int, default=3, help="Concurrent WSL solver instances")
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

    return list(ANCHOR_CODES) + [c for c in codes if c not in ANCHOR_CODES]


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


class Store:
    """Thread-safe incremental CSV/NPZ persistence."""

    def __init__(self, existing: pd.DataFrame, x_grid: np.ndarray | None,
                 cp_store: dict[str, dict[str, np.ndarray]]):
        self.df = existing
        self.x_grid = x_grid
        self.cp_store = cp_store
        self.lock = threading.Lock()
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict, cp: dict[str, np.ndarray] | None, x_grid: np.ndarray | None) -> None:
        with self.lock:
            self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
            self.df.to_csv(CSV_PATH, index=False)
            if cp is not None:
                self.cp_store[record["case_id"]] = cp
                if x_grid is not None and self.x_grid is None:
                    self.x_grid = x_grid
            if self.cp_store and len(self.df) % 20 == 0:
                self._save_npz()

    def save_npz(self) -> None:
        with self.lock:
            self._save_npz()

    def _save_npz(self) -> None:
        if not self.cp_store or self.x_grid is None:
            return
        ids = list(self.cp_store.keys())
        NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            NPZ_PATH,
            x_grid=self.x_grid.astype(np.float32),
            cp_upper=np.array([self.cp_store[c]["cp_upper"] for c in ids], dtype=np.float32),
            cp_lower=np.array([self.cp_store[c]["cp_lower"] for c in ids], dtype=np.float32),
            delta_cp=np.array([self.cp_store[c]["delta_cp"] for c in ids], dtype=np.float32),
            case_ids=np.array(ids),
        )


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
    print(f"Mesh level      : {args.mesh_level} | jobs: {args.jobs}")
    print(f"Total cases     : {total}")

    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()
    done_ids = set(existing["case_id"]) if len(existing) else set()

    cp_store: dict[str, dict[str, np.ndarray]] = {}
    x_grid: np.ndarray | None = None
    if NPZ_PATH.exists():
        prev = np.load(NPZ_PATH, allow_pickle=False)
        x_grid = prev["x_grid"]
        for cid, cu, cl_, cd_ in zip(prev["case_ids"], prev["cp_upper"], prev["cp_lower"], prev["delta_cp"]):
            cp_store[str(cid)] = {"cp_upper": cu, "cp_lower": cl_, "delta_cp": cd_}

    store = Store(existing, x_grid, cp_store)
    print(f"Resume: {len(done_ids)} cases already done" if done_ids else "Fresh start")

    def run_one(code: str, reynolds: float, alpha: float) -> None:
        case_id = f"NACA{code}_a{alpha:+.1f}_Re{reynolds:.0e}".replace("e+0", "e").replace("e+", "e")
        if case_id in done_ids:
            return
        sample = AirfoilSample(
            case_id=case_id, naca_code=code,
            m=int(code[0]) / 100.0, p=int(code[1]) / 10.0, t=int(code[2:4]) / 100.0,
            alpha_deg=alpha, reynolds=reynolds,
        )
        case_dir = RAW_DIR / case_id
        t0 = time.perf_counter()
        try:
            result = run_single_cfd_sample(
                sample, case_dir, cfg, mesh_level=args.mesh_level, timeout_sec=args.timeout
            )
        except Exception as exc:
            result = {"success": False, "status": f"error: {type(exc).__name__}"}
        elapsed = time.perf_counter() - t0

        cp_payload = None
        cp_data = result.get("cp_data", {})
        if result.get("success"):
            record = dict(result["record"])
            if cp_data.get("available"):
                cp_payload = {
                    "cp_upper": np.asarray(cp_data["cp_upper"], dtype=np.float32),
                    "cp_lower": np.asarray(cp_data["cp_lower"], dtype=np.float32),
                    "delta_cp": np.asarray(cp_data["delta_cp"], dtype=np.float32),
                }
        else:
            record = {
                "case_id": case_id, "naca_code": code,
                "m": sample.m, "p": sample.p, "t": sample.t,
                "alpha_deg": alpha, "reynolds": reynolds,
                "status": result.get("status", "solver_failed"), "success": False,
            }
        record["wall_time_s"] = round(elapsed, 1)
        cleanup_case_dir(case_dir)
        store.add(record, cp_payload, np.asarray(cp_data["x_grid"], dtype=np.float32) if cp_payload else None)

        with _print_lock:
            cl = record.get("cl", float("nan"))
            cd = record.get("cd", float("nan"))
            print(
                f"  {case_id:<32} {record.get('status', '?'):<12} "
                f"CL={cl:+.4f} CD={cd:.5f} ({elapsed:.0f}s)",
                flush=True,
            )

    jobs = [(c, r, a) for c in codes for r in reynolds_list for a in alphas]
    if args.limit:
        jobs = jobs[: args.limit]

    t_campaign = time.perf_counter()
    new_count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, c, r, a) for c, r, a in jobs]
        for fut in as_completed(futures):
            fut.result()
            new_count += 1

    store.save_npz()
    total_time = time.perf_counter() - t_campaign
    print("=" * 72)
    print(f"CAMPAIGN DONE: {new_count} new cases in {total_time / 3600:.2f} h")
    if len(store.df):
        print("Status distribution:")
        print(store.df["status"].value_counts().to_string())
        print(f"CSV : {CSV_PATH} ({len(store.df)} rows)")
        print(f"NPZ : {NPZ_PATH} ({len(store.cp_store)} Cp fields)")
    print("=" * 72)


if __name__ == "__main__":
    main()
