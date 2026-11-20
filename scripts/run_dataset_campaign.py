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
import json
import shutil
import sys
import threading
import time
from collections.abc import Iterable
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
from src.dataset.cst_sampler import sample_general_cst_airfoils
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
    p.add_argument("--sampler", choices=["naca", "cst"], default="naca",
                   help="naca: LHS over NACA 4-digit codes; cst: general-CST manifold")
    p.add_argument("--airfoils", type=int, default=21, help="Number of LHS-sampled airfoils")
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14", help="Comma-separated alphas (deg)")
    p.add_argument("--res", type=str, default="1e6,2e6,3e6,6e6", help="Comma-separated Reynolds numbers")
    p.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    p.add_argument("--timeout", type=int, default=900, help="Per-case solver timeout (s)")
    p.add_argument("--jobs", type=int, default=3, help="Concurrent WSL solver instances")
    p.add_argument("--limit", type=int, default=0, help="Stop after N new cases (0 = no limit)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iterations", type=int, default=0,
                   help="Override solver max_iterations (0 = config value 3000); "
                        "use 6000 to recover unconverged cases")
    p.add_argument("--manifest", type=str, default="",
                   help="Path to an active-learning manifest JSON (overrides --sampler)")
    return p.parse_args()


def sample_airfoil_codes(n_airfoils: int, seed: int) -> list[str]:
    """Return exactly ``n_airfoils`` unique NACA codes, anchors first."""
    if isinstance(n_airfoils, (bool, np.bool_)) or not isinstance(
        n_airfoils, (int, np.integer)
    ) or n_airfoils <= 0:
        raise ValueError("n_airfoils must be a positive integer")

    target = int(n_airfoils)
    codes = list(ANCHOR_CODES[:target])
    if len(codes) == target:
        return codes

    sampler = qmc.LatinHypercube(d=3, seed=seed)
    attempts = max(target * 20, 100)
    unit = sampler.random(attempts)
    scaled = qmc.scale(unit, [0.0, 0.2, 0.06], [0.06, 0.60, 0.18])

    for m_raw, p_raw, t_raw in scaled:
        code = naca_code_from_params(float(m_raw), float(p_raw), float(t_raw))
        if code not in codes:
            codes.append(code)
        if len(codes) == target:
            return codes

    raise RuntimeError(
        f"Could not generate {target} unique NACA codes after {attempts} samples."
    )


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


def _parse_bool(value: object, *, default: bool = False) -> bool:
    """Parse CSV booleans without treating non-empty strings as true."""
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return default
        return numeric != 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0", ""}:
        return False
    return default


def completed_case_ids(existing: pd.DataFrame, cp_case_ids: Iterable[str]) -> set[str]:
    """Return successful, durable cases represented by the current CSV/NPZ pair."""
    if existing.empty or "case_id" not in existing.columns:
        return set()

    cp_ids = {str(case_id) for case_id in cp_case_ids}
    # Older campaign versions appended retries. The last row is the latest
    # attempt and therefore the authoritative success/failure state.
    rows = existing.loc[existing["case_id"].notna()].copy()
    rows["_case_id_key"] = rows["case_id"].astype(str)
    rows = rows.drop_duplicates("_case_id_key", keep="last")
    done: set[str] = set()
    for _, row in rows.iterrows():
        success = _parse_bool(row.get("success"), default="success" not in existing.columns)
        if not success:
            continue
        cp_required = _parse_bool(row.get("cp_available"), default=False)
        case_id = str(row["case_id"])
        if not cp_required or case_id in cp_ids:
            done.add(case_id)
    return done


def pending_jobs(jobs: Iterable[AirfoilSample], done_ids: set[str], limit: int = 0) -> list[AirfoilSample]:
    """Filter completed jobs before applying the new-case campaign limit."""
    pending = [sample for sample in jobs if sample.case_id not in done_ids]
    return pending[:limit] if limit else pending


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
            case_id = str(record["case_id"])
            record = dict(record)
            record["case_id"] = case_id
            cp_available = _parse_bool(record.get("cp_available"), default=False)
            if cp_available != (cp is not None):
                raise ValueError(
                    "cp_available must match the supplied Cp payload for "
                    f"case {case_id}."
                )
            if cp is not None:
                if x_grid is None:
                    raise ValueError("x_grid is required when persisting Cp data")
                incoming_grid = np.asarray(x_grid)
                if self.x_grid is not None and not np.array_equal(self.x_grid, incoming_grid):
                    raise ValueError("x_grid does not match the persisted Cp grid")
            else:
                incoming_grid = None
            previous_df = self.df
            had_previous_cp = case_id in self.cp_store
            previous_cp = self.cp_store.get(case_id)
            previous_x_grid = self.x_grid
            replacement = pd.DataFrame([record])
            if not self.df.empty and "case_id" in self.df.columns:
                matching = self.df["case_id"].astype(str).eq(case_id)
                if matching.any():
                    # Keep the latest attempt at the first matching position and remove
                    # any historical retry duplicates left by older campaign versions.
                    first_position = int(np.flatnonzero(matching.to_numpy())[0])
                    remaining = self.df.loc[~matching]
                    insert_at = int((~matching.iloc[:first_position]).sum())
                    before = remaining.iloc[:insert_at]
                    after = remaining.iloc[insert_at:]
                    self.df = pd.concat([before, replacement, after], ignore_index=True)
                else:
                    self.df = pd.concat([self.df, replacement], ignore_index=True)
            else:
                self.df = replacement

            if cp is not None:
                self.cp_store[case_id] = cp
                if self.x_grid is None:
                    assert incoming_grid is not None
                    self.x_grid = incoming_grid
            elif had_previous_cp:
                self.cp_store.pop(case_id)

            if cp is not None or had_previous_cp:
                # Persist Cp first: an NPZ-extra/CSV-missing crash state is recoverable.
                try:
                    self._save_npz()
                except Exception:
                    if had_previous_cp:
                        self.cp_store[case_id] = previous_cp
                    else:
                        self.cp_store.pop(case_id, None)
                    self.x_grid = previous_x_grid
                    self.df = previous_df
                    raise

            csv_tmp = CSV_PATH.parent / (CSV_PATH.name + ".tmp")
            try:
                self.df.to_csv(csv_tmp, index=False)
                csv_tmp.replace(CSV_PATH)
            except Exception:
                csv_tmp.unlink(missing_ok=True)
                self.df = previous_df
                # The NPZ was already committed with the new Cp state; restore the
                # previous Cp state so disk stays consistent with the surviving CSV.
                if had_previous_cp:
                    self.cp_store[case_id] = previous_cp
                else:
                    self.cp_store.pop(case_id, None)
                self.x_grid = previous_x_grid
                self._rollback_npz_after_csv_failure()
                raise

    def _rollback_npz_after_csv_failure(self) -> None:
        """Best-effort NPZ restoration after the CSV persistence step failed."""
        try:
            if self.cp_store or self.x_grid is not None:
                self._save_npz()
            else:
                # No Cp ever persisted: the correct durable state is no NPZ at all.
                NPZ_PATH.unlink(missing_ok=True)
        except Exception:
            # The next successful add() rewrites the NPZ anyway.
            pass

    def save_npz(self) -> None:
        with self.lock:
            self._save_npz()

    def _save_npz(self) -> None:
        if self.x_grid is None:
            return
        ids = list(self.cp_store.keys())
        if ids:
            cp_upper = np.array(
                [self.cp_store[c]["cp_upper"] for c in ids], dtype=np.float32
            )
            cp_lower = np.array(
                [self.cp_store[c]["cp_lower"] for c in ids], dtype=np.float32
            )
            delta_cp = np.array(
                [self.cp_store[c]["delta_cp"] for c in ids], dtype=np.float32
            )
        else:
            empty_shape = (0, len(self.x_grid))
            cp_upper = np.empty(empty_shape, dtype=np.float32)
            cp_lower = np.empty(empty_shape, dtype=np.float32)
            delta_cp = np.empty(empty_shape, dtype=np.float32)
        NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        npz_tmp = NPZ_PATH.parent / (NPZ_PATH.name + ".tmp.npz")
        try:
            np.savez_compressed(
                npz_tmp,
                x_grid=self.x_grid.astype(np.float32),
                cp_upper=cp_upper,
                cp_lower=cp_lower,
                delta_cp=delta_cp,
                case_ids=np.array(ids),
            )
            npz_tmp.replace(NPZ_PATH)
        except Exception:
            npz_tmp.unlink(missing_ok=True)
            raise


def main() -> None:
    args = parse_args()
    cfg = load_config()
    # getattr: persistence tests build a minimal Namespace without this field.
    max_iterations = getattr(args, "max_iterations", 0)
    if max_iterations:
        # In-memory override only; the case builder reads this config key and
        # every persisted record stores the budget actually used.
        cfg["solver"]["max_iterations"] = int(max_iterations)

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    reynolds_list = [float(r.strip()) for r in args.res.split(",") if r.strip()]

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest_cases = manifest["cases"]
        prebuilt = [
            AirfoilSample(
                case_id=c["case_id"], naca_code=c["shape_id"],
                m=c["m"], p=c["p"], t=c["t"],
                alpha_deg=c["alpha_deg"], reynolds=c["reynolds"],
                cst_upper=tuple(c["cst_upper"]) if c.get("cst_upper") else None,
                cst_lower=tuple(c["cst_lower"]) if c.get("cst_lower") else None,
            )
            for c in manifest_cases
        ]
        codes = [s.naca_code for s in prebuilt]
        total = len(prebuilt)
        print("=" * 72)
        print(f"REAL-CFD CAMPAIGN (active-learning manifest, round {manifest.get('round', '?')}, {total} cases)")
        print("=" * 72)
        print(f"Shape ids       : {', '.join(sorted(set(codes)))}")
        cst_shapes = []
    elif args.sampler == "cst":
        cst_shapes = sample_general_cst_airfoils(args.airfoils, seed=args.seed)
        shape_ids = [s.sample_id for s in cst_shapes]
        total = len(shape_ids) * len(alphas) * len(reynolds_list)
        print("=" * 72)
        print(f"REAL-CFD DATASET CAMPAIGN (sampler: general CST, {len(shape_ids)} shapes)")
        print("=" * 72)
        print(f"Shape ids       : {', '.join(shape_ids)}")
        codes = shape_ids  # used only for the progress/summary prints below
    else:
        cst_shapes = []
        codes = sample_airfoil_codes(args.airfoils, args.seed)
        total = len(codes) * len(alphas) * len(reynolds_list)
        print("=" * 72)
        print(f"REAL-CFD DATASET CAMPAIGN (sampler: NACA LHS, {len(codes)} airfoils)")
        print("=" * 72)
        print(f"Airfoils ({len(codes)}) : {', '.join(codes)}")
    print(f"Alphas          : {alphas}")
    print(f"Reynolds        : {reynolds_list}")
    print(f"Mesh level      : {args.mesh_level} | jobs: {args.jobs}")
    print(f"Total cases     : {total}")

    # String dtypes preserve leading zeros in NACA codes ("0012") across resumes.
    existing = (
        pd.read_csv(CSV_PATH, dtype={"case_id": str, "naca_code": str, "airfoil_code": str})
        if CSV_PATH.exists() else pd.DataFrame()
    )

    cp_store: dict[str, dict[str, np.ndarray]] = {}
    x_grid: np.ndarray | None = None
    if NPZ_PATH.exists():
        with np.load(NPZ_PATH, allow_pickle=False) as prev:
            x_grid = prev["x_grid"]
            for cid, cu, cl_, cd_ in zip(prev["case_ids"], prev["cp_upper"], prev["cp_lower"], prev["delta_cp"]):
                cp_store[str(cid)] = {"cp_upper": cu, "cp_lower": cl_, "delta_cp": cd_}

    done_ids = completed_case_ids(existing, cp_store)
    store = Store(existing, x_grid, cp_store)
    print(f"Resume: {len(done_ids)} cases already done" if done_ids else "Fresh start")

    def build_sample(shape_id: str, reynolds: float, alpha: float) -> AirfoilSample:
        """Create the campaign AirfoilSample for either sampler mode."""
        if args.sampler == "cst":
            cst_sample = next(s for s in cst_shapes if s.sample_id == shape_id)
            case_id = f"{shape_id}_a{alpha:+.1f}_Re{reynolds:.0e}".replace("e+0", "e").replace("e+", "e")
            return AirfoilSample(
                case_id=case_id, naca_code=shape_id,
                m=cst_sample.measured_camber,
                p=cst_sample.geometry.p,
                t=cst_sample.measured_thickness,
                alpha_deg=alpha, reynolds=reynolds,
                cst_upper=cst_sample.cst_upper, cst_lower=cst_sample.cst_lower,
            )
        code = shape_id
        case_id = f"NACA{code}_a{alpha:+.1f}_Re{reynolds:.0e}".replace("e+0", "e").replace("e+", "e")
        return AirfoilSample(
            case_id=case_id, naca_code=code,
            m=int(code[0]) / 100.0, p=int(code[1]) / 10.0, t=int(code[2:4]) / 100.0,
            alpha_deg=alpha, reynolds=reynolds,
        )

    def run_one(sample: AirfoilSample) -> bool:
        case_id = sample.case_id
        if case_id in done_ids:
            return False
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
                record["cp_available"] = True
            else:
                record.setdefault("cp_available", False)
        else:
            record = {
                "case_id": case_id, "naca_code": sample.naca_code,
                "m": sample.m, "p": sample.p, "t": sample.t,
                "alpha_deg": sample.alpha_deg, "reynolds": sample.reynolds,
                "status": result.get("status", "solver_failed"), "success": False,
                "cp_available": False,
            }
        record.setdefault("data_source", "openfoam")
        record.setdefault("is_synthetic", False)
        record.setdefault("solver_backend", "simpleFoam")
        record.setdefault("mesh_level", args.mesh_level)
        record.setdefault("mesh_split_target",
                          result.get("mesh_split_target", 0.30))
        record.setdefault("max_iterations", int(cfg["solver"]["max_iterations"]))
        record.setdefault("case_id", case_id)
        record.setdefault("success", True)
        record["wall_time_s"] = round(elapsed, 1)
        store.add(record, cp_payload, np.asarray(cp_data["x_grid"], dtype=np.float32) if cp_payload else None)
        # Store.add returns only after all state required by this row is durable.
        cleanup_case_dir(case_dir)

        with _print_lock:
            cl = record.get("cl", float("nan"))
            cd = record.get("cd", float("nan"))
            print(
                f"  {case_id:<36} {record.get('status', '?'):<12} "
                f"CL={cl:+.4f} CD={cd:.5f} ({elapsed:.0f}s)",
                flush=True,
            )
        return True

    if args.manifest:
        jobs = prebuilt
    else:
        jobs = [build_sample(s, r, a) for s in codes for r in reynolds_list for a in alphas]
    jobs = pending_jobs(jobs, done_ids, args.limit)

    t_campaign = time.perf_counter()
    new_count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, s) for s in jobs]
        for fut in as_completed(futures):
            if fut.result():
                new_count += 1

    store.save_npz()
    total_time = time.perf_counter() - t_campaign
    print("=" * 72)
    print(f"CAMPAIGN DONE: {new_count} new cases in {total_time / 3600:.2f} h")
    if len(store.df):
        print("Status distribution:")
        if "status" in store.df.columns:
            print(store.df["status"].value_counts().to_string())
        else:
            print("unknown")
        print(f"CSV : {CSV_PATH} ({len(store.df)} rows)")
        print(f"NPZ : {NPZ_PATH} ({len(store.cp_store)} Cp fields)")
    print("=" * 72)


if __name__ == "__main__":
    main()
