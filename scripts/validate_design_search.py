"""Independent validation of the SLSQP design search.

Compares, for the SAME operating point:
  1. surrogate prediction of the optimization baseline geometry
  2. surrogate prediction of the surrogate-optimized geometry
  3. actual OpenFOAM result for the optimized geometry (reused from the
     converged campaign when that exact case exists; solved fresh otherwise)

This distinguishes "the optimizer runs" from "the optimized geometry actually
performs better under CFD". Outputs reports/evaluation/design_search_validation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.dataset.pipeline import run_single_cfd_sample  # noqa: E402
from src.dataset.sampler import AirfoilSample  # noqa: E402
from src.models.inference import AeroSurrogateEngine  # noqa: E402

OUT = ROOT / "reports" / "evaluation" / "design_search_validation.json"
CASES_DIR = ROOT / "data" / "design_validation"


def case_id_for(m: float, p: float, t: float, alpha: float, re: float) -> str:
    md = int(round(m * 100))
    pd_ = int(round(p * 10)) if m > 0 else 0
    td = int(round(t * 100))
    return (f"NACA{md}{pd_}{td:02d}_a{alpha:+.1f}_Re{re:.0e}"
            .replace("e+0", "e").replace("e+", "e"))


def cfd_or_campaign(case_id: str, m: float, p: float, t: float,
                    alpha: float, re: float) -> dict | None:
    """Return converged campaign CFD values if this exact case exists; else None."""
    csv = ROOT / "data/processed/airfoil_cfd_dataset.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, dtype={"case_id": str, "naca_code": str})
    latest = df.drop_duplicates("case_id", keep="last")
    hit = latest[(latest.case_id == case_id) & (latest.status == "converged")]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {"cl": float(row.cl), "cd": float(row.cd), "cm": float(row.cm),
            "source": "campaign"}


def run_cfd(case_id: str, m: float, p: float, t: float,
            alpha: float, re: float) -> dict | None:
    cfg = load_config()
    out_dir = CASES_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = AirfoilSample(case_id=case_id, naca_code=f"{m:.2f}".replace("0.", "")
                           .ljust(2, "0") + f"{p:.1f}".replace("0.", "")
                           .ljust(1, "0") + f"{t * 100:02.0f}",
                           m=m, p=p, t=t, alpha_deg=alpha, reynolds=re)
    result = run_single_cfd_sample(sample, out_dir, cfg,
                                   mesh_level="medium", timeout_sec=1800)
    rec = result.get("record") or {}
    if rec.get("status") != "converged":
        return {"status": rec.get("status", result.get("status", "unknown")),
                "source": "fresh_cfd_failed"}
    return {"cl": rec["cl"], "cd": rec["cd"], "cm": rec["cm"],
            "source": "fresh_cfd"}


def main() -> None:
    eng = AeroSurrogateEngine(models_dir=ROOT / "models")
    print("running SLSQP design search …")
    # A feasible start matters: x0 must already satisfy the lift target or
    # SLSQP aborts at the first line search (baseline 0012 @ a=4 deg gives
    # CL 0.44 < 0.50). NACA 2412 at a=4 deg sits at CL ~ 0.5.
    x0 = (0.02, 0.4, 0.12, 4.0)
    opt = eng.optimize_airfoil(x0=x0)  # L/D max, CL>=0.5, wingbox limits
    bp = opt["baseline_params"]
    op = opt.get("opt_params")
    baseline = {"m": float(bp["m"]), "p": float(bp["p"]),
                  "t": float(bp["t"]), "alpha": float(bp["alpha"])}
    optimized = {"m": float(op["m"]), "p": float(op["p"]),
                  "t": float(op["t"]), "alpha": float(op["alpha"])}
    print(f"baseline  (m,p,t,α) = {baseline}")
    print(f"optimized (m,p,t,α) = {optimized}")

    reynolds = 6.0e6
    payload = {"baseline_geometry": baseline, "optimized_geometry": optimized,
               "reynolds": reynolds, "optimizer_success": opt["success"],
               "optimizer_constraints_met": opt.get("constraints_met")}

    for tag, geom in (("baseline", baseline), ("optimized", optimized)):
        sp = eng.predict(geom["m"], geom["p"], geom["t"], geom["alpha"],
                         reynolds, compute_structures=False)
        payload[f"{tag}_surrogate"] = {
            "cl": sp["cl"], "cd": sp["cd"], "cm": sp["cm"],
            "lift_to_drag": sp["lift_to_drag"]}
        cid = case_id_for(geom["m"], geom["p"], geom["t"], geom["alpha"], reynolds)
        payload[f"{tag}_case_id"] = cid
        cfd = cfd_or_campaign(cid, geom["m"], geom["p"], geom["t"],
                              geom["alpha"], reynolds)
        if cfd is None:
            print(f"{tag}: no converged campaign case ({cid}) — running OpenFOAM …")
            cfd = run_cfd(cid, geom["m"], geom["p"], geom["t"],
                          geom["alpha"], reynolds)
        payload[f"{tag}_cfd"] = cfd

    def ld(d):
        return d["lift_to_drag"] if "lift_to_drag" in d else (
            d["cl"] / d["cd"] if d.get("cd") else None)

    payload["summary"] = {
        "baseline_surrogate_LD": payload["baseline_surrogate"]["lift_to_drag"],
        "optimized_surrogate_LD": payload["optimized_surrogate"]["lift_to_drag"],
        "surrogate_predicted_LD_gain": (
            payload["optimized_surrogate"]["lift_to_drag"]
            - payload["baseline_surrogate"]["lift_to_drag"]),
        "baseline_CFD_LD": ld(payload["baseline_cfd"]) if payload["baseline_cfd"] else None,
        "optimized_CFD_LD": ld(payload["optimized_cfd"]) if payload["optimized_cfd"] else None,
        "CFD_measured_LD_gain": (
            (ld(payload["optimized_cfd"]) - ld(payload["baseline_cfd"]))
            if payload.get("baseline_cfd") and payload.get("optimized_cfd")
            and "status" not in payload["baseline_cfd"]
            and "status" not in payload["optimized_cfd"] else None),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
