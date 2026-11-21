"""Breakdown evaluation of the shipped surrogate bundle (no aggregate-only claims).

Computes, against the converged real-CFD dataset:
  - grouped-holdout metrics (airfoils excluded from training) AND in-domain
    (training-airfoil) metrics, per target
  - per-airfoil breakdown of the held-out geometries
  - per-alpha operating-condition breakdown
  - CD and CM focused analysis: relative MAE, bias, spread by Reynolds
  - per-station Cp error profile (spatial analysis along x/c)

Outputs reports/evaluation/surrogate_breakdown.json + two figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.inference import AeroSurrogateEngine  # noqa: E402

CSV = ROOT / "data/processed/airfoil_cfd_dataset.csv"
OUT = ROOT / "reports" / "evaluation"
HOLDOUT = ["3414", "5315", "5611"]  # grouped test airfoils from the final training run

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "--", "grid.linewidth": 0.5, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
C1, C2, C3 = "#FF9838", "#3A7CC2", "#6FCF97"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = 1 - np.sum((y - p) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
    return float(ss)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "R2": r2(y, p), "RMSE": rmse(y, p), "MAE": mae(y, p),
        "bias": float(np.mean(p - y)),
        "n": int(len(y)),
    }


def main() -> None:
    eng = AeroSurrogateEngine(models_dir=ROOT / "models")
    df = pd.read_csv(CSV, dtype={"case_id": str, "naca_code": str})
    latest = df.drop_duplicates("case_id", keep="last")
    conv = latest[latest.status == "converged"].copy()
    conv["naca_code"] = conv.naca_code.str.zfill(4)
    conv = conv.sort_values(["naca_code", "reynolds", "alpha_deg"]).reset_index(drop=True)

    x_grid = np.linspace(0, 1, 161)
    npz = np.load(ROOT / "data/processed/cp_surface_distributions.npz")
    npz_ids = [str(c) for c in npz["case_ids"]]
    meas_cp = {cid: (np.asarray(npz["cp_upper"][i], float),
                     np.asarray(npz["cp_lower"][i], float))
               for i, cid in enumerate(npz_ids)}

    records = []
    for _, row in conv.iterrows():
        res = eng.predict(m=row.m, p=row.p, t=row.t,
                          alpha_deg=row.alpha_deg, reynolds=row.reynolds)
        records.append({
            "case_id": row.case_id, "m": row.m, "p": row.p, "t": row.t,
            "naca_code": row.naca_code, "alpha_deg": row.alpha_deg,
            "reynolds": row.reynolds,
            "cl_true": row.cl, "cd_true": row.cd, "cm_true": row.cm,
            "cl_pred": res["cl"], "cd_pred": res["cd"], "cm_pred": res["cm"],
            "cp_upper_pred": res["cp_upper"], "cp_lower_pred": res["cp_lower"],
            "is_holdout": row.naca_code in HOLDOUT,
        })
    ev = pd.DataFrame(records)
    print(f"evaluated {len(ev)} converged rows")

    targets = {"cl": "cl", "cd": "cd", "cm": "cm"}
    breakdown: dict = {}

    # in-domain vs grouped-holdout, per target
    for tag, mask in (("in_domain", ~ev.is_holdout), ("grouped_holdout", ev.is_holdout)):
        sub = ev[mask]
        breakdown[tag] = {
            t: metrics(sub[f"{k}_true"], sub[f"{k}_pred"]) for t, k in targets.items()
        }

    # per-airfoil (holdout geometries)
    per_airfoil = {}
    for code, g in ev[ev.is_holdout].groupby("naca_code"):
        per_airfoil[code] = {
            t: metrics(g[f"{k}_true"], g[f"{k}_pred"]) for t, k in targets.items()
        }
    breakdown["holdout_per_airfoil"] = per_airfoil

    # per-alpha bins (holdout + in-domain separately)
    per_alpha = {}
    for tag in ("grouped_holdout", "in_domain"):
        bins = {}
        sub = ev[ev.is_holdout] if tag == "grouped_holdout" else ev[~ev.is_holdout]
        for a, g in sub.groupby("alpha_deg"):
            bins[str(a)] = {t: metrics(g[f"{k}_true"], g[f"{k}_pred"])
                            for t, k in targets.items()}
        per_alpha[tag] = bins
    breakdown["per_alpha"] = per_alpha

    # CD focused: relative MAE + bias by Reynolds (holdout)
    cd_focus = {}
    hold = ev[ev.is_holdout]
    mean_cd_true = float(hold.cd_true.mean())
    cd_focus["holdout_mean_cd_true"] = mean_cd_true
    cd_focus["holdout_relative_MAE"] = float(
        np.mean(np.abs(hold.cd_pred - hold.cd_true)) / mean_cd_true)
    cd_focus["holdout_bias"] = float(np.mean(hold.cd_pred - hold.cd_true))
    by_re = {}
    for re, g in hold.groupby("reynolds"):
        by_re[str(re)] = metrics(g.cd_true, g.cd_pred) | {
            "relative_MAE": float(np.mean(np.abs(g.cd_pred - g.cd_true))
                                  / float(g.cd_true.mean()))}
    cd_focus["by_reynolds"] = by_re
    breakdown["cd_focus"] = cd_focus

    # per-station Cp error profile (holdout, upper + lower)
    prof = {"x": x_grid.tolist(),
            "upper_mean_abs_err": [], "lower_mean_abs_err": []}
    hold_ids = [c for c in ev.loc[ev.is_holdout, "case_id"] if c in meas_cp]
    upper_errs, lower_errs = [], []
    for cid in hold_ids:
        row = ev[ev.case_id == cid].iloc[0]
        res = eng.predict(m=row.m, p=row.p, t=row.t,
                          alpha_deg=row.alpha_deg, reynolds=row.reynolds)
        mu, ml = meas_cp[cid]
        upper_errs.append(np.abs(np.asarray(res["cp_upper"]) - mu))
        lower_errs.append(np.abs(np.asarray(res["cp_lower"]) - ml))
    prof["upper_mean_abs_err"] = np.mean(np.stack(upper_errs), axis=0).tolist()
    prof["lower_mean_abs_err"] = np.mean(np.stack(lower_errs), axis=0).tolist()
    breakdown["cp_station_profile"] = {
        "n_holdout_cases": len(upper_errs),
        "upper_max_station_err": float(np.max(prof["upper_mean_abs_err"])),
        "lower_max_station_err": float(np.max(prof["lower_mean_abs_err"])),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "surrogate_breakdown.json").write_text(
        json.dumps(breakdown, indent=2), encoding="utf-8")

    # ---- figures ----
    # 1. per-airfoil R2 bars (holdout)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True)
    codes = list(per_airfoil.keys())
    for ax, t, ttl in zip(axes, ("cl", "cd", "cm"),
                          ("C$_L$", "C$_D$", "C$_m$")):
        vals = [per_airfoil[c][t]["R2"] for c in codes]
        ax.bar(range(len(codes)), vals, width=0.55, color=C1, alpha=0.9)
        ax.axhline(0.95, color="#3A3D42", lw=0.8, ls="--")
        ax.set_xticks(range(len(codes)), codes, fontsize=8)
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("R² (holdout airfoil)")
    fig.suptitle("Per-airfoil extrapolation accuracy — grouped holdout",
                 x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "per_airfoil_r2.png")
    plt.close(fig)

    # 2. per-alpha MAE (holdout)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True)
    alphas = sorted(per_alpha["grouped_holdout"], key=float)
    for ax, t, ttl in zip(axes, ("cl", "cd", "cm"),
                          ("C$_L$ MAE", "C$_D$ MAE", "C$_m$ MAE")):
        vals = [per_alpha["grouped_holdout"][a][t]["MAE"] for a in alphas]
        ax.plot(range(len(alphas)), vals, marker="o", ms=4, color=C2)
        ax.set_xticks(range(len(alphas)), alphas, fontsize=8)
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("MAE (holdout rows)")
    fig.suptitle("Error by angle of attack — grouped holdout",
                 x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "per_alpha_mae.png")
    plt.close(fig)

    # 3. per-station Cp error profile
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(x_grid, prof["upper_mean_abs_err"], color=C1, lw=1.4, label="upper surface")
    ax.plot(x_grid, prof["lower_mean_abs_err"], color=C3, lw=1.4, label="lower surface")
    ax.set_xlabel("x/c"); ax.set_ylabel("mean |ΔCp| (holdout airfoils)")
    ax.set_title("Spatial Cp error profile", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cp_station_error.png")
    plt.close(fig)

    ev.to_csv(OUT / "per_case_predictions.csv", index=False)
    print("saved:", OUT / "surrogate_breakdown.json")
    print("figures:", FIG)


if __name__ == "__main__":
    main()
