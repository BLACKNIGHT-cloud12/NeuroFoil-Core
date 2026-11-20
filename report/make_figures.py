"""Generate publication figures for the NeuroFoil technical report.

All data are measured artifacts: campaign snapshot CSV/NPZ, mesh-study JSON,
and the shipped real-CFD model bundle. Style: serif, low-saturation, no chart
junk (top/right spines removed, light dashed grids), 300 dpi PNG.
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

ROOT = Path("/home/soumik/Documents/NeuroFoil-Core")
SNAP = sorted((ROOT / "data/snapshots").glob("full_*"))[-1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#1f5f8b", "#c44e52", "#55a868", "#8172b2"  # muted blue/red/green/purple

df = pd.read_csv(SNAP / "airfoil_cfd_dataset.csv",
                 dtype={"case_id": str, "naca_code": str})
latest = df.drop_duplicates("case_id", keep="last")
latest["naca_code"] = latest["naca_code"].str.zfill(4)
conv = latest[latest.status == "converged"].copy()
unconv = latest[latest.status == "unconverged"].copy()
ALPHAS = [-6.0, -2.0, 2.0, 6.0, 10.0, 14.0]


def fig_coverage() -> None:
    tab = pd.crosstab(conv.naca_code, conv.alpha_deg).reindex(columns=ALPHAS, fill_value=0)
    tab = tab.sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(range(len(ALPHAS)), [f"{a:+.0f}°" for a in ALPHAS])
    ax.set_yticks(range(len(tab)), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v >= 3 else "#333333")
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("NACA airfoil")
    ax.set_title("Converged cases per (airfoil × α) cell", loc="left", fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("converged cases (of 4 Re)", fontsize=8)
    fig.savefig(FIG / "fig_coverage.png")
    plt.close(fig)


def fig_cl_alpha() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    shades = plt.cm.Blues(np.linspace(0.35, 0.85, len(conv.naca_code.unique())))
    for color, (code, g) in zip(shades, conv.groupby("naca_code")):
        g = g.sort_values("alpha_deg")
        ax.plot(g.alpha_deg, g.cl, marker="o", ms=2.5, lw=0.7, color=color, label=code)
    ax.set_xlabel("Angle of attack α (deg)"); ax.set_ylabel("Lift coefficient C$_L$")
    ax.set_title("Measured C$_L$(α) — converged campaign rows", loc="left", fontweight="bold")
    ax.axhline(0, color="k", lw=0.4)
    ax.legend(ncol=4, frameon=False, loc="lower right", handlelength=1.0)
    fig.savefig(FIG / "fig_cl_alpha.png")
    plt.close(fig)


def fig_lift_slope() -> None:
    slopes = []
    for (code, re), g in conv[conv.alpha_deg.abs() <= 6].groupby(["naca_code", "reynolds"]):
        if len(g) >= 3:
            s = float(np.polyfit(g.alpha_deg, g.cl, 1)[0]) * 180 / np.pi
            slopes.append((code, re, s))
    sl = pd.DataFrame(slopes, columns=["code", "re", "slope"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for code, g in sl.groupby("code"):
        g = g.sort_values("re")
        ax.plot(g.re / 1e6, g.slope, marker="o", ms=3, lw=0.8, color=C1, alpha=0.55)
    ax.axhline(2 * np.pi, color=C2, lw=1.2, ls="--")
    ax.text(6.15, 2 * np.pi + 0.06, "thin-airfoil theory $2\\pi$", color=C2,
            ha="right", fontsize=8)
    med = sl.groupby("re").slope.median()
    ax.plot(med.index / 1e6, med.values, marker="s", ms=5, lw=1.6, color="#222222",
            label="median across airfoils")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 6], ["1", "2", "3", "6"])
    ax.set_xlabel("Reynolds number Re (×10$^6$)")
    ax.set_ylabel("dC$_L$/dα (rad$^{-1}$)")
    ax.set_title("Measured lift slope vs Reynolds number", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "fig_lift_slope.png")
    plt.close(fig)


def fig_mesh() -> None:
    study = json.loads((ROOT / "reports/mesh_study/mesh_sensitivity.json").read_text())
    runs = [r for r in study["runs"] if r["iters"] == 6000 and r["status"] == "converged"]
    labels = [f"{r['level']}\n({r['cells']:,} cells)" for r in runs]
    x = np.arange(len(runs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    a1.bar(x, [r["cl"] for r in runs], width=0.55, color=C1, alpha=0.85)
    a1.set_ylabel("C$_L$"); a1.set_title("Lift", loc="left", fontweight="bold")
    a2.bar(x, [r["cd"] * 1e3 for r in runs], width=0.55, color=C3, alpha=0.85)
    a2.set_ylabel("C$_D$ (×10$^{-3}$)"); a2.set_title("Drag", loc="left", fontweight="bold")
    for a in (a1, a2):
        a.set_xticks(x, labels, fontsize=7); a.set_ylim(bottom=0)
    fig.suptitle("NACA 0012, α = 4°, Re = 6×10$^6$ — equal convergence depth (6000 iterations)",
                 fontsize=9, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig_mesh.png")
    plt.close(fig)


def fig_limit_cycle() -> None:
    cases = {
        "converged": "NACA0012_a+4.0_Re6e6",
        "unconverged": "NACA0012_a-2.0_Re6e6",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=False)
    for ax, (kind, cid) in zip(axes, cases.items()):
        cands = list((ROOT / "data/raw_cases" / cid).glob("postProcessing/**/coefficient.dat"))
        arr = np.loadtxt(cands[0], comments="#")
        t, cl = arr[:, 0], arr[:, 4]
        ax.plot(t, cl, lw=0.5, color=C1 if kind == "converged" else C2)
        w = min(100, len(cl))
        ax.axhline(cl[-w:].mean(), color="#222222", lw=0.8, ls=":")
        ax.set_xlabel("iteration"); ax.set_title(f"{kind} — {cid.split('_a')[0]} α={cid.split('_a')[1].split('_')[0]}",
                                                 loc="left", fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("C$_L$ (iteration history)")
    fig.suptitle("Steady-RANS limit cycle vs settled solution", fontsize=9,
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_limit_cycle.png")
    plt.close(fig)


def fig_cp_pred_vs_meas() -> None:
    sys.path.insert(0, str(ROOT))
    from src.models.inference import AeroSurrogateEngine
    eng = AeroSurrogateEngine(models_dir="/tmp/nf_final_bundle")
    holdout = ["NACA3414", "NACA5315", "NACA5611"]
    colors = {1e6: C1, 2e6: C3, 3e6: C2, 6e6: C4}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
    codes = {"NACA3414": "3414", "NACA5315": "5315", "NACA5611": "5611"}
    for ax, hull in zip(axes, holdout):
        g = conv[(conv.naca_code == hull[4:])].sort_values("alpha_deg")
        row = g.iloc[len(g) // 2]  # mid-range alpha case for the shape
        cid = row.case_id
        z = np.load(ROOT / "data/processed/cp_surface_distributions.npz")
        ids = [str(c) for c in z["case_ids"]]
        j = ids.index(cid)
        x = z["x_grid"]
        ax.plot(x, z["cp_upper"][j], lw=1.0, color="#222222", label="measured")
        ax.plot(x, z["cp_lower"][j], lw=1.0, color="#222222")
        m, p_, t_ = row.m, row.p, row.t
        res = eng.predict(m=m, p=p_, t=t_, alpha_deg=row.alpha_deg,
                          reynolds=row.reynolds)
        ax.plot(x, res["cp_upper"], lw=1.0, ls="--", color=C2, label="surrogate")
        ax.plot(x, res["cp_lower"], lw=1.0, ls="--", color=C2)
        ax.set_title(f"{hull} α={row.alpha_deg:+.0f}° Re={row.reynolds:.0e}",
                     fontsize=8, fontweight="bold", loc="left")
        ax.set_xlabel("x/c")
    axes[0].set_ylabel("C$_p$"); axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Surface pressure on grouped-holdout airfoils (never trained on)",
                 fontsize=9, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_cp_holdout.png")
    plt.close(fig)


def fig_walltime() -> None:
    ok = latest[latest.success == True].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for st, color, lbl in (("converged", C1, "converged"),
                           ("unconverged", C2, "unconverged")):
        seg = ok[ok.status == st]
        ax.hist(seg.wall_time_s.dropna(), bins=36, alpha=0.65, color=color, label=lbl)
    ax.set_xlabel("wall time per case (s)"); ax.set_ylabel("cases")
    ax.set_title("Campaign wall-time distribution (all completed rows)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig_walltime.png")
    plt.close(fig)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.axis("off")
    stages = [
        ("Geometry\nNACA4\n+ CST", "#EFF6FF"),
        ("Mesh\nC-grid\nadaptive", "#EFF6FF"),
        ("RANS\nsimpleFoam\nSST", "#EFF6FF"),
        ("Extract\nCp + forces\nQC", "#F0FDF4"),
        ("Dataset\nCSV/NPZ\natomic", "#F0FDF4"),
        ("Surrogate\nMLP +\ndecoder", "#FDF2F8"),
        ("Serving\nUQ +\nOOD guard", "#FDF2F8"),
    ]
    w, h, gap = 0.115, 0.62, 0.03
    for i, (label, color) in enumerate(stages):
        x0 = i * (w + gap)
        ax.add_patch(plt.Rectangle((x0, 0.16), w, h, facecolor=color,
                                   edgecolor="#94a3b8", lw=0.8, joinstyle="round"))
        ax.text(x0 + w / 2, 0.47, label, ha="center", va="center", fontsize=7.0)
        if i:
            ax.annotate("", xy=(x0 - gap / 2 + 0.004, 0.47), xytext=(x0 - gap / 2 - 0.004, 0.47),
                        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.0))
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig_architecture.png")
    plt.close(fig)


for fn in (fig_architecture, fig_coverage, fig_cl_alpha, fig_lift_slope,
           fig_mesh, fig_limit_cycle, fig_cp_pred_vs_meas, fig_walltime):
    fn()
    print("done:", fn.__name__)
print("figures ->", FIG)
