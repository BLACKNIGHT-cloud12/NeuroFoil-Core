"""Quality-control audit for the real-CFD campaign dataset.

Checks the campaign CSV + Cp NPZ before any surrogate training:
  1. Schema and provenance (converged rows must be data_source="openfoam").
  2. Grid completeness against the expected NACA-LHS campaign grid.
  3. Convergence-status and wall-time summary.
  4. Physics-range flags (NaN/inf, non-positive drag, lift-slope sanity).
  5. Cp-field integrity (alignment, finiteness, delta_cp consistency).
  6. Grouped-split readiness (per-airfoil coverage).

Outputs a text report, PNG diagnostics, and a verdict JSON to --report-dir.
Exit code 1 with --strict if any check fails.

Usage:
  python scripts/qc_campaign.py                          # defaults below
  python scripts/qc_campaign.py --strict                 # gate for training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dataset_campaign import CSV_PATH, NPZ_PATH, sample_airfoil_codes  # noqa: E402

PROVENANCE_COLUMNS = ("case_id", "naca_code", "alpha_deg", "reynolds",
                      "cl", "cd", "cm", "status", "data_source")
TARGETS = ("cl", "cd", "cm")
# RANS k-omega SST lift slope on this grid is expected between ~0.75x and
# ~1.25x thin-airfoil theory (2*pi ~ 6.28 rad^-1); flags stalled/broken cases.
LIFT_SLOPE_RANGE = (4.7, 7.9)
MAX_ABS_CL = 2.2  # pre-stall bound at alpha=14deg for this thickness range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC audit for the campaign dataset.")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--npz", type=Path, default=NPZ_PATH)
    p.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "qc")
    p.add_argument("--airfoils", type=int, default=21)
    p.add_argument("--alphas", type=str, default="-6,-2,2,6,10,14")
    p.add_argument("--res", type=str, default="1e6,2e6,3e6,6e6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 when any check fails (default: report only).")
    return p.parse_args()


def expected_case_grid(n_airfoils: int, alphas: list[float],
                       reynolds: list[float], seed: int) -> set[str]:
    """Replicate the campaign's case_id format for the full expected grid."""
    codes = sample_airfoil_codes(n_airfoils, seed)
    ids = set()
    for code in codes:
        for reynolds_value in reynolds:
            for alpha in alphas:
                ids.add(
                    f"NACA{code}_a{alpha:+.1f}_Re{reynolds_value:.0e}"
                    .replace("e+0", "e").replace("e+", "e")
                )
    return ids


def check_schema_and_provenance(df: pd.DataFrame, findings: list[str]) -> bool:
    missing = [c for c in PROVENANCE_COLUMNS if c not in df.columns]
    if missing:
        findings.append(f"FAIL schema: missing columns {missing}")
        return False
    converged = df["status"].eq("converged")
    bad_source = df.loc[converged, "data_source"].fillna("<missing>")
    n_bad = int((bad_source != "openfoam").sum())
    if n_bad:
        findings.append(f"FAIL provenance: {n_bad} converged rows without "
                        'data_source="openfoam"')
        return False
    if "is_synthetic" in df.columns and bool(df["is_synthetic"].fillna(True).any()):
        findings.append("FAIL provenance: is_synthetic rows present")
        return False
    dup = int(df["case_id"].duplicated().sum())
    if dup:
        findings.append(f"FAIL schema: {dup} duplicate case_id rows")
        return False
    findings.append(f"PASS schema/provenance ({len(df)} rows, all unique, "
                    f"{int(converged.sum())} converged openfoam rows)")
    return True


def check_grid_completeness(df: pd.DataFrame, expected: set[str],
                            findings: list[str]) -> bool:
    present = set(df["case_id"])
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        findings.append(f"WARN grid: {len(missing)}/{len(expected)} expected cases "
                        f"missing (e.g. {missing[:5]})")
    if extra:
        findings.append(f"WARN grid: {len(extra)} rows outside the expected grid "
                        f"(e.g. {extra[:5]})")
    if not missing and not extra:
        findings.append(f"PASS grid: all {len(expected)} expected cases present")
    return not missing


def check_physics(df: pd.DataFrame, findings: list[str]) -> list[str]:
    flagged: list[str] = []
    for target in TARGETS:
        values = pd.to_numeric(df[target], errors="coerce")
        n_nonfinite = int((~np.isfinite(values) | values.isna()).sum())
        if n_nonfinite:
            flagged.extend(df.loc[values.isna() | ~np.isfinite(values), "case_id"])
            findings.append(f"FAIL physics: {n_nonfinite} non-finite {target} values")
    cd = pd.to_numeric(df["cd"], errors="coerce")
    n_negative = int((cd <= 0).sum())
    if n_negative:
        flagged.extend(df.loc[cd <= 0, "case_id"])
        findings.append(f"FAIL physics: {n_negative} rows with CD <= 0")
    n_overcl = int(pd.to_numeric(df["cl"], errors="coerce").abs().gt(MAX_ABS_CL).sum())
    if n_overcl:
        flagged.extend(df.loc[pd.to_numeric(df["cl"], errors="coerce").abs()
                              .gt(MAX_ABS_CL), "case_id"])
        findings.append(f"WARN physics: {n_overcl} rows with |CL| > {MAX_ABS_CL}")

    slope_issues = _check_lift_slopes(df)
    flagged.extend(slope_issues)
    if slope_issues:
        findings.append(f"WARN physics: odd lift slope on {len(slope_issues)} airfoil/Re "
                        f"series (e.g. {slope_issues[:3]}); inspect CL(alpha) plot")
    if not any(f.startswith(("FAIL physics", "WARN physics")) for f in findings):
        findings.append("PASS physics: all targets finite, CD > 0, "
                        f"|CL| <= {MAX_ABS_CL}, lift slopes within "
                        f"{LIFT_SLOPE_RANGE} rad^-1")
    return sorted(set(flagged))


def _check_lift_slopes(df: pd.DataFrame) -> list[str]:
    """Linear dCL/dalpha over the attached range |alpha| <= 6 deg, per airfoil+Re."""
    issues: list[str] = []
    attached = df[pd.to_numeric(df["alpha_deg"], errors="coerce").abs() <= 6.0]
    for (code, reynolds_value), group in attached.groupby(["naca_code", "reynolds"]):
        group = group.assign(a=pd.to_numeric(group["alpha_deg"], errors="coerce"),
                             cl=pd.to_numeric(group["cl"], errors="coerce")).dropna(
                                 subset=["a", "cl"])
        if len(group) < 3:
            continue
        slope = float(np.polyfit(group["a"], group["cl"], 1)[0])
        if not LIFT_SLOPE_RANGE[0] <= slope <= LIFT_SLOPE_RANGE[1]:
            issues.append(f"{code}@Re{reynolds_value:.0e} slope={slope:.2f}")
    return issues


def check_cp_fields(df: pd.DataFrame, npz: np.lib.npyio.NpzFile,
                    findings: list[str]) -> list[str]:
    flagged: list[str] = []
    cp_ids = [str(cid) for cid in npz["case_ids"]]
    x_grid = np.asarray(npz["x_grid"], dtype=float)
    if len(cp_ids) != len(set(cp_ids)):
        findings.append("FAIL cp: duplicate case_ids in NPZ")
        return flagged
    if not np.all(np.diff(x_grid) > 0):
        findings.append("FAIL cp: x_grid not strictly increasing")
    csv_with_cp = set(df.loc[df.get("cp_available", pd.Series(dtype=object))
                             .eq(True), "case_id"]) if "cp_available" in df else set()
    missing_field = sorted(csv_with_cp - set(cp_ids))
    if missing_field:
        findings.append(f"FAIL cp: {len(missing_field)} CSV rows claim cp_available "
                        f"but lack an NPZ field (e.g. {missing_field[:3]})")
    for name in ("cp_upper", "cp_lower", "delta_cp"):
        arr = np.asarray(npz[name], dtype=float)
        if arr.shape != (len(cp_ids), x_grid.size):
            findings.append(f"FAIL cp: {name} shape {arr.shape} != "
                            f"({len(cp_ids)}, {x_grid.size})")
            return flagged
        nonfinite = int((~np.isfinite(arr)).sum())
        if nonfinite:
            flagged.extend(cp_ids[i] for i in np.unique(np.where(~np.isfinite(arr))[0]))
            findings.append(f"FAIL cp: {nonfinite} non-finite values in {name}")
    upper = np.asarray(npz["cp_upper"], dtype=float)
    lower = np.asarray(npz["cp_lower"], dtype=float)
    delta = np.asarray(npz["delta_cp"], dtype=float)
    # Extractor convention: delta_cp = cp_lower - cp_upper (positive where the
    # lower surface carries more pressure; integrates to the Cp-based lift).
    bad_delta = np.abs((lower - upper) - delta) > 1e-4
    if bad_delta.any():
        flagged.extend(cp_ids[i] for i in np.unique(np.where(bad_delta)[0]))
        findings.append(f"FAIL cp: {int(bad_delta.any(axis=1).sum())} cases where "
                        "delta_cp != cp_lower - cp_upper beyond 1e-4")
    jumps = np.abs(np.diff(upper, axis=1))
    jumpy = np.where((jumps > 1.5).any(axis=1))[0]
    if len(jumpy):
        findings.append(f"WARN cp: {len(jumpy)} cases with |dCp/dx| > 1.5 jumps "
                        f"(e.g. {[cp_ids[i] for i in jumpy[:3]]})")
        flagged.extend(cp_ids[i] for i in jumpy)
    if not any(f.startswith(("FAIL cp", "WARN cp")) for f in findings):
        findings.append(f"PASS cp: {len(cp_ids)} fields aligned with x_grid "
                        f"({x_grid.size} pts), finite, delta_cp consistent")
    return sorted(set(flagged))


def check_split_readiness(df: pd.DataFrame, findings: list[str]) -> None:
    counts = df.groupby("naca_code")["case_id"].count()
    thin = counts[counts < 4]
    if len(thin):
        findings.append(f"WARN splits: {len(thin)} airfoils have <4 cases "
                        f"({dict(thin)}); grouped holdout quality suffers")
    else:
        findings.append(f"PASS splits: {len(counts)} airfoils, "
                        f"{counts.min()}-{counts.max()} cases each — "
                        "grouped per-airfoil holdout viable")


def write_plots(df: pd.DataFrame, npz: np.lib.npyio.NpzFile,
                report_dir: Path) -> None:
    df = df.assign(cl=pd.to_numeric(df["cl"], errors="coerce"),
                   cd=pd.to_numeric(df["cd"], errors="coerce"),
                   alpha=pd.to_numeric(df["alpha_deg"], errors="coerce"),
                   re=pd.to_numeric(df["reynolds"], errors="coerce"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    df["status"].value_counts().plot.bar(ax=axes[0, 0], color="#2a7fb8")
    axes[0, 0].set_title("Convergence status")
    axes[0, 0].set_ylabel("cases")
    if "wall_time_s" in df.columns:
        df["wall_time_s"].dropna().plot.hist(bins=40, ax=axes[0, 1], color="#3a3")
        axes[0, 1].set_title("Wall time per case (s)")
    for code, group in df.groupby("naca_code"):
        group = group.sort_values("alpha")
        axes[1, 0].plot(group["alpha"], group["cl"], marker="o", ms=3, lw=0.7)
    axes[1, 0].set_title("CL(alpha) per airfoil")
    axes[1, 0].set_xlabel("alpha (deg)")
    axes[1, 0].axhline(0.0, color="k", lw=0.4)
    for reynolds_value, group in df.groupby("re"):
        axes[1, 1].scatter(group["alpha"], group["cd"], s=8, alpha=0.5,
                           label=f"Re {reynolds_value:.1e}")
    axes[1, 1].set_title("CD(alpha) by Reynolds")
    axes[1, 1].set_xlabel("alpha (deg)")
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(report_dir / "qc_overview.png", dpi=130)
    plt.close(fig)

    cp_ids = [str(cid) for cid in npz["case_ids"]]
    x_grid = np.asarray(npz["x_grid"], dtype=float)
    if cp_ids:
        fig, ax = plt.subplots(figsize=(11, 5))
        for idx in range(0, len(cp_ids), max(1, len(cp_ids) // 12)):
            ax.plot(x_grid, np.asarray(npz["cp_upper"])[idx], lw=0.7)
            ax.plot(x_grid, np.asarray(npz["cp_lower"])[idx], lw=0.7)
        ax.set_title(f"Cp spot-check ({min(12, len(cp_ids))} of {len(cp_ids)} fields)")
        ax.set_xlabel("x/c")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(report_dir / "qc_cp_spotcheck.png", dpi=130)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    findings: list[str] = []
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"case_id": str, "naca_code": str})
    npz = np.load(args.npz, allow_pickle=False)

    schema_ok = check_schema_and_provenance(df, findings)
    expected = expected_case_grid(args.airfoils, [float(a) for a in args.alphas.split(",")],
                                  [float(r) for r in args.res.split(",")], args.seed)
    grid_ok = check_grid_completeness(df, expected, findings)
    physics_flags = check_physics(df, findings)
    cp_flags = check_cp_fields(df, npz, findings)
    check_split_readiness(df, findings)

    write_plots(df, npz, args.report_dir)

    verdict = {
        "schema_ok": schema_ok,
        "grid_complete": grid_ok,
        "n_rows": len(df),
        "n_expected_cases": len(expected),
        "n_converged": int(df["status"].eq("converged").sum()),
        "physics_flagged": physics_flags,
        "cp_flagged": cp_flags,
        "n_flagged_cases": len(set(physics_flags) | set(cp_flags)),
        "findings": findings,
    }
    verdict_path = args.report_dir / "qc_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("=" * 72)
    print("CAMPAIGN DATASET QC")
    print("=" * 72)
    for finding in findings:
        print(f"  {finding}")
    print("-" * 72)
    print(f"Flagged cases: {verdict['n_flagged_cases']} | "
          f"report: {args.report_dir}")
    failed = any(f.startswith("FAIL") for f in findings)
    print("VERDICT:", "FAIL" if failed else "PASS")
    if failed and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
