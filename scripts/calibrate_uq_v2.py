"""UQ v2: Mondrian (alpha-regime) conformal calibration + simultaneous Cp band.

Reproduces the training notebook's airfoil-grouped split exactly (rng seed 42),
identifies the 52 validation/calibration rows, and recomputes conformal
quantiles from bundle-vs-CFD residuals on those rows:

  - marginal (reproduces the shipped quantiles; sanity check vs meta)
  - Mondrian: per-alpha-regime quantiles (neg / low / high)
  - simultaneous Cp band: max-statistic scale factor lambda over calibration
    cases, applied to the per-station pointwise half-widths

Coverage of every variant is then measured on the UNTOUCHED test rows
(61 rows, airfoils 3414/5315/5611). No retraining; test rows are never used
for calibration.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.inference import AeroSurrogateEngine  # noqa: E402

SEED = 42
CSV = ROOT / "data/processed/airfoil_cfd_dataset.csv"
META = ROOT / "models/surrogate_meta.json"
OUT = ROOT / "reports/evaluation/uq_v2_calibration.json"


def conformal_quantile(abs_res: np.ndarray, coverage: float = 0.95) -> float:
    """Finite-sample order statistic, same formula as the training notebook."""
    rank = min(math.ceil((len(abs_res) + 1) * coverage), len(abs_res))
    return float(np.sort(abs_res)[rank - 1])


def regime(alpha: float) -> str:
    if alpha < 0:
        return "neg"
    if alpha <= 6:
        return "low"
    return "high"


def main() -> None:
    meta = json.loads(META.read_text())
    eng = AeroSurrogateEngine(models_dir=ROOT / "models")
    df = pd.read_csv(CSV, dtype={"case_id": str, "naca_code": str})
    latest = df.drop_duplicates("case_id", keep="last")
    conv = latest[latest.status == "converged"].copy()
    conv["naca_code"] = conv.naca_code.str.zfill(4)
    conv = conv.sort_values(["naca_code", "reynolds", "alpha_deg"]).reset_index(drop=True)

    # ---- reproduce the notebook's airfoil-grouped split ----
    groups = conv["naca_code"].values
    unique_codes = sorted(conv["naca_code"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_codes)
    n_test = max(2, int(round(0.15 * len(unique_codes))))
    n_val = max(2, int(round(0.15 * len(unique_codes))))
    test_codes = set(unique_codes[:n_test])
    val_codes = set(unique_codes[n_test:n_test + n_val])
    is_test = np.isin(groups, list(test_codes))
    is_val = np.isin(groups, list(val_codes))
    print(f"test airfoils: {sorted(test_codes)}")
    print(f"calibration airfoils: {sorted(val_codes)} | cal rows: {int(is_val.sum())}")

    npz = np.load(ROOT / "data/processed/cp_surface_distributions.npz")
    npz_ids = [str(c) for c in npz["case_ids"]]
    meas = {cid: (np.asarray(npz["cp_upper"][i], float),
                  np.asarray(npz["cp_lower"][i], float))
            for i, cid in enumerate(npz_ids)}

    def predict_row(row) -> dict:
        return eng.predict(m=row.m, p=row.p, t=row.t,
                           alpha_deg=row.alpha_deg, reynolds=row.reynolds)

    # ---- residuals on the calibration (val) rows ----
    cal_res = {"cl": [], "cd": [], "cm": []}
    cal_cp = []  # (case_id, alpha, err_upper_abs, err_lower_abs)
    for _, row in conv[is_val].iterrows():
        res = predict_row(row)
        cal_res["cl"].append(abs(res["cl"] - row.cl))
        cal_res["cd"].append(abs(res["cd"] - row.cd))
        cal_res["cm"].append(abs(res["cm"] - row.cm))
        cid = row.case_id
        if cid in meas:
            mu, ml = meas[cid]
            cal_cp.append((cid, row.alpha_deg,
                           np.abs(np.asarray(res["cp_upper"]) - mu),
                           np.abs(np.asarray(res["cp_lower"]) - ml)))
    res_cl = np.array(cal_res["cl"]); res_cd = np.array(cal_res["cd"])
    res_cm = np.array(cal_res["cm"])

    # marginal quantiles (reproduce shipped)
    q_marginal = {"cl": conformal_quantile(res_cl),
                  "cd": conformal_quantile(res_cd),
                  "cm": conformal_quantile(res_cm)}
    shipped = meta["conformal"]
    print("\nmarginal quantiles (reproduced vs shipped):")
    for t in ("cl", "cd", "cm"):
        print(f"  {t}: {q_marginal[t]:.4f} vs {shipped['q_' + t]:.4f}")

    # ---- Mondrian: per-regime quantiles ----
    regimes = {"neg": [], "low": [], "high": []}
    for (_, row), rc, rd, rm in zip(conv[is_val].iterrows(), res_cl, res_cd, res_cm):
        regimes[regime(row.alpha_deg)].append((rc, rd, rm))
    q_mondrian = {}
    for name, triples in regimes.items():
        arr = np.array(triples) if triples else np.zeros((0, 3))
        q_mondrian[name] = {
            "n": int(len(arr)),
            "q_cl": conformal_quantile(arr[:, 0]) if len(arr) else None,
            "q_cd": conformal_quantile(arr[:, 1]) if len(arr) else None,
            "q_cm": conformal_quantile(arr[:, 2]) if len(arr) else None,
        }
    print("\nMondrian per-regime quantiles:")
    for name, d in q_mondrian.items():
        print(f"  {name}: n={d['n']} q_cl={d['q_cl']} q_cd={d['q_cd']} q_cm={d['q_cm']}")

    # ---- simultaneous Cp band: max-statistic lambda over calibration ----
    lam_cal = []
    for _, _, eu, el in cal_cp:
        hu = np.maximum(meta["conformal"]["q_cp_upper"], 1e-6)
        hl = np.maximum(meta["conformal"]["q_cp_lower"], 1e-6)
        lam_cal.append(max(np.max(eu / hu), np.max(el / hl)))
    lambda_cal = float(np.quantile(np.array(lam_cal), 0.95))
    print(f"\nsimultaneous lambda (calibration max-statistic, 95%): {lambda_cal:.3f}")

    # ---- coverage evaluation on the untouched TEST rows ----
    def coverage(sub, q):
        hits = {t: 0 for t in ("cl", "cd", "cm")}
        for t, arr_key in (("cl", "cl"), ("cd", "cd"), ("cm", "cm")):
            res = {"cl": res_cl_test, "cd": res_cd_test, "cm": res_cm_test}[arr_key]
        return hits

    # test-row residuals
    test_res = {"cl": [], "cd": [], "cm": []}
    test_rows = []
    test_cp = []
    for _, row in conv[is_test].iterrows():
        res = predict_row(row)
        test_res["cl"].append(abs(res["cl"] - row.cl))
        test_res["cd"].append(abs(res["cd"] - row.cd))
        test_res["cm"].append(abs(res["cm"] - row.cm))
        test_rows.append((row.naca_code, row.alpha_deg))
        cid = row.case_id
        if cid in meas:
            mu, ml = meas[cid]
            test_cp.append((np.abs(np.asarray(res["cp_upper"]) - mu),
                            np.abs(np.asarray(res["cp_lower"]) - ml)))
    rcl = np.array(test_res["cl"]); rcd = np.array(test_res["cd"]); rcm = np.array(test_res["cm"])
    n_test_rows = len(rcl)

    def cov_mask(mask, q):
        return float(np.mean(mask))

    def covered(arr, q):
        return arr <= q

    # marginal coverage on test
    cov_marginal = {t: float(np.mean(np.array(test_res[t]) <= q_marginal[t]))
                    for t in ("cl", "cd", "cm")}
    # Mondrian coverage on test (regime of each test row decides the quantile)
    hits = {"cl": 0, "cd": 0, "cm": 0}
    for (code, alpha), rc, rd, rm in zip(test_rows, rcl, rcd, rcm):
        d = q_mondrian[regime(alpha)]
        for t, v, q in (("cl", rc, d["q_cl"]), ("cd", rd, d["q_cd"]),
                        ("cm", rm, d["q_cm"])):
            if v <= q:
                hits[t] += 1
    cov_mondrian = {t: hits[t] / n_test_rows for t in ("cl", "cd", "cm")}

    # Cp coverage on test: pointwise (shipped band) vs simultaneous (lambda-scaled)
    q_cp_u = np.maximum(meta["conformal"]["q_cp_upper"], 1e-6)
    q_cp_l = np.maximum(meta["conformal"]["q_cp_lower"], 1e-6)
    pw_hits, sim_hits, n_st = 0, 0, 0
    for eu, el in test_cp:
        ok_u = (eu <= q_cp_u) & (el <= q_cp_l)
        pw_hits += int(ok_u.sum()) + int((el <= q_cp_l).sum())
        n_st += 2 * len(eu)
        if bool((eu <= lambda_cal * q_cp_u).all() and (el <= lambda_cal * q_cp_l).all()):
            sim_hits += 1
    cp_pointwise_cov = pw_hits / n_st
    cp_sim_cov = sim_hits / len(test_cp)

    result = {
        "calibration_rows": int(is_val.sum()),
        "test_rows": int(n_test_rows),
        "test_airfoils": sorted(test_codes),
        "calibration_airfoils": sorted(val_codes),
        "marginal_quantiles_reproduced": q_marginal,
        "marginal_coverage_on_test": cov_marginal,
        "mondrian_regimes": q_mondrian,
        "mondrian_coverage_on_test": cov_mondrian,
        "cp": {
            "pointwise_coverage_on_test": cp_pointwise_cov,
            "simultaneous_coverage_marginal_band_on_test": cp_sim_cov,
            "simultaneous_lambda_from_calibration": lambda_cal,
            "note": ("lambda is the 95th percentile of per-case max(|err|/band) "
                     "over calibration rows; the scaled band is then evaluated "
                     "on untouched test rows"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== COVERAGE ON TEST ROWS (n=%d) ===" % n_test_rows)
    for t in ("cl", "cd", "cm"):
        print(f"  {t}: marginal {cov_marginal[t]:.3f} -> Mondrian {cov_mondrian[t]:.3f}")
    print(f"  Cp pointwise {cp_pointwise_cov:.3f} | simultaneous (marginal band) "
          f"{cp_sim_cov:.3f} | lambda {lambda_cal:.3f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
