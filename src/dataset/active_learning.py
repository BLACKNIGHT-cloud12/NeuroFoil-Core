"""
Conformal-guided active learning for the CFD campaign.

Active-learning loop:
  round k: train the surrogate on all data collected so far (Kaggle notebook)
  -> models/ holds weights + conformal meta. This module then scores the
  *unqueried* candidate pool, preferring designs where the surrogate is least
  certain (Mahalanobis distance on the training manifold + stall-proximity
  risk), selects a diverse batch (greedy max-min in feature space), and writes
  a manifest that scripts/run_dataset_campaign.py --manifest consumes for the
  next CFD round.

With the current UQ stack the per-candidate uncertainty signal is the
Mahalanobis distance (the conformal quantiles in surrogate_meta.json are
global). Upgrading to per-candidate conformal widths (conditional conformal)
is the natural extension once the first two rounds are done.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.inference import AeroSurrogateEngine


@dataclass(frozen=True)
class Candidate:
    """One unqueried design point in the active-learning pool."""

    case_id: str
    shape_id: str
    m: float
    p: float
    t: float
    alpha_deg: float
    reynolds: float
    cst_upper: tuple[float, ...] | None = None
    cst_lower: tuple[float, ...] | None = None


class ActiveLearningSelector:
    """Scores candidate designs and selects the next diverse CFD batch."""

    STALL_WEIGHT = 2.0  # acquisition weight for stall-proximity risk

    def __init__(self, models_dir: Path | str, dataset_csv: Path | str):
        self.engine = AeroSurrogateEngine(models_dir=models_dir)
        if self.engine.backend != "neural":
            raise RuntimeError(
                "Active learning requires trained surrogate weights in models/ "
                "(run the Kaggle notebook on the previous round's dataset first)."
            )
        self.dataset_csv = Path(dataset_csv)

    # ------------------------------------------------------------------
    def _feature_vector(self, c: Candidate) -> np.ndarray:
        return np.array([c.m, c.p, c.t, c.alpha_deg, c.reynolds], dtype=float)

    def _calibrate_manifold(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and (ridged) inverse covariance of the training manifold."""
        df = pd.read_csv(self.dataset_csv)
        feats = df[["m", "p", "t", "alpha_deg", "reynolds"]].values
        mean = feats.mean(axis=0)
        cov = np.cov(feats, rowvar=False)
        cov = np.atleast_2d(cov) + np.eye(cov.shape[0]) * 1e-6
        return mean, np.linalg.inv(cov)

    # ------------------------------------------------------------------
    def score_candidates(self, candidates: list[Candidate]) -> list[float]:
        """
        Acquisition score per candidate: normalized Mahalanobis distance to the
        training manifold plus stall-proximity risk. Higher = more informative.
        """
        mean, inv_cov = self._calibrate_manifold()
        std = np.sqrt(np.diag(np.linalg.inv(inv_cov)))

        scores: list[float] = []
        for c in candidates:
            x = self._feature_vector(c)
            diff = x - mean
            d_m = float(np.sqrt(max(0.0, diff @ inv_cov @ diff)))
            stall = 1.0 / (1.0 + np.exp(-min(15.0, max(-15.0, abs(c.alpha_deg) - 10.5))))
            scores.append(d_m / np.max(std) + self.STALL_WEIGHT * stall)
        return scores

    # ------------------------------------------------------------------
    def select_batch(
        self,
        candidates: list[Candidate],
        batch_size: int,
        diversity_pool_factor: int = 5,
    ) -> list[Candidate]:
        """
        Pick `batch_size` candidates: take the top
        `diversity_pool_factor * batch_size` by acquisition score, then greedily
        select the batch maximizing min-distance in normalized feature space
        (avoids spending the whole round on one corner of the manifold).
        """
        scores = self.score_candidates(candidates)
        order = np.argsort(scores)[::-1]
        pool = [candidates[i] for i in order[: diversity_pool_factor * batch_size]]

        feats = np.array([self._feature_vector(c) for c in pool])
        span = np.maximum(feats.max(axis=0) - feats.min(axis=0), 1e-12)
        feats_n = feats / span

        selected_idx = [int(np.argmax(scores[: len(pool)]))] if pool else []
        while len(selected_idx) < min(batch_size, len(pool)):
            dist = np.min(
                np.linalg.norm(feats_n[:, None, :] - feats_n[None, selected_idx, :], axis=2),
                axis=1,
            )
            dist[selected_idx] = -1.0
            selected_idx.append(int(np.argmax(dist)))
        return [pool[i] for i in selected_idx]


# ----------------------------------------------------------------------
def build_candidate_pool(
    csv_path: Path | str,
    n_shapes: int,
    alphas: list[float],
    reynolds: list[float],
    sampler: str = "cst",
    seed: int = 1234,
) -> list[Candidate]:
    """Enumerate the unqueried candidate pool (already-run case_ids excluded)."""
    if sampler == "cst":
        from src.dataset.cst_sampler import sample_general_cst_airfoils

        shapes = sample_general_cst_airfoils(n_shapes, seed=seed)
    else:
        from scripts.run_dataset_campaign import sample_airfoil_codes

        shapes = sample_airfoil_codes(n_shapes, seed=seed)

    done = set()
    if Path(csv_path).exists():
        done = set(pd.read_csv(csv_path)["case_id"])

    pool: list[Candidate] = []
    for s in shapes:
        shape_id = getattr(s, "sample_id", s)  # CSTSample.sample_id or NACA code
        m = s.measured_camber if sampler == "cst" else int(shape_id[0]) / 100.0
        p = s.geometry.p if sampler == "cst" else int(shape_id[1]) / 10.0
        t = s.measured_thickness if sampler == "cst" else int(shape_id[2:4]) / 100.0
        cst_u = getattr(s, "cst_upper", None)
        cst_l = getattr(s, "cst_lower", None)
        for a in alphas:
            for re_ in reynolds:
                tag = f"Re{re_:.0e}".replace("e+0", "e").replace("e+", "e")
                case_id = f"{shape_id}_a{a:+.1f}_{tag}"
                if case_id in done:
                    continue
                pool.append(
                    Candidate(
                        case_id=case_id, shape_id=shape_id, m=m, p=p, t=t,
                        alpha_deg=a, reynolds=re_, cst_upper=cst_u, cst_lower=cst_l,
                    )
                )
    return pool


def write_manifest(cases: list[Candidate], path: Path | str, round_index: int) -> Path:
    """Write the selected batch as a campaign-driver manifest."""
    payload = {
        "round": round_index,
        "n_cases": len(cases),
        "cases": [asdict(c) for c in cases],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
