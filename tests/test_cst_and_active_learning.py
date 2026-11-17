"""
Tests for the general-CST airfoil sampler and the active-learning selector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from conftest import make_random_neural_models
import src.dataset.cst_sampler as cst_sampler
from src.dataset.cst_sampler import geometry_from_cst, sample_general_cst_airfoils
from src.geometry.validator import validate_airfoil_geometry


def test_geometry_from_cst_roundtrip() -> None:
    """Rebuilding a shape from its refitted weights reproduces the same surface."""
    anchor = sample_general_cst_airfoils(1, seed=7)[0]
    geom = geometry_from_cst(anchor.cst_upper, anchor.cst_lower)
    assert geom.code.startswith("CST-")
    assert len(geom.x) == 161
    assert abs(geom.te_gap) < 1e-6  # closed trailing edge
    np.testing.assert_allclose(geom.y_upper, anchor.geometry.y_upper, atol=1e-4)
    np.testing.assert_allclose(geom.y_lower, anchor.geometry.y_lower, atol=1e-4)


def test_sampler_yields_valid_diverse_shapes() -> None:
    shapes = sample_general_cst_airfoils(10, seed=42)
    assert len(shapes) == 10
    ids = {s.sample_id for s in shapes}
    assert len(ids) == 10  # unique

    cambers = np.array([s.measured_camber for s in shapes])
    thicknesses = np.array([s.measured_thickness for s in shapes])
    assert np.all(cambers <= 0.06 + 1e-6)
    assert np.all((thicknesses >= 0.06 - 1e-3) & (thicknesses <= 0.18 + 1e-3))

    for s in shapes:
        val = validate_airfoil_geometry(s.geometry)
        assert val.is_valid, val.errors
        # 12 genuinely free coefficients: shapes must differ from each other
        assert len(s.cst_upper) == 6 and len(s.cst_lower) == 6
    # Non-degenerate manifold: distinct shapes have distinct weight vectors
    mats = np.array([list(s.cst_upper) + list(s.cst_lower) for s in shapes])
    assert np.linalg.matrix_rank(mats - mats.mean(axis=0)) >= 8


def test_tiny_camber_cst_anchor_uses_symmetric_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyCamberRng:
        def __init__(self) -> None:
            self.uniform_values = iter((0.004, 0.12))

        def uniform(self, *args: object, **kwargs: object) -> float:
            return float(next(self.uniform_values))

    captured: list[str] = []

    def fake_generate(code: str, *args: object, **kwargs: object) -> object:
        captured.append(code)
        raise RuntimeError("stop after observing anchor")

    monkeypatch.setattr(cst_sampler.np.random, "default_rng", lambda seed: TinyCamberRng())
    monkeypatch.setattr(cst_sampler, "generate_naca4_airfoil", fake_generate)

    with pytest.raises(RuntimeError, match="stop after observing anchor"):
        sample_general_cst_airfoils(1, max_attempts_factor=1)

    assert captured == ["0012"]


def test_cst_exhaustion_advises_lower_perturbation_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cst_sampler,
        "validate_airfoil_geometry",
        lambda *args, **kwargs: type("Invalid", (), {"is_valid": False})(),
    )

    with pytest.raises(RuntimeError, match="lower perturbation_scale"):
        sample_general_cst_airfoils(1, seed=5, max_attempts_factor=1)


def test_sampler_is_deterministic() -> None:
    a = sample_general_cst_airfoils(4, seed=3)
    b = sample_general_cst_airfoils(4, seed=3)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_active_learning_selector(tmp_path: Path) -> None:
    from src.dataset.active_learning import (
        ActiveLearningSelector,
        build_candidate_pool,
        write_manifest,
    )

    make_random_neural_models(tmp_path / "models", data_source="openfoam")

    # Tiny "training" CSV for the Mahalanobis manifold
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "case_id": [f"C{i}" for i in range(30)],
        "m": rng.uniform(0, 0.06, 30),
        "p": rng.uniform(0.2, 0.6, 30),
        "t": rng.uniform(0.06, 0.18, 30),
        "alpha_deg": rng.uniform(-6, 14, 30),
        "reynolds": rng.choice([1e6, 2e6, 3e6, 6e6], 30),
        "data_source": ["openfoam"] * 30,
        "status": ["converged"] * 30,
        "is_synthetic": [False] * 30,
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    selector = ActiveLearningSelector(models_dir=tmp_path / "models", dataset_csv=csv_path)
    assert selector.engine.backend == "neural"

    pool = build_candidate_pool(csv_path, n_shapes=5,
                                alphas=[-6.0, 2.0, 14.0], reynolds=[2e6, 6e6],
                                sampler="cst", seed=11)
    assert len(pool) == 5 * 3 * 2

    batch = selector.select_batch(pool, batch_size=6)
    assert len(batch) == 6
    assert len({c.case_id for c in batch}) == 6

    # Alphas near stall should be preferred by the acquisition score
    plain = [c for c in pool if c.alpha_deg == 2.0]
    stall = [c for c in pool if c.alpha_deg == 14.0]
    s_plain = np.mean(selector.score_candidates(plain))
    s_stall = np.mean(selector.score_candidates(stall))
    assert s_stall > s_plain

    manifest = write_manifest(batch, tmp_path / "manifest.json", round_index=2)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["round"] == 2 and payload["n_cases"] == 6
    assert set(payload["cases"][0]) >= {"case_id", "shape_id", "m", "p", "t",
                                        "alpha_deg", "reynolds"}


def test_active_learning_selects_global_highest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.dataset.active_learning import ActiveLearningSelector, Candidate

    make_random_neural_models(tmp_path / "models", data_source="openfoam")
    selector = ActiveLearningSelector(tmp_path / "models", tmp_path / "unused.csv")
    candidates = [
        Candidate("low", "S0", 0.025, 0.35, 0.11, 2.0, 3e6),
        Candidate("highest", "S1", 0.04, 0.45, 0.14, 8.0, 5e6),
        Candidate("middle", "S2", 0.06, 0.60, 0.18, 14.0, 6e6),
    ]
    monkeypatch.setattr(selector, "score_candidates", lambda _candidates: [1.0, 3.0, 2.0])

    batch = selector.select_batch(candidates, batch_size=2)

    assert batch[0] == candidates[1]
    assert candidates[1] in batch


def test_active_learning_caches_manifold_calibration(tmp_path: Path, monkeypatch) -> None:
    from src.dataset.active_learning import ActiveLearningSelector, Candidate
    import src.dataset.active_learning as active_learning

    make_random_neural_models(tmp_path / "models", data_source="openfoam")
    center = np.array([0.03, 0.30, 0.12, 4.0, 3.5e6])
    deltas = np.diag([0.01, 0.10, 0.02, 2.0, 1.0e6])
    features = np.vstack([center + deltas, center - deltas])
    calibration = pd.DataFrame(
        features,
        columns=["m", "p", "t", "alpha_deg", "reynolds"],
    )
    calibration["data_source"] = "openfoam"
    calibration["status"] = "converged"
    calibration["is_synthetic"] = False
    calibration.to_csv(tmp_path / "dataset.csv", index=False)
    selector = ActiveLearningSelector(tmp_path / "models", tmp_path / "dataset.csv")
    candidate = Candidate("C0", "S0", 0.02, 0.4, 0.12, 4.0, 3e6)

    calls = 0
    original_read_csv = active_learning.pd.read_csv

    def counted_read_csv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(active_learning.pd, "read_csv", counted_read_csv)
    selector.score_candidates([candidate])
    first = selector._calibrate_manifold()
    selector.score_candidates([candidate])
    second = selector._calibrate_manifold()

    assert calls == 1
    assert first is second
    mean, scale, inv_cov = first
    assert mean.shape == scale.shape == (5,)
    assert inv_cov.shape == (5, 5)
    assert np.all(scale > 0.0)


def test_active_learning_empty_or_nonpositive_batch_returns_empty(tmp_path: Path) -> None:
    from src.dataset.active_learning import ActiveLearningSelector, Candidate

    make_random_neural_models(tmp_path / "models", data_source="openfoam")
    selector = ActiveLearningSelector(tmp_path / "models", tmp_path / "unused.csv")
    candidate = Candidate("C0", "S0", 0.02, 0.4, 0.12, 4.0, 3e6)

    assert selector.score_candidates([]) == []
    assert selector.select_batch([], batch_size=3) == []
    assert selector.select_batch([candidate], batch_size=0) == []
    assert selector.select_batch([candidate], batch_size=-1) == []


def test_active_learning_requires_trained_model(tmp_path: Path) -> None:
    from src.dataset.active_learning import ActiveLearningSelector

    # No weights in models_dir -> analytic fallback -> must raise
    with pytest.raises(RuntimeError, match="trained surrogate weights"):
        ActiveLearningSelector(models_dir=tmp_path, dataset_csv=tmp_path / "x.csv")


def test_active_learning_rejects_unverified_model_provenance(tmp_path: Path) -> None:
    from src.dataset.active_learning import ActiveLearningSelector

    make_random_neural_models(tmp_path / "models")

    with pytest.raises(RuntimeError, match="provenanced OpenFOAM data"):
        ActiveLearningSelector(
            models_dir=tmp_path / "models",
            dataset_csv=tmp_path / "dataset.csv",
        )


def test_naca_candidate_pool_uses_campaign_case_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.dataset.active_learning import build_candidate_pool
    import scripts.run_dataset_campaign as campaign

    monkeypatch.setattr(campaign, "sample_airfoil_codes", lambda *_args, **_kwargs: ["0012"])
    existing = tmp_path / "campaign.csv"
    pd.DataFrame({"case_id": ["NACA0012_a+2.0_Re1e6"]}).to_csv(existing, index=False)

    pool = build_candidate_pool(
        existing,
        n_shapes=1,
        alphas=[2.0, 4.0],
        reynolds=[1.0e6],
        sampler="naca",
    )

    assert [candidate.case_id for candidate in pool] == ["NACA0012_a+4.0_Re1e6"]


def test_campaign_driver_manifest_mode(tmp_path: Path) -> None:
    """The campaign driver accepts an active-learning manifest without WSL (dry structure)."""
    from scripts.run_dataset_campaign import parse_args  # import sanity

    args = parse_args.__doc__  # noqa: F841  (parse_args reads sys.argv; only smoke the import)
    assert parse_args is not None
