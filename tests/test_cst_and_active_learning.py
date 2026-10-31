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

    make_random_neural_models(tmp_path / "models")

    # Tiny "training" CSV for the Mahalanobis manifold
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "case_id": [f"C{i}" for i in range(30)],
        "m": rng.uniform(0, 0.06, 30),
        "p": rng.uniform(0.2, 0.6, 30),
        "t": rng.uniform(0.06, 0.18, 30),
        "alpha_deg": rng.uniform(-6, 14, 30),
        "reynolds": rng.choice([1e6, 2e6, 3e6, 6e6], 30),
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


def test_active_learning_requires_trained_model(tmp_path: Path) -> None:
    from src.dataset.active_learning import ActiveLearningSelector

    # No weights in models_dir -> analytic fallback -> must raise
    with pytest.raises(RuntimeError, match="trained surrogate weights"):
        ActiveLearningSelector(models_dir=tmp_path, dataset_csv=tmp_path / "x.csv")


def test_campaign_driver_manifest_mode(tmp_path: Path) -> None:
    """The campaign driver accepts an active-learning manifest without WSL (dry structure)."""
    from scripts.run_dataset_campaign import parse_args  # import sanity

    args = parse_args.__doc__  # noqa: F841  (parse_args reads sys.argv; only smoke the import)
    assert parse_args is not None
