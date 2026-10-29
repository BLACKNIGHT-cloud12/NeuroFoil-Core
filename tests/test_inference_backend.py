"""
Tests for the trained-weights NumPy inference backend, conformal UQ, and
dataset-campaign helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    DEFAULT_FEATURE_ORDER,
    TARGET_ORDER,
    WEIGHTS_FILE,
    META_FILE,
    AeroSurrogateEngine,
)
from src.models.uncertainty import AerodynamicUncertaintyGuard


def _random_weights(models_dir: Path, seed: int = 0) -> None:
    """Export architecturally-correct random weights so the numpy runtime can load them."""
    rng = np.random.default_rng(seed)
    in_dim = len(DEFAULT_FEATURE_ORDER)  # 17
    w = {
        "s_in_w": rng.normal(0, 0.02, (128, in_dim)), "s_in_b": rng.normal(0, 0.02, 128),
        "s_in_bn_m": rng.normal(0, 0.1, 128), "s_in_bn_v": rng.uniform(0.9, 1.1, 128),
        "s_in_bn_g": rng.uniform(0.9, 1.1, 128), "s_in_bn_b": rng.normal(0, 0.02, 128),
    }
    for r in (1, 2):
        w.update({
            f"s_r{r}_fc_w": rng.normal(0, 0.05, (128, 128)), f"s_r{r}_fc_b": rng.normal(0, 0.02, 128),
            f"s_r{r}_bn1_m": rng.normal(0, 0.1, 128), f"s_r{r}_bn1_v": rng.uniform(0.9, 1.1, 128),
            f"s_r{r}_bn1_g": rng.uniform(0.9, 1.1, 128), f"s_r{r}_bn1_b": rng.normal(0, 0.02, 128),
            f"s_r{r}_fc2_w": rng.normal(0, 0.05, (128, 128)), f"s_r{r}_fc2_b": rng.normal(0, 0.02, 128),
            f"s_r{r}_bn2_m": rng.normal(0, 0.1, 128), f"s_r{r}_bn2_v": rng.uniform(0.9, 1.1, 128),
            f"s_r{r}_bn2_g": rng.uniform(0.9, 1.1, 128), f"s_r{r}_bn2_b": rng.normal(0, 0.02, 128),
        })
    w.update({
        "s_h1_w": rng.normal(0, 0.05, (64, 128)), "s_h1_b": rng.normal(0, 0.02, 64),
        "s_out_w": rng.normal(0, 0.05, (3, 64)), "s_out_b": rng.normal(0, 0.02, 3),
        "f_l1_w": rng.normal(0, 0.02, (128, in_dim)), "f_l1_b": rng.normal(0, 0.02, 128),
        "f_l2_w": rng.normal(0, 0.02, (256, 128)), "f_l2_b": rng.normal(0, 0.02, 256),
        "f_l3_w": rng.normal(0, 0.02, (512, 256)), "f_l3_b": rng.normal(0, 0.02, 512),
        "f_l4_w": rng.normal(0, 0.02, (322, 512)), "f_l4_b": rng.normal(0, 0.02, 322),
    })
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(models_dir / WEIGHTS_FILE, **w)

    np.save(models_dir / "scaler_x_mean.npy", np.zeros(in_dim))
    np.save(models_dir / "scaler_x_scale.npy", np.ones(in_dim))
    np.save(models_dir / "scaler_y_mean.npy", np.zeros(3))
    np.save(models_dir / "scaler_y_scale.npy", np.ones(3))

    meta = {
        "feature_order": DEFAULT_FEATURE_ORDER,
        "target_order": TARGET_ORDER,
        "conformal": {
            "method": "split_conformal_abs_residual_95pct",
            "calibration_n": 40,
            "q_cl": 0.05, "q_cd": 0.002, "q_cm": 0.01,
            "q_cp_upper": [0.1] * 161, "q_cp_lower": [0.08] * 161,
        },
    }
    (models_dir / META_FILE).write_text(json.dumps(meta))


def test_feature_order_shape() -> None:
    assert len(DEFAULT_FEATURE_ORDER) == 17
    assert TARGET_ORDER == ["cl", "cd", "cm"]


def test_neural_backend_end_to_end(tmp_path: Path) -> None:
    _random_weights(tmp_path)
    engine = AeroSurrogateEngine(models_dir=tmp_path)
    assert engine.backend == "neural"

    result = engine.predict(m=0.02, p=0.4, t=0.12, alpha_deg=4.0)
    assert result["model_backend"] == "neural"
    for key in ("cl", "cd", "cm", "lift_to_drag", "cp_upper", "cp_lower", "delta_cp", "uq", "wingbox"):
        assert key in result
    assert result["cp_upper"].shape == (161,)
    assert result["cp_lower"].shape == (161,)
    assert np.all(np.isfinite(result["cp_upper"]))
    assert result["wingbox"] is not None

    # Conformal intervals from the meta file must surface in the UQ report
    assert abs(result["uq"].cl_ci95[0] - (result["cl"] - 0.05)) < 1e-9
    assert abs(result["uq"].cl_ci95[1] - (result["cl"] + 0.05)) < 1e-9


def test_numpy_forward_matches_torch(tmp_path: Path) -> None:
    """The pure-NumPy forward pass must equal the PyTorch model it was exported from."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    _random_weights(tmp_path)
    engine = AeroSurrogateEngine(models_dir=tmp_path)

    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
                nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class MLP(nn.Module):
        def __init__(self, i, o=3):
            super().__init__()
            self.input_layer = nn.Sequential(nn.Linear(i, 128), nn.BatchNorm1d(128), nn.GELU())
            self.res1 = ResidualBlock(128)
            self.res2 = ResidualBlock(128)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, o))

        def forward(self, x):
            return self.head(self.res2(self.res1(self.input_layer(x))))

    W = engine.weights
    sd = {
        "input_layer.0.weight": W["s_in_w"], "input_layer.0.bias": W["s_in_b"],
        "input_layer.1.running_mean": W["s_in_bn_m"], "input_layer.1.running_var": W["s_in_bn_v"],
        "input_layer.1.weight": W["s_in_bn_g"], "input_layer.1.bias": W["s_in_bn_b"],
    }
    for r in (1, 2):
        p = f"res{r}.net."
        sd.update({
            f"{p}0.weight": W[f"s_r{r}_fc_w"], f"{p}0.bias": W[f"s_r{r}_fc_b"],
            f"{p}1.running_mean": W[f"s_r{r}_bn1_m"], f"{p}1.running_var": W[f"s_r{r}_bn1_v"],
            f"{p}1.weight": W[f"s_r{r}_bn1_g"], f"{p}1.bias": W[f"s_r{r}_bn1_b"],
            f"{p}3.weight": W[f"s_r{r}_fc2_w"], f"{p}3.bias": W[f"s_r{r}_fc2_b"],
            f"{p}4.running_mean": W[f"s_r{r}_bn2_m"], f"{p}4.running_var": W[f"s_r{r}_bn2_v"],
            f"{p}4.weight": W[f"s_r{r}_bn2_g"], f"{p}4.bias": W[f"s_r{r}_bn2_b"],
        })
    sd.update({
        "head.0.weight": W["s_h1_w"], "head.0.bias": W["s_h1_b"],
        "head.2.weight": W["s_out_w"], "head.2.bias": W["s_out_b"],
    })

    model = MLP(len(DEFAULT_FEATURE_ORDER))
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in sd.items()})
    model.eval()

    from src.geometry.naca import generate_naca4_airfoil

    feats = engine._build_features(generate_naca4_airfoil("2412", 161), 4.0, 6.0e6)
    x = (feats - engine.x_mean) / engine.x_scale

    with torch.no_grad():
        y_torch = model(torch.tensor(x, dtype=torch.float32)).numpy()
    y_numpy = engine._forward_scalars(x)

    # numpy runs float64, torch float32: relative agreement is the invariant
    np.testing.assert_allclose(y_numpy, y_torch, rtol=1e-5, atol=1e-4)


def test_conformal_guard_intervals() -> None:
    guard = AerodynamicUncertaintyGuard(conformal={"q_cl": 0.05, "q_cd": 0.002,
                                                   "q_cp_upper": [0.1] * 161,
                                                   "q_cp_lower": [0.08] * 161})
    beta = np.linspace(0, np.pi, 161)
    x_grid = 0.5 * (1.0 - np.cos(beta))
    cp_u, cp_l = -0.5 * np.ones(161), 0.2 * np.ones(161)

    report = guard.evaluate(m=0.02, p=0.4, t=0.12, alpha_deg=4.0,
                            cl=0.5, cd=0.01, l_over_d=50.0,
                            x_grid=x_grid, cp_upper=cp_u, cp_lower=cp_l,
                            reynolds=6.0e6)
    assert abs(report.cl_ci95[0] - 0.45) < 1e-9
    assert abs(report.cl_ci95[1] - 0.55) < 1e-9
    assert abs(report.cd_ci95[0] - 0.008) < 1e-9
    np.testing.assert_allclose(report.cp_upper_ci95_low, cp_u - 0.1)
    np.testing.assert_allclose(report.cp_lower_ci95_high, cp_l + 0.08)


def test_reynolds_envelope_flagged() -> None:
    guard = AerodynamicUncertaintyGuard()
    beta = np.linspace(0, np.pi, 161)
    x_grid = 0.5 * (1.0 - np.cos(beta))
    report = guard.evaluate(m=0.02, p=0.4, t=0.12, alpha_deg=4.0,
                            cl=0.5, cd=0.01, l_over_d=50.0,
                            x_grid=x_grid, cp_upper=-0.5 * np.ones(161),
                            cp_lower=0.2 * np.ones(161),
                            reynolds=20.0e6)
    assert report.is_ood
    assert any("Reynolds" in w for w in report.warning_messages)


def test_campaign_helpers() -> None:
    from scripts.run_dataset_campaign import sample_airfoil_codes

    codes = sample_airfoil_codes(21, seed=42)
    assert codes[:3] == ["0012", "2412", "4412"]  # anchors first
    assert len(codes) == len(set(codes))          # unique
    assert len(codes) == 21
    for c in codes:
        assert len(c) == 4 and c.isdigit()

    # case_id determinism / uniqueness across the campaign grid
    ids = set()
    for c in codes:
        for re_ in (1e6, 2e6, 3e6, 6e6):
            for a in (-6.0, -2.0, 2.0, 6.0, 10.0, 14.0):
                cid = f"NACA{c}_a{a:+.1f}_Re{re_:.0e}".replace("e+0", "e").replace("e+", "e")
                ids.add(cid)
    assert len(ids) == 21 * 4 * 6  # 504 unique case ids
