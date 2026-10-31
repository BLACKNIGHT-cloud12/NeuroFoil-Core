"""Shared test fixtures/helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.inference import DEFAULT_FEATURE_ORDER, TARGET_ORDER, WEIGHTS_FILE, META_FILE


def make_random_neural_models(models_dir: Path, seed: int = 0) -> None:
    """
    Write architecturally-correct random surrogate weights + scalars + conformal
    meta into models_dir, so AeroSurrogateEngine loads with backend="neural".
    """
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
    (models_dir / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
