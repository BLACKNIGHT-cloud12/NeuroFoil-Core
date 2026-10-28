"""
Unit tests for Phase III dataset sampler and pipeline utilities.
"""

from pathlib import Path
import numpy as np
import pytest

from src.dataset.sampler import (
    AirfoilSample,
    generate_canonical_benchmark_cases,
    naca_code_from_params,
    sample_latin_hypercube,
)
from src.dataset.pipeline import generate_physics_augmented_dataset


def test_naca_code_from_params() -> None:
    assert naca_code_from_params(0.0, 0.0, 0.12) == "0012"
    assert naca_code_from_params(0.02, 0.4, 0.12) == "2412"
    assert naca_code_from_params(0.04, 0.4, 0.15) == "4415"


def test_sample_latin_hypercube() -> None:
    samples = sample_latin_hypercube(n_samples=10, seed=42)
    assert len(samples) == 10
    for s in samples:
        assert isinstance(s, AirfoilSample)
        assert -4.0 <= s.alpha_deg <= 12.0
        assert 0.06 <= s.t <= 0.18


def test_generate_canonical_benchmark_cases() -> None:
    cases = generate_canonical_benchmark_cases(codes=["0012", "2412"], alphas=[0.0, 4.0])
    assert len(cases) == 4
    assert cases[0].naca_code == "0012"


def test_generate_physics_augmented_dataset(tmp_path: Path) -> None:
    csv_file, npz_file = generate_physics_augmented_dataset(n_samples=20, output_dir=tmp_path)
    assert csv_file.exists()
    assert npz_file.exists()

    npz = np.load(npz_file)
    assert len(npz["x_grid"]) == 161
    assert npz["cp_upper"].shape[0] == 20
