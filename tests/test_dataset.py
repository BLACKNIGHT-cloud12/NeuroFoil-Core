"""
Unit tests for Phase III dataset sampler and pipeline utilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.dataset.sampler import (
    AirfoilSample,
    generate_canonical_benchmark_cases,
    naca_code_from_params,
    sample_latin_hypercube,
)
from src.dataset.pipeline import generate_physics_augmented_dataset, run_single_cfd_sample


def test_naca_code_from_params() -> None:
    assert naca_code_from_params(0.0, 0.0, 0.12) == "0012"
    assert naca_code_from_params(0.02, 0.4, 0.12) == "2412"
    assert naca_code_from_params(0.04, 0.4, 0.15) == "4415"


def test_sample_latin_hypercube() -> None:
    samples = sample_latin_hypercube(n_samples=10, seed=42)
    assert len(samples) == 10
    assert len({s.case_id for s in samples}) == 10
    for s in samples:
        assert isinstance(s, AirfoilSample)
        assert -4.0 <= s.alpha_deg <= 12.0
        assert 0.06 <= s.t <= 0.18
        assert 1.0e6 <= s.reynolds <= 6.0e6

    decoded_rows = []
    for sample in samples:
        m_decoded = int(sample.naca_code[0]) / 100.0
        p_decoded = int(sample.naca_code[1]) / 10.0
        if m_decoded == 0.0 or p_decoded == 0.0:
            m_decoded = p_decoded = 0.0
        decoded_rows.append((m_decoded, p_decoded, int(sample.naca_code[2:4]) / 100.0))
    decoded = np.array(decoded_rows)
    raw = np.array([(s.m, s.p, s.t) for s in samples])
    assert not np.allclose(raw, decoded)


def test_latin_hypercube_reynolds_sampling_is_bounded_and_deterministic() -> None:
    first = sample_latin_hypercube(n_samples=64, seed=17)
    second = sample_latin_hypercube(n_samples=64, seed=17)

    re_first = np.array([sample.reynolds for sample in first])
    re_second = np.array([sample.reynolds for sample in second])
    np.testing.assert_allclose(re_first, re_second, rtol=0.0, atol=0.0)
    assert np.all((re_first >= 1.0e6) & (re_first <= 6.0e6))
    assert re_first.min() < 1.2e6
    assert re_first.max() > 5.0e6


def test_latin_hypercube_fixed_reynolds_compatibility() -> None:
    samples = sample_latin_hypercube(n_samples=8, reynolds=3.0e6, seed=9)
    repeated = sample_latin_hypercube(n_samples=8, reynolds=3.0e6, seed=9)
    assert all(sample.reynolds == 3.0e6 for sample in samples)
    assert [sample.case_id for sample in samples] == [sample.case_id for sample in repeated]
    np.testing.assert_allclose(
        [(sample.m, sample.p, sample.t, sample.alpha_deg, sample.reynolds) for sample in samples],
        [(sample.m, sample.p, sample.t, sample.alpha_deg, sample.reynolds) for sample in repeated],
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="n_samples"):
        sample_latin_hypercube(n_samples=0)
    with pytest.raises(ValueError, match="Reynolds bounds"):
        sample_latin_hypercube(n_samples=2, reynolds_min=6.0e6, reynolds_max=1.0e6)


def test_pipeline_passes_raw_naca_parameters_to_geometry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample = AirfoilSample(
        case_id="continuous_case",
        naca_code="2413",
        m=0.0237,
        p=0.413,
        t=0.1275,
        alpha_deg=3.0,
        reynolds=2.5e6,
    )
    captured: dict[str, object] = {}

    class DummyAirfoil:
        def save_dat(self, path: Path) -> Path:
            path.write_text("dummy", encoding="utf-8")
            return path

        def save_metadata(self, path: Path) -> Path:
            path.write_text("{}", encoding="utf-8")
            return path

    dummy_airfoil = DummyAirfoil()

    def fake_generate(code: str, *args: object, **kwargs: object) -> DummyAirfoil:
        captured["code"] = code
        captured.update(kwargs)
        return dummy_airfoil

    class FailedMesh:
        mesh_ok = False
        n_cells = 0
        max_non_orthogonality_deg = 0.0
        max_skewness = 0.0

    def fake_mesh_dict(**kwargs: object) -> object:
        assert kwargs["airfoil"] is dummy_airfoil
        return object()

    monkeypatch.setattr("src.dataset.pipeline.generate_naca4_airfoil", fake_generate)
    monkeypatch.setattr("src.dataset.pipeline.generate_blockmesh_dict", fake_mesh_dict)
    monkeypatch.setattr("src.dataset.pipeline.run_blockmesh_and_checkmesh", lambda **kwargs: FailedMesh())

    cfg = {
        "mesh": {
            "levels": {"coarse": {"n_airfoil_side": 20, "n_wake": 10, "n_normal": 10}},
            "farfield_radius_chords": 20.0,
            "wake_length_chords": 20.0,
            "span_thickness_m": 0.01,
        },
        "wsl_environment": {"distro": "unused", "openfoam_bashrc": "unused"},
    }
    result = run_single_cfd_sample(sample, tmp_path / "case", cfg)

    assert result["status"] == "mesh_failed"
    assert result["data_source"] == "openfoam"
    assert result["is_synthetic"] is False
    assert result["solver_backend"] == "simpleFoam"
    assert result["mesh_level"] == "coarse"
    assert captured["code"] == sample.naca_code
    assert captured["m"] == sample.m
    assert captured["p"] == sample.p
    assert captured["t"] == sample.t


def test_generate_canonical_benchmark_cases() -> None:
    cases = generate_canonical_benchmark_cases(codes=["0012", "2412"], alphas=[0.0, 4.0])
    assert len(cases) == 4
    assert cases[0].naca_code == "0012"


def test_generate_physics_augmented_dataset(tmp_path: Path) -> None:
    csv_file, npz_file = generate_physics_augmented_dataset(n_samples=20, output_dir=tmp_path)
    assert csv_file.name == "synthetic_airfoil_dataset.csv"
    assert npz_file.name == "synthetic_cp_surface_distributions.npz"
    assert csv_file.exists()
    assert npz_file.exists()

    df = pd.read_csv(csv_file)
    assert set(df["data_source"]) == {"synthetic_analytic"}
    assert bool(df["is_synthetic"].all())
    assert set(df["solver_backend"]) == {"analytic_placeholder"}

    npz = np.load(npz_file)
    assert npz["data_source"].item() == "synthetic_analytic"
    assert len(npz["x_grid"]) == 161
    assert npz["cp_upper"].shape[0] == 20
