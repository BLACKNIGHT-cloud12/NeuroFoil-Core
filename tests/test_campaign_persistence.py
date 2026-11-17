from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_dataset_campaign as campaign


def _patch_store_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "processed" / "campaign.csv"
    npz_path = tmp_path / "processed" / "cp.npz"
    monkeypatch.setattr(campaign, "CSV_PATH", csv_path)
    monkeypatch.setattr(campaign, "NPZ_PATH", npz_path)
    monkeypatch.setattr(campaign, "RAW_DIR", tmp_path / "raw")
    return csv_path, npz_path


def _cp(value: float) -> dict[str, np.ndarray]:
    return {
        "cp_upper": np.array([value, value + 1], dtype=np.float32),
        "cp_lower": np.array([value + 2, value + 3], dtype=np.float32),
        "delta_cp": np.array([2.0, 2.0], dtype=np.float32),
    }


def test_store_add_includes_current_cp_and_atomically_replaces_npz(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, npz_path = _patch_store_paths(monkeypatch, tmp_path)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def recording_replace(source: Path, target: Path) -> Path:
        replace_calls.append((source, Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    store = campaign.Store(pd.DataFrame(), None, {})
    store.add(
        {"case_id": "case-a", "success": True, "cp_available": True},
        _cp(1.0),
        np.array([0.0, 1.0], dtype=np.float32),
    )

    assert csv_path.exists()
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as saved:
        assert saved["case_ids"].tolist() == ["case-a"]
        np.testing.assert_array_equal(saved["cp_upper"][0], [1.0, 2.0])

    npz_replaces = [(source, target) for source, target in replace_calls if target == npz_path]
    assert npz_replaces == [(npz_path.parent / (npz_path.name + ".tmp.npz"), npz_path)]
    assert replace_calls.index(npz_replaces[0]) < next(
        index for index, (_, target) in enumerate(replace_calls) if target == csv_path
    )
    assert not (npz_path.parent / (npz_path.name + ".tmp.npz")).exists()


def test_store_retry_replaces_row_and_cp_instead_of_duplicating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, npz_path = _patch_store_paths(monkeypatch, tmp_path)
    existing = pd.DataFrame(
        [
            {"case_id": "case-a", "success": False, "status": "old"},
            {"case_id": "case-b", "success": True, "status": "keep"},
            {"case_id": "case-a", "success": False, "status": "duplicate"},
        ]
    )
    store = campaign.Store(existing, np.array([0.0, 1.0]), {"case-a": _cp(0.0)})

    store.add(
        {"case_id": "case-a", "success": True, "status": "new", "cp_available": True},
        _cp(9.0),
        np.array([0.0, 1.0]),
    )

    saved_csv = pd.read_csv(csv_path)
    assert saved_csv["case_id"].tolist() == ["case-a", "case-b"]
    assert saved_csv.loc[saved_csv["case_id"] == "case-a", "status"].item() == "new"
    with np.load(npz_path, allow_pickle=False) as saved_npz:
        assert saved_npz["case_ids"].tolist().count("case-a") == 1
        np.testing.assert_array_equal(saved_npz["cp_upper"][0], [9.0, 10.0])


def test_store_retry_without_cp_removes_stale_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, npz_path = _patch_store_paths(monkeypatch, tmp_path)
    existing = pd.DataFrame([
        {
            "case_id": "case-a",
            "success": True,
            "status": "converged",
            "cp_available": True,
        }
    ])
    store = campaign.Store(
        existing,
        np.array([0.0, 1.0], dtype=np.float32),
        {"case-a": _cp(1.0)},
    )

    store.add(
        {
            "case_id": "case-a",
            "success": False,
            "status": "solver_failed",
            "cp_available": False,
        },
        None,
        None,
    )

    assert "case-a" not in store.cp_store
    saved_csv = pd.read_csv(csv_path)
    assert not bool(saved_csv.loc[0, "cp_available"])
    with np.load(npz_path, allow_pickle=False) as saved_npz:
        assert saved_npz["case_ids"].size == 0
        assert saved_npz["cp_upper"].shape == (0, 2)
        assert saved_npz["cp_lower"].shape == (0, 2)
        assert saved_npz["delta_cp"].shape == (0, 2)


def test_store_csv_failure_rolls_back_cp_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, npz_path = _patch_store_paths(monkeypatch, tmp_path)
    csv_path.parent.mkdir(parents=True)
    csv_path.mkdir()  # CSV replacement target is a directory -> persistence fails
    existing = pd.DataFrame([
        {
            "case_id": "case-a",
            "success": True,
            "status": "converged",
            "cp_available": True,
        }
    ])
    store = campaign.Store(
        existing,
        np.array([0.0, 1.0], dtype=np.float32),
        {"case-a": _cp(1.0)},
    )

    with pytest.raises(OSError):
        store.add(
            {
                "case_id": "case-a",
                "success": False,
                "status": "solver_failed",
                "cp_available": False,
            },
            None,
            None,
        )

    assert store.cp_store["case-a"]["cp_upper"].tolist() == [1.0, 2.0]
    assert bool(store.df.loc[store.df["case_id"] == "case-a", "cp_available"].item())
    with np.load(npz_path, allow_pickle=False) as saved_npz:
        assert saved_npz["case_ids"].tolist() == ["case-a"]
        np.testing.assert_array_equal(saved_npz["cp_upper"][0], [1.0, 2.0])


def test_store_validates_cp_grid_before_mutating_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_store_paths(monkeypatch, tmp_path)
    existing = pd.DataFrame([{"case_id": "case-a", "status": "old"}])
    store = campaign.Store(existing, None, {})

    with pytest.raises(ValueError, match="x_grid is required"):
        store.add(
            {"case_id": "case-a", "status": "new", "cp_available": True},
            _cp(1.0),
            None,
        )

    assert store.df.equals(existing)
    assert store.cp_store == {}


def test_completed_case_ids_reconciles_csv_booleans_and_npz() -> None:
    existing = pd.DataFrame(
        [
            {"case_id": "cp-ok", "success": "True", "cp_available": "true"},
            {"case_id": "cp-missing", "success": "TRUE", "cp_available": "1"},
            {"case_id": "scalar", "success": "yes", "cp_available": "False"},
            {"case_id": "failed", "success": "False", "cp_available": "false"},
            {"case_id": "failed-number", "success": "0", "cp_available": "0"},
            {"case_id": "missing-success", "success": np.nan, "cp_available": "false"},
            {"case_id": "unknown", "success": "not-a-bool", "cp_available": "false"},
            {"case_id": "retried", "success": "True", "cp_available": "false"},
            {"case_id": "retried", "success": "False", "cp_available": "false"},
        ]
    )

    assert campaign.completed_case_ids(existing, {"cp-ok", "npz-extra"}) == {"cp-ok", "scalar"}


def test_pending_jobs_applies_limit_after_completed_cases_are_removed() -> None:
    class Job:
        def __init__(self, case_id: str):
            self.case_id = case_id

    jobs = [Job("done-1"), Job("pending-1"), Job("done-2"), Job("pending-2"), Job("pending-3")]
    selected = campaign.pending_jobs(jobs, {"done-1", "done-2"}, limit=2)
    assert [job.case_id for job in selected] == ["pending-1", "pending-2"]


def _main_args(limit: int = 0) -> Namespace:
    return Namespace(
        sampler="naca",
        airfoils=3,
        alphas="0",
        res="1e6",
        mesh_level="medium",
        timeout=10,
        jobs=1,
        limit=limit,
        seed=42,
        manifest="",
    )


def test_main_filters_pending_before_applying_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, _ = _patch_store_paths(monkeypatch, tmp_path)
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"case_id": "NACA0012_a+0.0_Re1e6", "success": True, "cp_available": False, "status": "done"}]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(campaign, "parse_args", lambda: _main_args(limit=1))
    monkeypatch.setattr(campaign, "load_config", lambda: object())
    monkeypatch.setattr(campaign, "sample_airfoil_codes", lambda *_: ["0012", "2412", "4412"])
    monkeypatch.setattr(campaign, "cleanup_case_dir", lambda _: None)
    attempted: list[str] = []

    def fake_run(sample, *args, **kwargs):
        attempted.append(sample.case_id)
        return {"success": False, "status": "expected-test-failure"}

    monkeypatch.setattr(campaign, "run_single_cfd_sample", fake_run)

    campaign.main()

    assert attempted == ["NACA2412_a+0.0_Re1e6"]


@pytest.mark.parametrize("success", [True, False])
def test_main_adds_openfoam_provenance_to_success_and_failure_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, success: bool,
) -> None:
    csv_path, _ = _patch_store_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(campaign, "parse_args", lambda: _main_args())
    monkeypatch.setattr(campaign, "load_config", lambda: object())
    monkeypatch.setattr(campaign, "sample_airfoil_codes", lambda *_: ["0012"])
    monkeypatch.setattr(campaign, "cleanup_case_dir", lambda _: None)

    if success:
        result = {
            "success": True,
            "record": {
                "case_id": "NACA0012_a+0.0_Re1e6",
                "naca_code": "0012",
                "success": True,
                "status": "converged",
                "cp_available": False,
            },
            "cp_data": {"available": False},
        }
    else:
        result = {"success": False, "status": "solver_failed"}
    monkeypatch.setattr(campaign, "run_single_cfd_sample", lambda *args, **kwargs: result)

    campaign.main()

    row = pd.read_csv(csv_path).iloc[0]
    assert row["data_source"] == "openfoam"
    assert not bool(row["is_synthetic"])
    assert row["solver_backend"] == "simpleFoam"
    assert row["mesh_level"] == "medium"


def test_main_preserves_existing_provenance_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    csv_path, _ = _patch_store_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(campaign, "parse_args", lambda: _main_args())
    monkeypatch.setattr(campaign, "load_config", lambda: object())
    monkeypatch.setattr(campaign, "sample_airfoil_codes", lambda *_: ["0012"])
    monkeypatch.setattr(campaign, "cleanup_case_dir", lambda _: None)
    monkeypatch.setattr(
        campaign,
        "run_single_cfd_sample",
        lambda *args, **kwargs: {
            "success": True,
            "record": {
                "case_id": "NACA0012_a+0.0_Re1e6",
                "naca_code": "0012",
                "success": True,
                "status": "converged",
                "cp_available": False,
                "data_source": "custom-source",
                "is_synthetic": True,
                "solver_backend": "custom-solver",
                "mesh_level": "custom-mesh",
            },
            "cp_data": {"available": False},
        },
    )

    campaign.main()

    row = pd.read_csv(csv_path).iloc[0]
    assert row["data_source"] == "custom-source"
    assert bool(row["is_synthetic"])
    assert row["solver_backend"] == "custom-solver"
    assert row["mesh_level"] == "custom-mesh"
