from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_xfoil_baseline as baseline


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            "# alpha CL CD CDp CM Top_Xtr Bot_Xtr\n",
            ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"],
        ),
        (
            "  alpha     CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n",
            ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"],
        ),
    ],
)
def test_polar_column_names_detects_commented_and_plain_headers(
    tmp_path: Path, header: str, expected: list[str],
) -> None:
    polar = tmp_path / "polar.txt"
    polar.write_text(
        "XFOIL Version 6.99\n" + header + " ------ ------ ------ ------ ------ ------ ------\n"
        "  2.000  0.2200  0.00850  0.00310 -0.0200  0.9100  0.8200\n",
        encoding="utf-8",
    )

    assert baseline._polar_column_names(polar) == expected
    frame = baseline._polar_frame(polar)
    assert frame.columns.tolist() == expected
    np.testing.assert_allclose(
        frame.iloc[0].to_numpy(),
        [2.0, 0.22, 0.0085, 0.0031, -0.02, 0.91, 0.82],
    )


def test_polar_frame_uses_documented_fallback_order_without_header(tmp_path: Path) -> None:
    polar = tmp_path / "polar.txt"
    polar.write_text(
        "  -2.000 -0.2200 0.00910 0.00390 -0.0100 0.8200 0.7100\n"
        "   2.000  0.2200 0.00850 0.00310 -0.0200 0.9100 0.8200\n",
        encoding="utf-8",
    )

    frame = baseline._polar_frame(polar)
    assert frame.columns.tolist() == baseline._FALLBACK_POLAR_COLUMNS
    assert frame.iloc[0].to_dict() == {
        "alpha": -2.0,
        "cl": -0.22,
        "cd": 0.0091,
        "cdp": 0.0039,
        "cm": -0.01,
        "top_xtr": 0.82,
        "bot_xtr": 0.71,
    }


def test_polar_frame_truncates_rows_and_header_to_consistent_width(tmp_path: Path) -> None:
    polar = tmp_path / "polar.txt"
    polar.write_text(
        "alpha CL CD CDp CM Top_Xtr Bot_Xtr Cpmin\n"
        "0.0 0.1 0.01 0.004 -0.02 0.8 0.7\n"
        "1.0 0.2 0.02 0.005 -0.03 0.9 0.8 -1.4\n",
        encoding="utf-8",
    )

    frame = baseline._polar_frame(polar)
    assert frame.columns.tolist() == ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]
    assert frame.shape == (2, 7)


def test_exec_backend_closes_script_stdin_and_parses_plain_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(baseline.shutil, "which", lambda _: "/usr/bin/xfoil")
    monkeypatch.setattr(baseline.tempfile, "TemporaryDirectory", lambda: _TemporaryDirectory(tmp_path))
    observed: dict[str, object] = {}

    def fake_run(command, *, stdin, **kwargs):
        observed["stdin"] = stdin
        polar_path = tmp_path / "polar.txt"
        polar_path.write_text(
            "alpha CL CD CDp CM Top_Xtr Bot_Xtr\n"
            "0.0 0.1 0.01 0.004 -0.02 0.8 0.7\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(baseline.subprocess, "run", fake_run)
    dat_path = tmp_path / "airfoil.dat"
    dat_path.write_text("foil\n", encoding="utf-8")

    frame = baseline._run_xfoil_exec(dat_path, [0.0], 1.0e6, 0.0, 100)

    assert observed["stdin"].closed
    assert frame.columns.tolist() == baseline._FALLBACK_POLAR_COLUMNS


def test_compiled_backend_uses_airfoil_object_and_canonical_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAirfoil:
        def __init__(self, x, y):
            self.x = np.asarray(x)
            self.y = np.asarray(y)

    class FakeXFoil:
        latest = None

        def __init__(self):
            self.airfoil = None
            FakeXFoil.latest = self

        def a(self, alpha):
            return 0.42, 0.009, -0.03, -1.2

    xfoil_module = ModuleType("xfoil")
    xfoil_module.XFoil = FakeXFoil
    xfoil_model = ModuleType("xfoil.model")
    xfoil_model.Airfoil = FakeAirfoil
    monkeypatch.setitem(sys.modules, "xfoil", xfoil_module)
    monkeypatch.setitem(sys.modules, "xfoil.model", xfoil_model)

    geom = SimpleNamespace(
        to_closed_contour=lambda: np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    )
    frame = baseline.run_case(geom, [2.0], 1.0e6, 0.0, 100)

    assert isinstance(FakeXFoil.latest.airfoil, FakeAirfoil)
    assert frame.columns.tolist() == ["alpha", "cl", "cd", "cm"]
    assert frame.iloc[0].to_dict() == {
        "alpha": 2.0,
        "cl": 0.42,
        "cd": 0.009,
        "cm": -0.03,
    }


def test_from_dataset_reproduces_campaign_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame([
        {"naca_code": "0012", "reynolds": 1.0e6, "alpha_deg": 0.0},
        {"naca_code": "0012", "reynolds": 6.0e6, "alpha_deg": 2.0},
        {"naca_code": "0012", "reynolds": 6.0e6, "alpha_deg": 4.0},
        {
            "naca_code": "CST-abc", "reynolds": 2.0e6, "alpha_deg": 6.0,
            **{f"cst_u_{i}": 0.1 for i in range(6)},
            **{f"cst_l_{i}": 0.1 for i in range(6)},
        },
        {
            "naca_code": "CST-abc", "reynolds": 2.0e6, "alpha_deg": 8.0,
            **{f"cst_u_{i}": np.nan for i in range(6)},
            **{f"cst_l_{i}": np.nan for i in range(6)},
        },
    ]).to_csv(processed / "airfoil_cfd_dataset.csv", index=False)

    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(baseline, "OUT_CSV", tmp_path / "out" / "xfoil_baseline.csv")
    monkeypatch.setattr(
        baseline, "parse_args",
        lambda: SimpleNamespace(
            from_dataset=True, codes="", alphas="0.0", re=6.0e6, mach=0.0, itrmax=100,
        ),
    )
    calls: list[tuple[str, float, list[float]]] = []

    def fake_run_case(geom, alphas_item, re_value, mach, itrmax):
        calls.append((geom.code, re_value, list(alphas_item)))
        return pd.DataFrame(
            [{"alpha": a, "cl": 1.0, "cd": 0.01, "cm": 0.0} for a in alphas_item]
        )

    monkeypatch.setattr(baseline, "run_case", fake_run_case)
    monkeypatch.setattr(
        baseline, "geometry_from_cst",
        lambda upper, lower: SimpleNamespace(code="CST-abc"),
    )
    monkeypatch.setattr(
        baseline, "generate_naca4_airfoil",
        lambda code, *args, **kwargs: SimpleNamespace(code=code),
    )

    baseline.main()

    assert ("0012", 1.0e6, [0.0]) in calls
    assert ("0012", 6.0e6, [2.0, 4.0]) in calls
    assert ("CST-abc", 2.0e6, [6.0, 8.0]) in calls
    assert len(calls) == 3


class _TemporaryDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *args) -> None:
        return None
