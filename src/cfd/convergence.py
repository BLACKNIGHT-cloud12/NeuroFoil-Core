"""
Convergence Evaluator for OpenFOAM Aerodynamic Simulations (Milestone 1.3).

Parses force coefficient time histories (postProcessing/forceCoeffs) and solver
residuals (log.simpleFoam) to evaluate convergence state, statistical stability
(sliding-window standard deviation), and mean aerodynamic performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ConvergenceResult:
    """Aerodynamic convergence evaluation metrics."""

    status: str  # "converged", "unconverged", "diverged"
    iterations_completed: int
    cl_mean: float
    cl_std: float
    cd_mean: float
    cd_std: float
    cm_mean: float
    cm_std: float
    cl_history: tuple[float, ...] = field(default_factory=tuple)
    cd_history: tuple[float, ...] = field(default_factory=tuple)
    cm_history: tuple[float, ...] = field(default_factory=tuple)
    residuals: dict[str, tuple[float, ...]] = field(default_factory=dict)
    reasons: tuple[str, ...] = field(default_factory=tuple)


def parse_force_coeffs_file(dat_file: Path) -> dict[str, np.ndarray]:
    """Parse OpenFOAM postProcessing forceCoeffs/0/coefficient.dat file."""
    if not dat_file.exists():
        return {}

    lines = dat_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_col_map: dict[str, int] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # Clean header and look for column titles without stripping parenthesis
            header_line = stripped.lstrip("#").strip()
            tokens = header_line.split()
            tokens_lower = [t.lower() for t in tokens]
            if "time" in tokens_lower and ("cd" in tokens_lower or "cl" in tokens_lower):
                for idx, t in enumerate(tokens):
                    tl = t.lower()
                    if tl == "time" and "time" not in header_col_map:
                        header_col_map["time"] = idx
                    elif tl == "cd" and "cd" not in header_col_map:
                        header_col_map["cd"] = idx
                    elif tl == "cl" and "cl" not in header_col_map:
                        header_col_map["cl"] = idx
                    elif tl in ("cmpitch", "cm", "pitch") and "cm" not in header_col_map:
                        header_col_map["cm"] = idx

    data_rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            vals = [float(v) for v in stripped.split()]
            if len(vals) >= 4:
                data_rows.append(vals)
        except ValueError:
            continue

    if not data_rows:
        return {}

    arr = np.array(data_rows, dtype=float)

    # Defaults if headers were not explicitly identified
    t_idx = header_col_map.get("time", 0)
    cd_idx = header_col_map.get("cd", 1)
    cl_idx = header_col_map.get("cl", 4 if arr.shape[1] > 4 else 3)
    cm_idx = header_col_map.get("cm", 7 if arr.shape[1] > 7 else (5 if arr.shape[1] > 5 else 4))

    return {
        "time": arr[:, t_idx] if t_idx < arr.shape[1] else np.arange(len(arr)),
        "cd": arr[:, cd_idx] if cd_idx < arr.shape[1] else np.zeros(len(arr)),
        "cl": arr[:, cl_idx] if cl_idx < arr.shape[1] else np.zeros(len(arr)),
        "cm": arr[:, cm_idx] if cm_idx < arr.shape[1] else np.zeros(len(arr)),
    }


def parse_solver_log(log_file: Path) -> dict[str, Any]:
    """Parse log.simpleFoam for completed iterations and solver residuals."""
    if not log_file.exists():
        return {"iterations": 0, "diverged": False, "residuals": {}}

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    # Avoid false positive on startup message "Floating point exception trapping enabled"
    fatal_patterns = [
        r"FOAM FATAL",
        r"Floating point exception(?!\s+trapping)",
        r"\b(NaN|nan)\b",
    ]
    diverged = any(re.search(p, text) for p in fatal_patterns)

    # Extract iterations
    time_matches = re.findall(r"^Time = (\d+)", text, re.MULTILINE)
    iterations = int(time_matches[-1]) if time_matches else 0

    residuals: dict[str, list[float]] = {"p": [], "Ux": [], "Uy": [], "k": [], "omega": []}
    for match in re.finditer(
        r"Solving for (Ux|Uy|p|k|omega), Initial residual = ([0-9.eE+-]+)",
        text,
    ):
        var = match.group(1)
        res_val = float(match.group(2))
        residuals[var].append(res_val)

    return {
        "iterations": iterations,
        "diverged": diverged,
        "residuals": {k: np.array(v) for k, v in residuals.items() if v},
    }


def evaluate_convergence(
    case_dir: Path | str,
    window_size: int = 100,
    cl_std_tol: float = 5.0e-4,
    cd_std_tol: float = 2.0e-4,
    cm_std_tol: float = 5.0e-4,
    max_iterations: int = 1500,
) -> ConvergenceResult:
    """
    Evaluate aerodynamic convergence from OpenFOAM force coefficients and log residuals.
    """
    case_path = Path(case_dir)
    reasons: list[str] = []

    # 1. Search for force coefficients file
    dat_candidates = list(case_path.glob("postProcessing/**/coefficient.dat")) + list(
        case_path.glob("postProcessing/**/forces*.dat")
    )
    if not dat_candidates:
        reasons.append("No coefficient.dat found in postProcessing directory.")
        return ConvergenceResult(
            status="diverged",
            iterations_completed=0,
            cl_mean=float("nan"),
            cl_std=float("nan"),
            cd_mean=float("nan"),
            cd_std=float("nan"),
            cm_mean=float("nan"),
            cm_std=float("nan"),
            reasons=tuple(reasons),
        )

    coeffs = parse_force_coeffs_file(dat_candidates[0])
    log_info = parse_solver_log(case_path / "log.simpleFoam")

    cl = coeffs.get("cl", np.array([]))
    cd = coeffs.get("cd", np.array([]))
    cm = coeffs.get("cm", np.array([]))
    n_pts = len(cl)

    if n_pts == 0:
        reasons.append("Empty force coefficient data extracted.")
        return ConvergenceResult(
            status="diverged",
            iterations_completed=log_info["iterations"],
            cl_mean=float("nan"),
            cl_std=float("nan"),
            cd_mean=float("nan"),
            cd_std=float("nan"),
            cm_mean=float("nan"),
            cm_std=float("nan"),
            reasons=tuple(reasons),
        )

    iters_completed = max(n_pts, log_info["iterations"])

    # Check for NaN / Inf in aerodynamic coefficients
    if not (np.all(np.isfinite(cl)) and np.all(np.isfinite(cd)) and np.all(np.isfinite(cm))):
        reasons.append("NaN or Infinite values detected in aerodynamic force histories.")
        return ConvergenceResult(
            status="diverged",
            iterations_completed=iters_completed,
            cl_mean=float("nan"),
            cl_std=float("nan"),
            cd_mean=float("nan"),
            cd_std=float("nan"),
            cm_mean=float("nan"),
            cm_std=float("nan"),
            reasons=tuple(reasons),
        )

    # Window analysis
    w = min(window_size, n_pts)
    cl_window = cl[-w:]
    cd_window = cd[-w:]
    cm_window = cm[-w:]

    cl_mean = float(np.mean(cl_window))
    cl_std = float(np.std(cl_window))
    cd_mean = float(np.mean(cd_window))
    cd_std = float(np.std(cd_window))
    cm_mean = float(np.mean(cm_window))
    cm_std = float(np.std(cm_window))

    # Evaluate criteria
    cl_converged = cl_std <= cl_std_tol
    cd_converged = cd_std <= cd_std_tol
    cm_converged = cm_std <= cm_std_tol
    force_converged = cl_converged and cd_converged and cm_converged

    if log_info.get("diverged", False):
        status = "diverged"
        reasons.append("Solver log indicates numerical failure (divergence or floating point error).")
    elif force_converged and w >= min(50, window_size):
        status = "converged"
    else:
        status = "unconverged"
        if not cl_converged:
            reasons.append(f"CL std ({cl_std:.2e}) exceeded tolerance ({cl_std_tol:.2e}).")
        if not cd_converged:
            reasons.append(f"CD std ({cd_std:.2e}) exceeded tolerance ({cd_std_tol:.2e}).")
        if not cm_converged:
            reasons.append(f"Cm std ({cm_std:.2e}) exceeded tolerance ({cm_std_tol:.2e}).")

    residuals_tuple = {k: tuple(float(x) for x in v) for k, v in log_info["residuals"].items()}

    return ConvergenceResult(
        status=status,
        iterations_completed=iters_completed,
        cl_mean=cl_mean,
        cl_std=cl_std,
        cd_mean=cd_mean,
        cd_std=cd_std,
        cm_mean=cm_mean,
        cm_std=cm_std,
        cl_history=tuple(float(x) for x in cl),
        cd_history=tuple(float(x) for x in cd),
        cm_history=tuple(float(x) for x in cm),
        residuals=residuals_tuple,
        reasons=tuple(reasons),
    )
