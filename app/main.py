"""
NeuroFoil: Interactive Aerodynamic Research Engineering Platform.

Streamlit application providing:
1. Airfoil geometry morphing and surrogate estimates with explicit backend reporting.
2. Continuous surface pressure distributions (Cp) with nominal 95% uncertainty intervals.
3. Domain-applicability and Mahalanobis out-of-distribution (OOD) indicators.
4. Aero-structural wingbox mechanics and fuel-bay cross-sectional area sizing.
5. Constrained multidisciplinary surrogate design search.
6. Optional live OpenFOAM CFD comparison in the configured solver environment.
7. Cached polar sweeps (CL vs alpha, CD vs CL, L/D vs alpha).
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.geometry.naca import generate_naca4_airfoil
from src.geometry.validator import validate_airfoil_geometry
from src.models.inference import AeroSurrogateEngine

# Page configuration
st.set_page_config(
    page_title="NeuroFoil — Research Engineering Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize surrogate engine
@st.cache_resource
def get_engine() -> AeroSurrogateEngine:
    """Load one inference engine per Streamlit process."""
    return AeroSurrogateEngine()


# Polar-sweep cache
@st.cache_data(show_spinner=False)
def get_polar_sweep(
    m: float,
    p: float,
    t: float,
    reynolds: float,
    model_backend: str,
    field_backend: str,
    artifact_fingerprint: str,
) -> pd.DataFrame:
    """Return a cached 41-point scalar polar for one geometry/backend artifact."""
    # Backend/provenance inputs are intentionally part of Streamlit's cache key so
    # changed artifacts cannot reuse an older polar. Cp is not used in the table,
    # but including its backend gives callers the full prediction identity.
    if not model_backend or not field_backend or not artifact_fingerprint:
        raise ValueError("backend identity and artifact fingerprint must be non-empty")
    sweep_engine = get_engine()
    sweep_alphas = np.linspace(-6.0, 14.0, 41)
    rows = []
    for sweep_alpha in sweep_alphas:
        sweep_pred = sweep_engine.predict(
            m,
            p,
            t,
            float(sweep_alpha),
            reynolds,
            compute_structures=False,
        )
        if not isinstance(sweep_pred, dict):
            raise RuntimeError("polar sweep prediction returned an unexpected result type")
        if str(sweep_pred.get("model_backend", "unreported")) != model_backend:
            raise RuntimeError("scalar backend changed while computing the cached polar")
        if str(sweep_pred.get("field_backend", "unreported")) != field_backend:
            raise RuntimeError("Cp field backend changed while computing the cached polar")
        try:
            sweep_values = {
                "CL": float(sweep_pred["cl"]),
                "CD": float(sweep_pred["cd"]),
                "LD": float(sweep_pred["lift_to_drag"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid scalar output in polar sweep: {exc}") from exc
        if not all(np.isfinite(value) for value in sweep_values.values()):
            raise RuntimeError("non-finite scalar output in polar sweep")
        rows.append(
            {
                "alpha": float(sweep_alpha),
                **sweep_values,
            }
        )
    return pd.DataFrame(rows)


engine = get_engine()

# Helper function to apply aerospace dark theme to matplotlib plots
def apply_aero_theme(fig, ax):
    """Apply aerospace dark cockpit theme to matplotlib figures."""
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    ax.grid(True, color="#334155", alpha=0.5, linestyle="--", linewidth=0.8)


def style_legend(ax, loc="upper right", fontsize=7.5):
    """Style legend for dark aerospace theme."""
    legend = ax.legend(fontsize=fontsize, loc=loc, facecolor="#1E293B", edgecolor="#334155")
    if legend:
        for text in legend.get_texts():
            text.set_color("#F1F5F9")
    return legend


def format_domain_status(uq_report) -> str:
    """Map internal UQ status text to precise, non-certification UI language."""
    if bool(uq_report.is_ood):
        return "Outside calibrated training domain"
    status_text = str(uq_report.status)
    if "Moderate" in status_text or "Elevated" in status_text:
        return "In-domain surrogate estimate — elevated uncertainty/separation indicator"
    return "In-domain surrogate estimate"


def safe_case_token(value: float, decimals: int) -> str:
    """Format a numeric value for an unambiguous filesystem-safe case ID."""
    return f"{value:.{decimals}f}".replace("-", "m").replace("+", "p").replace(".", "p")


# Modern Executive Aerospace CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base App background */
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        letter-spacing: 0.03em;
        margin-bottom: 1.25rem;
        font-weight: 500;
    }

    /* Top KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 14px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.50);
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(14, 165, 233, 0.18);
    }
    .kpi-label {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        color: #F8FAFC;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Domain-applicability status badges */
    .cert-badge-green {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 5px solid #10B981;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
        color: #34D399;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.08);
    }
    .cert-badge-yellow {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
        color: #FBBF24;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.08);
    }
    .cert-badge-red {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 5px solid #EF4444;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 16px;
        color: #F87171;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.08);
    }

    /* Buttons & Interactive Elements */
    div.stButton > button {
        cursor: pointer !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #1D4ED8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(2, 132, 199, 0.55) !important;
        border-color: #38BDF8 !important;
    }
    div.stButton > button[kind="primary"]:active {
        transform: translateY(1px) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #334155 !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        transform: translateY(-1px) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        cursor: pointer !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        color: #94A3B8 !important;
        transition: color 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #38BDF8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Sliders, Selectboxes, and Cursors */
    div[data-baseweb="slider"] * {
        cursor: grab !important;
    }
    div[data-baseweb="slider"] *:active {
        cursor: grabbing !important;
    }
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    .streamlit-expanderHeader {
        cursor: pointer !important;
        font-weight: 600 !important;
    }

    /* Streamlit Metric Card Polish */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.45rem !important;
        color: #F8FAFC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">✈️ NeuroFoil</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Research Engineering Platform • Uncertainty & Domain-Applicability Indicators • Aero-Structural Design Search</div>',
    unsafe_allow_html=True,
)
st.caption("Research-use estimates only; this application does not provide airworthiness or design certification.")

# Sidebar: Geometry & Flow Inputs
st.sidebar.header("📐 Airfoil Geometry")

preset = st.sidebar.selectbox(
    "Load Benchmark Preset",
    ["Custom", "NACA 0012 (Symmetric)", "NACA 2412 (General Aviation)", "NACA 4412 (High Camber)", "NACA 0015 (Thick)"],
)

if preset == "NACA 0012 (Symmetric)":
    init_m, init_p, init_t = 0.00, 0.20, 0.12
elif preset == "NACA 2412 (General Aviation)":
    init_m, init_p, init_t = 0.02, 0.40, 0.12
elif preset == "NACA 4412 (High Camber)":
    init_m, init_p, init_t = 0.04, 0.40, 0.12
elif preset == "NACA 0015 (Thick)":
    init_m, init_p, init_t = 0.00, 0.20, 0.15
else:
    init_m, init_p, init_t = 0.02, 0.40, 0.12

camber_m = st.sidebar.slider("Maximum Camber (m)", 0.00, 0.06, init_m, step=0.005, format="%.3f")
camber_p = st.sidebar.slider("Camber Location (p)", 0.20, 0.60, init_p, step=0.05, format="%.2f")
thickness_t = st.sidebar.slider("Thickness Ratio (t)", 0.06, 0.18, init_t, step=0.005, format="%.3f")

st.sidebar.header("🌪️ Operating Conditions")
alpha_deg = st.sidebar.slider("Angle of Attack (α)", -6.0, 14.0, 4.0, step=0.5, format="%.1f°")
reynolds = st.sidebar.select_slider(
    "Reynolds Number (Re)",
    options=[1.0e6, 2.0e6, 3.0e6, 6.0e6],
    value=6.0e6,
    format_func=lambda x: f"{x:.1e}",
)

# Format NACA code
m_digit = int(round(camber_m * 100))
p_digit = int(round(camber_p * 10)) if camber_m > 0 else 0
t_digit = int(round(thickness_t * 100))
naca_code = f"{m_digit:1d}{p_digit:1d}{t_digit:02d}"

# Generate continuous geometry & validate. The four-digit code remains a rounded
# display/case identifier; plots and structural metrics use the slider values.
try:
    airfoil = generate_naca4_airfoil(
        naca_code,
        num_points=161,
        closed_te=True,
        m=camber_m,
        p=camber_p,
        t=thickness_t,
    )
    val_res = validate_airfoil_geometry(airfoil)
except Exception as exc:
    st.error(f"Error generating airfoil geometry near NACA {naca_code}: {exc}")
    st.stop()

# Surrogate evaluation with uncertainty and aero-structures
try:
    pred = engine.predict(camber_m, camber_p, thickness_t, alpha_deg, reynolds)
except Exception as exc:
    st.error(
        "Surrogate evaluation failed: "
        f"{type(exc).__name__}: {(' '.join(str(exc).split()) or 'no detail')[:500]}"
    )
    st.stop()
if not isinstance(pred, dict):
    st.error(f"Surrogate evaluation returned an unexpected result type: {type(pred).__name__}")
    st.stop()
required_prediction_keys = {
    "cl", "cd", "cm", "lift_to_drag", "x_grid", "cp_upper", "cp_lower", "uq"
}
missing_prediction_keys = sorted(required_prediction_keys.difference(pred))
if missing_prediction_keys:
    st.error(f"Surrogate result is missing required fields: {', '.join(missing_prediction_keys)}")
    st.stop()
try:
    scalar_values = {
        name: float(pred[name]) for name in ("cl", "cd", "cm", "lift_to_drag")
    }
    pred.update(scalar_values)
    pred_x = np.asarray(pred["x_grid"], dtype=float)
    pred_cp_upper = np.asarray(pred["cp_upper"], dtype=float)
    pred_cp_lower = np.asarray(pred["cp_lower"], dtype=float)
    pred["x_grid"] = pred_x
    pred["cp_upper"] = pred_cp_upper
    pred["cp_lower"] = pred_cp_lower
    if not all(np.isfinite(value) for value in scalar_values.values()):
        raise ValueError("scalar outputs contain non-finite values")
    if not (
        pred_x.ndim == pred_cp_upper.ndim == pred_cp_lower.ndim == 1
        and len(pred_x) == len(pred_cp_upper) == len(pred_cp_lower)
        and len(pred_x) > 1
        and np.all(np.isfinite(pred_x))
        and np.all(np.isfinite(pred_cp_upper))
        and np.all(np.isfinite(pred_cp_lower))
    ):
        raise ValueError("Cp outputs must be finite one-dimensional arrays with matching lengths")
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Surrogate result contains invalid numeric data: {exc}")
    st.stop()
uq = pred["uq"]
required_uq_attributes = (
    "cl_ci95", "cd_ci95", "cm_ci95", "ld_ci95", "is_ood", "status",
    "confidence_score", "mahalanobis_dist", "warning_messages",
    "cp_upper_ci95_low", "cp_upper_ci95_high", "cp_lower_ci95_low", "cp_lower_ci95_high",
)
missing_uq_attributes = [name for name in required_uq_attributes if not hasattr(uq, name)]
if missing_uq_attributes:
    st.error(f"Uncertainty report is missing required fields: {', '.join(missing_uq_attributes)}")
    st.stop()
try:
    confidence_score = float(uq.confidence_score)
    mahalanobis_dist = float(uq.mahalanobis_dist)
    if not np.isfinite(confidence_score) or not np.isfinite(mahalanobis_dist):
        raise ValueError("status metrics must be finite")
    uq_intervals: dict[str, tuple[float, float]] = {}
    uq_bands: dict[str, np.ndarray] = {}
    for interval_name in ("cl_ci95", "cd_ci95", "cm_ci95", "ld_ci95"):
        interval = tuple(float(value) for value in getattr(uq, interval_name))
        if len(interval) != 2 or not all(np.isfinite(interval)):
            raise ValueError(f"{interval_name} is not a finite two-value interval")
        if interval[0] > interval[1]:
            raise ValueError(f"{interval_name} bounds are reversed")
        uq_intervals[interval_name] = interval
    for band_name in (
        "cp_upper_ci95_low", "cp_upper_ci95_high", "cp_lower_ci95_low", "cp_lower_ci95_high"
    ):
        band = np.asarray(getattr(uq, band_name), dtype=float)
        if band.ndim != 1 or len(band) != len(pred_x) or not np.all(np.isfinite(band)):
            raise ValueError(f"{band_name} does not match the Cp grid")
        uq_bands[band_name] = band
    if np.any(uq_bands["cp_upper_ci95_low"] > uq_bands["cp_upper_ci95_high"]):
        raise ValueError("upper-surface Cp interval bounds are reversed")
    if np.any(uq_bands["cp_lower_ci95_low"] > uq_bands["cp_lower_ci95_high"]):
        raise ValueError("lower-surface Cp interval bounds are reversed")
except (TypeError, ValueError) as exc:
    st.error(f"Uncertainty report contains invalid numeric data: {exc}")
    st.stop()
wb = pred.get("wingbox")
model_backend = str(pred.get("model_backend", "unreported"))
field_backend = str(pred.get("field_backend", "unreported"))
model_provenance = str(pred.get("model_provenance", "unreported"))

# Backends are part of the engineering result, not hidden implementation details.
backend_col1, backend_col2 = st.columns(2)
backend_col1.metric("Scalar Model Backend", model_backend)
backend_col2.metric("Cp Field Backend", field_backend)
st.caption(
    f"Scalar artifact/data provenance: `{model_provenance}` • "
    f"Domain covariance provenance: `{getattr(uq, 'covariance_provenance', 'unreported')}`"
)
if model_backend == "neural" and model_provenance.lower() not in {
    "real_openfoam_cfd",
    "openfoam_cfd",
}:
    st.warning(
        "The scalar neural bundle is a legacy or otherwise unverified artifact "
        f"(provenance: {model_provenance}); it is not evidence of real-CFD performance."
    )
if model_backend.startswith("analytic_fallback"):
    st.warning(
        "Scalar backend is an analytic fallback. CL, CD, Cm, and L/D are not "
        "neural or CFD-trained predictions; treat them as preliminary analytic estimates."
    )
if field_backend.startswith("analytic_fallback"):
    st.warning(
        "Cp field backend is an analytic fallback. The displayed pressure field and "
        "its bands are not outputs from a CFD-trained neural field decoder."
    )

# Top KPI Metric Banner
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">LIFT COEFFICIENT (CL)</div><div class="kpi-value">{pred["cl"]:.4f}</div>'
        f'<div class="kpi-sub">Nominal 95% interval: [{uq_intervals["cl_ci95"][0]:.4f}, {uq_intervals["cl_ci95"][1]:.4f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">DRAG COEFFICIENT (CD)</div><div class="kpi-value">{pred["cd"]:.5f}</div>'
        f'<div class="kpi-sub">Nominal 95% interval: [{uq_intervals["cd_ci95"][0]:.5f}, {uq_intervals["cd_ci95"][1]:.5f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">AERO EFFICIENCY (L/D)</div><div class="kpi-value">{pred["lift_to_drag"]:.1f}</div>'
        f'<div class="kpi-sub">Nominal 95% interval: [{uq_intervals["ld_ci95"][0]:.1f}, {uq_intervals["ld_ci95"][1]:.1f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">MOMENT (Cm @ c/4)</div><div class="kpi-value">{pred["cm"]:.4f}</div>'
        f'<div class="kpi-sub">Nominal 95% interval: [{uq_intervals["cm_ci95"][0]:.4f}, {uq_intervals["cm_ci95"][1]:.4f}]</div></div>',
        unsafe_allow_html=True,
    )
st.caption(
    "Intervals are nominal; achieved coverage must be checked empirically on independent geometries "
    "and may differ from 95%. Analytic fallback intervals are indicative only."
)

# Domain-applicability and UQ status banner. This is an engineering-use
# indicator, not an airworthiness certification.
uq_status_text = str(uq.status)
if bool(uq.is_ood):
    cert_cls = "cert-badge-red"
    cert_icon = "🚨"
    cert_title = "OUTSIDE CALIBRATED TRAINING DOMAIN"
elif "Moderate" in uq_status_text or "Elevated" in uq_status_text:
    cert_cls = "cert-badge-yellow"
    cert_icon = "⚠️"
    cert_title = "IN-DOMAIN SURROGATE ESTIMATE — ELEVATED UNCERTAINTY / SEPARATION INDICATOR"
else:
    cert_cls = "cert-badge-green"
    cert_icon = "🛡️"
    cert_title = "IN-DOMAIN SURROGATE ESTIMATE"

warn_str = ""
try:
    first_warning = next(iter(uq.warning_messages), None)
except TypeError:
    first_warning = uq.warning_messages
if first_warning:
    warn_str = f" • Warning: {html.escape(str(first_warning)[:500])}"
st.markdown(
    f'<div class="{cert_cls}">'
    f'<strong>{cert_icon} {cert_title}</strong><br>'
    f'<span style="font-size: 0.88rem;">Heuristic confidence score: <strong>{confidence_score:.1f}%</strong> | '
    f'Mahalanobis Distance D<sub>M</sub>: <strong>{mahalanobis_dist:.2f}</strong> | '
    f'Domain/UQ Status: <em>{format_domain_status(uq)}</em>{warn_str}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
if getattr(uq, "covariance_provenance", "fallback") != "empirical":
    st.caption(
        "The domain-distance score is using fallback covariance rather than a calibrated empirical "
        "real-campaign manifold; interpret the in-domain label as provisional."
    )

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Real-Time Aero & Structural Analysis",
    "Surrogate inverse-design search",
    "💨 Live OpenFOAM CFD Comparison",
    "📈 Cached Polar Sweeps",
])

# -------------------------------------------------------------
# TAB 1: Real-Time Predictions & Plots
# -------------------------------------------------------------
with tab1:
    col_geom, col_cp = st.columns(2)

    with col_geom:
        fig_geom, ax_geom = plt.subplots(figsize=(7, 3.8))
        apply_aero_theme(fig_geom, ax_geom)

        ax_geom.plot(airfoil.x, airfoil.y_upper, color="#38BDF8", linewidth=2.2, label="Upper Skin")
        ax_geom.plot(airfoil.x, airfoil.y_lower, color="#38BDF8", linewidth=2.2, label="Lower Skin")

        # Plot Aero-Structural Wingbox and Spars
        if wb:
            (xf_l, yf_l), (xf_u, yf_u) = wb.front_spar_coords
            (xr_l, yr_l), (xr_u, yr_u) = wb.rear_spar_coords

            # Front Spar Web
            ax_geom.plot([xf_l, xf_u], [yf_l, yf_u], color="#F43F5E", linewidth=2.5, solid_capstyle="round",
                         label=f"Front Spar (20% c, h={wb.h_front_spar*100:.1f}%)")
            # Rear Spar Web
            ax_geom.plot([xr_l, xr_u], [yr_l, yr_u], color="#F59E0B", linewidth=2.5, solid_capstyle="round",
                         label=f"Rear Spar (65% c, h={wb.h_rear_spar*100:.1f}%)")

            # Fuel-Bay Cross-Section Shading
            mask_bay = (airfoil.x >= wb.x_front_spar) & (airfoil.x <= wb.x_rear_spar)
            ax_geom.fill_between(
                airfoil.x[mask_bay],
                airfoil.y_lower[mask_bay],
                airfoil.y_upper[mask_bay],
                color="#818CF8",
                alpha=0.32,
                label=f"Fuel-Bay Cross-Section ({wb.fuel_area_fraction*100:.0f}% of section area)",
            )
            # Neutral Axis
            ax_geom.axhline(wb.y_centroid, color="#94A3B8", linestyle=":", linewidth=1.2,
                            label=f"Neutral Axis (ȳ={wb.y_centroid*100:+.2f}%)")
        else:
            ax_geom.fill_between(airfoil.x, airfoil.y_lower, airfoil.y_upper, color="#38BDF8", alpha=0.18)

        ax_geom.set_aspect("equal")
        ax_geom.set_xlim(-0.05, 1.05)
        ax_geom.set_ylim(-0.25, 0.25)
        ax_geom.set_xlabel("x / c", fontsize=10)
        ax_geom.set_ylabel("y / c", fontsize=10)
        ax_geom.set_title(f"Aero-Structural Wingbox: Geometry Near NACA {naca_code} (t={thickness_t*100:.1f}%)", fontsize=11, fontweight="bold")
        style_legend(ax_geom, loc="upper right")
        st.pyplot(fig_geom)
        plt.close(fig_geom)

    with col_cp:
        fig_cp, ax_cp = plt.subplots(figsize=(7, 3.8))
        apply_aero_theme(fig_cp, ax_cp)

        # Mean Cp curves
        ax_cp.plot(pred["x_grid"], pred["cp_upper"], color="#F43F5E", linewidth=2.0, label="Upper Suction (Cp,upper)")
        ax_cp.plot(pred["x_grid"], pred["cp_lower"], color="#38BDF8", linewidth=2.0, label="Lower Pressure (Cp,lower)")

        # 95% Uncertainty Bands
        ax_cp.fill_between(pred["x_grid"], uq_bands["cp_upper_ci95_low"], uq_bands["cp_upper_ci95_high"],
                           color="#F43F5E", alpha=0.22, label="Nominal 95% UQ Band (Suction)")
        ax_cp.fill_between(pred["x_grid"], uq_bands["cp_lower_ci95_low"], uq_bands["cp_lower_ci95_high"],
                           color="#38BDF8", alpha=0.22, label="Nominal 95% UQ Band (Pressure)")

        ax_cp.invert_yaxis()  # Aerospace convention: negative Cp suction on top
        ax_cp.set_xlabel("Chordwise Position x / c", fontsize=10)
        ax_cp.set_ylabel("Pressure Coefficient (Cp)", fontsize=10)
        ax_cp.set_title(f"Surface Cp(x) with Nominal 95% UQ Bounds (α = {alpha_deg:.1f}°)", fontsize=11, fontweight="bold")
        style_legend(ax_cp, loc="lower right")
        st.pyplot(fig_cp)
        plt.close(fig_cp)

    # Aero-Structural Sizing Metrics Row
    if wb:
        st.markdown("##### 🏗️ Aero-Structural & Wingbox Cross-Section Metrics")
        st1, st2, st3, st4 = st.columns(4)
        st1.metric("Front Spar Depth (20% c)", f"{wb.h_front_spar*100:.2f}% chord", f"{wb.h_front_spar:.4f} c")
        st2.metric("Rear Spar Depth (65% c)", f"{wb.h_rear_spar*100:.2f}% chord", f"{wb.h_rear_spar:.4f} c")
        st3.metric("Fuel-Bay Cross-Sectional Area", f"{wb.fuel_tank_area*100:.2f}% c²", f"{wb.fuel_area_fraction*100:.1f}% of section area")
        st4.metric("Second Moment of Area (Ixx)", f"{wb.i_xx*1e4:.2f} × 10⁻⁴ c⁴", f"η={wb.bending_efficiency:.3f}")

    # Detailed metadata expander
    with st.expander("🔍 Airfoil Geometric Parameters & CST Bernstein Weights"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Thickness Measured", f"{airfoil.max_thickness_measured*100:.2f}%")
        m2.metric("Max Camber Measured", f"{airfoil.max_camber_measured*100:.2f}%")
        m3.metric("Trailing Edge Gap", f"{airfoil.te_gap:.2e}")
        m4.metric("Geometric Status", "VALID" if val_res.is_valid else "INVALID")
        st.write("**5th-Order Kulfan CST Polynomial Weights:**")
        cst_df = pd.DataFrame({
            "Upper Surface CST Weights": list(airfoil.cst_upper),
            "Lower Surface CST Weights": list(airfoil.cst_lower),
        })
        st.dataframe(cst_df.T, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Surrogate Inverse-Design Search
# -------------------------------------------------------------
with tab2:
    st.subheader("Constrained Surrogate Multidisciplinary Design Search")
    st.write(
        "Uses the active surrogate backend to search for an airfoil that improves aerodynamic efficiency ($L/D$) "
        "while satisfying user-selected lift, spar-depth, and fuel-bay cross-sectional area constraints. "
        "The result is a surrogate-optimized candidate and requires independent verification."
    )

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        st.markdown("**Aerodynamic Constraints**")
        target_cl = st.number_input("Target Cruise Lift (CL)", min_value=0.20, max_value=1.00, value=0.50, step=0.05)
        min_thick = st.slider("Minimum Thickness Ratio (t_min)", 0.08, 0.14, 0.10, step=0.01)
        max_thick = st.slider("Maximum Thickness Ratio (t_max)", 0.12, 0.18, 0.16, step=0.01)
        max_cam = st.slider("Maximum Camber (m_max)", 0.01, 0.05, 0.04, step=0.005)

        st.markdown("**Structural Wingbox Constraints**")
        min_front_h = st.slider("Min Front Spar Depth (h_front)", 0.06, 0.13, 0.085, step=0.005, format="%.3f c")
        min_rear_h = st.slider("Min Rear Spar Depth (h_rear)", 0.03, 0.08, 0.045, step=0.005, format="%.3f c")
        min_fuel = st.slider("Min Fuel-Bay Area (A_fuel)", 0.02, 0.06, 0.035, step=0.005, format="%.3f c²")

        run_opt = st.button("Run Surrogate Aero-Structural Inverse Design", type="primary")

    with col_opt2:
        if run_opt:
            with st.spinner("Solving the SLSQP search with the active surrogate backend..."):
                try:
                    opt_res = engine.optimize_airfoil(
                        cl_target=target_cl,
                        t_min=min_thick,
                        t_max=max_thick,
                        m_max=max_cam,
                        min_front_spar=min_front_h,
                        min_rear_spar=min_rear_h,
                        min_fuel_area=min_fuel,
                        x0=(camber_m, camber_p, thickness_t, alpha_deg),
                    )
                except Exception as exc:
                    st.error(
                        "Surrogate optimization failed: "
                        f"{type(exc).__name__}: {(' '.join(str(exc).split()) or 'no detail')[:500]}"
                    )
                    st.stop()

            if opt_res["success"]:
                st.success(
                    f"Surrogate search converged in {opt_res['elapsed_seconds']:.3f} seconds "
                    f"({opt_res['iterations']} iterations); independent verification is still required."
                )
            else:
                st.warning(
                    f"SLSQP did not fully converge after {opt_res['iterations']} iterations "
                    f"({opt_res['elapsed_seconds']:.3f} s) — showing the best surrogate candidate found; "
                    "independent verification is required."
                )

            bp = opt_res["baseline_params"]
            op = opt_res["opt_params"]
            bpred = opt_res["baseline_pred"]
            opred = opt_res["opt_pred"]
            bwb = bpred.get("wingbox")
            owb = opred.get("wingbox")
            buq = bpred["uq"]
            ouq = opred["uq"]
            structural_constraints_met = bool(opt_res.get("structural_constraints_met", False))
            st.caption(
                "Reported surrogate-candidate structural constraint check: "
                f"{'passed' if structural_constraints_met else 'not passed'} (geometric section constraints only)."
            )
            st.caption(
                "Optimization evaluation backends — baseline: "
                f"scalar `{bpred.get('model_backend', 'unreported')}` "
                f"(provenance `{bpred.get('model_provenance', 'unreported')}`), "
                f"Cp `{bpred.get('field_backend', 'unreported')}`; candidate: "
                f"scalar `{opred.get('model_backend', 'unreported')}` "
                f"(provenance `{opred.get('model_provenance', 'unreported')}`), "
                f"Cp `{opred.get('field_backend', 'unreported')}`."
            )

            comp_df = pd.DataFrame({
                "Metric": [
                    "Camber (m)", "Thickness (t)", "Angle of Attack (α)",
                    "Lift (CL)", "Drag (CD)", "Efficiency (L/D)",
                    "Front Spar Depth", "Rear Spar Depth", "Fuel-Bay Area",
                    "Second Moment of Area (Ixx)", "Heuristic Confidence Score",
                ],
                "Baseline": [
                    f"{bp['m']:.3f}", f"{bp['t']*100:.1f}%", f"{bp['alpha']:.1f}°",
                    f"{bpred['cl']:.4f}", f"{bpred['cd']:.5f}", f"{bpred['lift_to_drag']:.1f}",
                    f"{bwb.h_front_spar*100:.1f}%" if bwb else "-",
                    f"{bwb.h_rear_spar*100:.1f}%" if bwb else "-",
                    f"{bwb.fuel_tank_area*100:.2f}%" if bwb else "-",
                    f"{bwb.i_xx*1e4:.2f}e-4" if bwb else "-",
                    f"{buq.confidence_score:.1f}%",
                ],
                "Surrogate-Optimized Design": [
                    f"{op['m']:.3f}", f"{op['t']*100:.1f}%", f"{op['alpha']:.1f}°",
                    f"{opred['cl']:.4f}", f"{opred['cd']:.5f}", f"{opred['lift_to_drag']:.1f}",
                    f"{owb.h_front_spar*100:.1f}%" if owb else "-",
                    f"{owb.h_rear_spar*100:.1f}%" if owb else "-",
                    f"{owb.fuel_tank_area*100:.2f}%" if owb else "-",
                    f"{owb.i_xx*1e4:.2f}e-4" if owb else "-",
                    f"{ouq.confidence_score:.1f}%",
                ],
                "Status / Change": [
                    "-", "-", "-",
                    f"{(opred['cl']-bpred['cl']):+.3f}", f"{(opred['cd']-bpred['cd']):+.5f}", f"{opt_res['ld_improvement_pct']:+.1f}%",
                    "PASS" if owb and owb.h_front_spar >= min_front_h - 1e-4 else "NOT MET",
                    "PASS" if owb and owb.h_rear_spar >= min_rear_h - 1e-4 else "NOT MET",
                    "PASS" if owb and owb.fuel_tank_area >= min_fuel - 1e-4 else "NOT MET",
                    f"{(owb.i_xx - bwb.i_xx)*1e4:+.2f}e-4" if (owb and bwb) else "-",
                    f"{format_domain_status(ouq)}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Overlay uses direct continuous m/p/t geometry rather than rounded NACA digits.
            try:
                base_af = generate_naca4_airfoil(
                    naca_code,
                    num_points=161,
                    m=bp["m"],
                    p=bp["p"],
                    t=bp["t"],
                )
                opt_code = f"{int(round(op['m']*100)):1d}{int(round(op['p']*10)):1d}{int(round(op['t']*100)):02d}"
                opt_af = generate_naca4_airfoil(
                    opt_code,
                    num_points=161,
                    m=op["m"],
                    p=op["p"],
                    t=op["t"],
                )
            except Exception as exc:
                st.error(
                    "Could not render the surrogate candidate geometry: "
                    f"{type(exc).__name__}: {(' '.join(str(exc).split()) or 'no detail')[:500]}"
                )
                st.stop()

            fig_opt, ax_opt = plt.subplots(figsize=(8, 3.2))
            apply_aero_theme(fig_opt, ax_opt)

            ax_opt.plot(base_af.x, base_af.y_upper, color="#64748B", linestyle="--", linewidth=1.5, label=f"Baseline (near NACA {naca_code})")
            ax_opt.plot(base_af.x, base_af.y_lower, color="#64748B", linestyle="--", linewidth=1.5)
            ax_opt.plot(opt_af.x, opt_af.y_upper, color="#38BDF8", linewidth=2.2, label=f"Surrogate-Optimized Candidate (near NACA {opt_code})")
            ax_opt.plot(opt_af.x, opt_af.y_lower, color="#38BDF8", linewidth=2.2)

            if owb:
                (oxf_l, oyf_l), (oxf_u, oyf_u) = owb.front_spar_coords
                (oxr_l, oyr_l), (oxr_u, oyr_u) = owb.rear_spar_coords
                ax_opt.plot([oxf_l, oxf_u], [oyf_l, oyf_u], color="#F43F5E", linewidth=2.2, label="Front Spar")
                ax_opt.plot([oxr_l, oxr_u], [oyr_l, oyr_u], color="#F59E0B", linewidth=2.2, label="Rear Spar")
                mask_obay = (opt_af.x >= owb.x_front_spar) & (opt_af.x <= owb.x_rear_spar)
                ax_opt.fill_between(opt_af.x[mask_obay], opt_af.y_lower[mask_obay], opt_af.y_upper[mask_obay], color="#818CF8", alpha=0.25, label="Candidate Wingbox")

            ax_opt.set_aspect("equal")
            ax_opt.set_xlim(-0.05, 1.05)
            ax_opt.set_ylim(-0.25, 0.25)
            ax_opt.set_xlabel("x / c")
            ax_opt.set_ylabel("y / c")
            ax_opt.set_title(
                f"Aero-Structural Candidate Overlay ({opt_res['ld_improvement_pct']:+.1f}% L/D; Verify Constraints)",
                fontweight="bold",
            )
            style_legend(ax_opt, loc="upper right", fontsize=8)
            st.pyplot(fig_opt)
            plt.close(fig_opt)
        else:
            st.info("Adjust the aerodynamic and structural constraints on the left and click **'Run Surrogate Aero-Structural Inverse Design'**.")

# -------------------------------------------------------------
# TAB 3: Live OpenFOAM CFD Comparison
# -------------------------------------------------------------
with tab3:
    st.subheader("💨 Live OpenFOAM CFD Comparison")
    st.write(
        r"Run the configured steady RANS pipeline (`simpleFoam`, $k$-$\omega$ SST) "
        r"for an independent numerical comparison with the current surrogate estimate. "
        r"A single run is not ground-truth validation."
    )

    st.markdown(f"**Current Target Case:** geometry near NACA `{naca_code}` at $\\alpha = {alpha_deg:.1f}^\\circ$, $Re = {reynolds:.1e}$")
    cfd_mesh = st.selectbox("CFD Grid Refinement Level", ["coarse", "medium", "fine"], index=0)

    if st.button("Run Live OpenFOAM Simulation", type="primary"):
        from src.dataset.sampler import AirfoilSample
        from src.dataset.pipeline import run_single_cfd_sample

        cfg = load_config()
        sample = AirfoilSample(
            case_id=(
                f"NACA{naca_code}_m{safe_case_token(camber_m, 3)}_"
                f"loc{safe_case_token(camber_p, 2)}_t{safe_case_token(thickness_t, 3)}_"
                f"alpha{safe_case_token(alpha_deg, 1)}_Re{int(reynolds)}_verify"
            ),
            naca_code=naca_code,
            m=camber_m,
            p=camber_p,
            t=thickness_t,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
        )
        case_dir = PROJECT_ROOT / "data" / "raw_cases" / sample.case_id

        with st.spinner(f"Building the {cfd_mesh} C-grid and running the configured OpenFOAM solver..."):
            import time
            t0 = time.perf_counter()
            try:
                res = run_single_cfd_sample(sample, case_dir, cfg, mesh_level=cfd_mesh)
            except Exception as exc:  # Display actionable solver/environment failures in the UI.
                res = {
                    "success": False,
                    "status": "exception",
                    "error_type": type(exc).__name__,
                    "detail": (" ".join(str(exc).split()) or "no exception detail")[:500],
                }
            elapsed = time.perf_counter() - t0

        if res.get("success"):
            rec = res.get("record")
            if not isinstance(rec, dict):
                st.error(f"OpenFOAM run returned success without a usable record. Case directory: {case_dir}")
                st.stop()
            run_status = str(rec.get("status", "unreported"))
            try:
                convergence_details = ", ".join(
                    f"{name}={float(rec[name]):.3g}"
                    for name in ("cl_std", "cd_std", "cm_std")
                    if name in rec and rec[name] is not None
                )
            except (TypeError, ValueError):
                convergence_details = "reported convergence statistics were non-numeric"
            if run_status.lower() == "converged":
                detail_suffix = f" Reported statistics: {convergence_details}." if convergence_details else ""
                st.success(
                    f"OpenFOAM simulation finished in {elapsed:.1f} seconds; "
                    f"status: {run_status.upper()}.{detail_suffix}"
                )
            else:
                detail_suffix = f" Reported statistics: {convergence_details}." if convergence_details else ""
                st.warning(
                    f"OpenFOAM execution finished in {elapsed:.1f} seconds, but the reported "
                    f"solution status is {run_status.upper()}. Review convergence statistics before use."
                    f"{detail_suffix}"
                )

            # OpenFOAM comparison metrics
            try:
                cfd_cl = float(rec["cl"])
                cfd_cd = float(rec["cd"])
                cfd_ld = float(rec["lift_to_drag"])
            except (KeyError, TypeError, ValueError) as exc:
                st.error(
                    "OpenFOAM run completed, but aerodynamic metrics were unavailable: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                st.stop()
            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("Lift Coefficient (CL)", f"{cfd_cl:.4f}", f"{(cfd_cl - pred['cl']):+.4f} vs surrogate")
            with v2:
                st.metric("Drag Coefficient (CD)", f"{cfd_cd:.5f}", f"{(cfd_cd - pred['cd']):+.5f} vs surrogate")
            with v3:
                st.metric("Lift-to-Drag Ratio (L/D)", f"{cfd_ld:.1f}", f"{(cfd_ld - pred['lift_to_drag']):+.1f} vs surrogate")

            # Surface Cp comparison if available
            cp_data = res.get("cp_data") or {}
            if not isinstance(cp_data, dict):
                cp_data = {"available": False, "error": "unexpected Cp result format"}
            if cp_data.get("available"):
                try:
                    cfd_x = np.asarray(cp_data["x_grid"], dtype=float)
                    cfd_cp_upper = np.asarray(cp_data["cp_upper"], dtype=float)
                    cfd_cp_lower = np.asarray(cp_data["cp_lower"], dtype=float)
                    if not (
                        cfd_x.ndim == cfd_cp_upper.ndim == cfd_cp_lower.ndim == 1
                        and len(cfd_x) == len(cfd_cp_upper) == len(cfd_cp_lower)
                        and len(cfd_x) > 1
                        and np.all(np.isfinite(cfd_x))
                        and np.all(np.isfinite(cfd_cp_upper))
                        and np.all(np.isfinite(cfd_cp_lower))
                    ):
                        raise ValueError("Cp arrays must be finite one-dimensional arrays with matching lengths")
                except (KeyError, TypeError, ValueError) as exc:
                    st.warning(
                        "OpenFOAM Cp comparison unavailable because the returned field data were invalid: "
                        f"{type(exc).__name__}: {str(exc)[:300]}"
                    )
                else:
                    fig_vcp, ax_vcp = plt.subplots(figsize=(8, 3.5))
                    apply_aero_theme(fig_vcp, ax_vcp)

                    ax_vcp.plot(cfd_x, cfd_cp_upper, color="#10B981", label="OpenFOAM CFD Upper Cp", linewidth=1.8)
                    ax_vcp.plot(cfd_x, cfd_cp_lower, color="#34D399", label="OpenFOAM CFD Lower Cp", linewidth=1.8)
                    ax_vcp.plot(pred["x_grid"], pred["cp_upper"], color="#F43F5E", linestyle="--", label="Surrogate Upper Cp", linewidth=1.4)
                    ax_vcp.plot(pred["x_grid"], pred["cp_lower"], color="#38BDF8", linestyle="--", label="Surrogate Lower Cp", linewidth=1.4)
                    ax_vcp.invert_yaxis()
                    ax_vcp.set_xlabel("x / c")
                    ax_vcp.set_ylabel("Cp")
                    ax_vcp.set_title("Surface Cp Comparison: OpenFOAM RANS vs Surrogate", fontweight="bold")
                    style_legend(ax_vcp, loc="lower right", fontsize=8)
                    st.pyplot(fig_vcp)
                    plt.close(fig_vcp)
            else:
                cp_detail = cp_data.get("error") or "surface Cp was not available in the solver output"
                st.warning(f"OpenFOAM Cp comparison unavailable: {str(cp_detail)[:500]}")
        else:
            failure_status = str(res.get("status", "unknown_failure"))
            failure_details = {
                key: value
                for key, value in res.items()
                if key not in {"success", "status", "record", "cp_data"}
            }
            detail_text = ", ".join(
                f"{key}={' '.join(str(value).split())}"
                for key, value in failure_details.items()
            )[:1000]
            message = f"OpenFOAM run failed with status: {failure_status}."
            if detail_text:
                message += f" Details: {detail_text}."
            message += f" Case directory: {case_dir}"
            st.error(message)

# -------------------------------------------------------------
# TAB 4: Cached Aerodynamic Polar Sweeps
# -------------------------------------------------------------
with tab4:
    st.subheader(f"Aerodynamic Polars Near NACA {naca_code}")
    st.write(
        "Cached 41-point polar from the active scalar backend across angles of attack "
        "$\\alpha \\in [-6^\\circ, 14^\\circ]$. Values are backend-labelled estimates, "
        "not CFD verification."
    )

    artifact_fingerprint = f"{model_provenance}:{engine.meta.get('trained_at', 'no-artifact-metadata')}"
    try:
        sweep_df = get_polar_sweep(
            camber_m,
            camber_p,
            thickness_t,
            reynolds,
            model_backend,
            field_backend,
            artifact_fingerprint,
        )
    except Exception as exc:
        st.error(
            "Polar sweep failed: "
            f"{type(exc).__name__}: {(' '.join(str(exc).split()) or 'no detail')[:500]}"
        )
        st.stop()

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        fig_p1, ax_p1 = plt.subplots(figsize=(4.5, 3.8))
        apply_aero_theme(fig_p1, ax_p1)
        ax_p1.plot(sweep_df["alpha"], sweep_df["CL"], color="#38BDF8", linewidth=2.2)
        ax_p1.axvline(alpha_deg, color="#F43F5E", linestyle="--", alpha=0.8, label=f"Current α={alpha_deg:.1f}°")
        ax_p1.set_xlabel("α (deg)")
        ax_p1.set_ylabel("Lift Coefficient (CL)")
        ax_p1.set_title("Lift Curve (CL vs α)", fontweight="bold")
        style_legend(ax_p1, loc="upper left", fontsize=8)
        st.pyplot(fig_p1)
        plt.close(fig_p1)

    with p_col2:
        fig_p2, ax_p2 = plt.subplots(figsize=(4.5, 3.8))
        apply_aero_theme(fig_p2, ax_p2)
        ax_p2.plot(sweep_df["CD"], sweep_df["CL"], color="#10B981", linewidth=2.2)
        ax_p2.set_xlabel("Drag Coefficient (CD)")
        ax_p2.set_ylabel("Lift Coefficient (CL)")
        ax_p2.set_title("Drag Polar (CL vs CD)", fontweight="bold")
        st.pyplot(fig_p2)
        plt.close(fig_p2)

    with p_col3:
        fig_p3, ax_p3 = plt.subplots(figsize=(4.5, 3.8))
        apply_aero_theme(fig_p3, ax_p3)
        ax_p3.plot(sweep_df["alpha"], sweep_df["LD"], color="#F59E0B", linewidth=2.2)
        max_idx = int(np.argmax(sweep_df["LD"].to_numpy()))
        max_a = sweep_df["alpha"].iloc[max_idx]
        max_ld = sweep_df["LD"].iloc[max_idx]
        ax_p3.scatter([max_a], [max_ld], color="#EF4444", s=50, zorder=5, label=f"Max L/D={max_ld:.1f} @ {max_a:.1f}°")
        ax_p3.set_xlabel("α (deg)")
        ax_p3.set_ylabel("Aerodynamic Efficiency (L/D)")
        ax_p3.set_title("Efficiency Curve (L/D vs α)", fontweight="bold")
        style_legend(ax_p3, loc="upper right", fontsize=8)
        st.pyplot(fig_p3)
        plt.close(fig_p3)
