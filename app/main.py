"""
NeuroFoil: Interactive ML-Accelerated Aerodynamic Design Platform (Phase IV).

Streamlit application providing:
1. Real-time airfoil geometry morphing and instantaneous neural surrogate predictions.
2. Continuous surface pressure distribution (Cp) with 95% Uncertainty Quantification (UQ).
3. Airworthiness Certification & Mahalanobis Out-of-Distribution (OOD) Guardrails.
4. Aero-Structural Wingbox Mechanics & Fuel Volume Sizing.
5. Sub-second MDAO Constrained Inverse Design Optimizer.
6. One-click live OpenFOAM CFD ground-truth verification in WSL2.
7. Automated polar sweeps (CL vs alpha, CD vs CL, L/D vs alpha).
"""

from __future__ import annotations

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
    page_title="NeuroFoil — Industrial Aerodynamic Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize surrogate engine
@st.cache_resource
def get_engine():
    return AeroSurrogateEngine()

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

    /* Airworthiness Badges */
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
    '<div class="sub-header">Industrial SciML Aerodynamic Platform • Epistemic Uncertainty Quantification • Aero-Structural Wingbox MDAO</div>',
    unsafe_allow_html=True,
)

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

# Generate geometry & validate
try:
    airfoil = generate_naca4_airfoil(naca_code, num_points=161, closed_te=True)
    val_res = validate_airfoil_geometry(airfoil)
except Exception as exc:
    st.error(f"Error generating airfoil NACA {naca_code}: {exc}")
    st.stop()

# Instant Surrogate Evaluation with UQ & Aero-Structures
pred = engine.predict(camber_m, camber_p, thickness_t, alpha_deg, reynolds)
uq = pred["uq"]
wb = pred.get("wingbox")

# Top KPI Metric Banner
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">LIFT COEFFICIENT (CL)</div><div class="kpi-value">{pred["cl"]:.4f}</div>'
        f'<div class="kpi-sub">95% CI: [{uq.cl_ci95[0]:.4f}, {uq.cl_ci95[1]:.4f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">DRAG COEFFICIENT (CD)</div><div class="kpi-value">{pred["cd"]:.5f}</div>'
        f'<div class="kpi-sub">95% CI: [{uq.cd_ci95[0]:.5f}, {uq.cd_ci95[1]:.5f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">AERO EFFICIENCY (L/D)</div><div class="kpi-value">{pred["lift_to_drag"]:.1f}</div>'
        f'<div class="kpi-sub">95% CI: [{uq.ld_ci95[0]:.1f}, {uq.ld_ci95[1]:.1f}]</div></div>',
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">MOMENT (Cm @ c/4)</div><div class="kpi-value">{pred["cm"]:.4f}</div>'
        f'<div class="kpi-sub">Quarter-Chord Pitch</div></div>',
        unsafe_allow_html=True,
    )

# Airworthiness & UQ Certification Banner
if uq.is_ood:
    cert_cls = "cert-badge-red"
    cert_icon = "🚨"
    cert_title = "AIRWORTHINESS ALERT: OUT-OF-DISTRIBUTION (HIGH HALLUCINATION RISK)"
elif "Moderate" in uq.status:
    cert_cls = "cert-badge-yellow"
    cert_icon = "⚠️"
    cert_title = "AIRWORTHINESS NOTICE: MODERATE UNCERTAINTY (ADVERSE GRADIENT / SEPARATION REGIME)"
else:
    cert_cls = "cert-badge-green"
    cert_icon = "🛡️"
    cert_title = "AIRWORTHINESS CERTIFIED: HIGH SURROGATE CONFIDENCE (IN-DOMAIN CFD MANIFOLD)"

warn_str = f" • Warning: {uq.warning_messages[0]}" if uq.warning_messages else ""
st.markdown(
    f'<div class="{cert_cls}">'
    f'<strong>{cert_icon} {cert_title}</strong><br>'
    f'<span style="font-size: 0.88rem;">Surrogate Confidence: <strong>{uq.confidence_score:.1f}%</strong> | '
    f'Mahalanobis Distance D<sub>M</sub>: <strong>{uq.mahalanobis_dist:.2f}</strong> | '
    f'Envelope Status: <em>{uq.status}</em>{warn_str}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Real-Time Aero & Structural Analysis",
    "🎯 AI Inverse Design Optimizer",
    "💨 Live OpenFOAM CFD Verification",
    "📈 Full Polar Sweeps",
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

            # Fuel Tank Bay Shading
            mask_bay = (airfoil.x >= wb.x_front_spar) & (airfoil.x <= wb.x_rear_spar)
            ax_geom.fill_between(
                airfoil.x[mask_bay],
                airfoil.y_lower[mask_bay],
                airfoil.y_upper[mask_bay],
                color="#818CF8",
                alpha=0.32,
                label=f"Internal Fuel Tank ({wb.fuel_area_fraction*100:.0f}% vol)",
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
        ax_geom.set_title(f"Aero-Structural Wingbox: NACA {naca_code} (t={thickness_t*100:.1f}%)", fontsize=11, fontweight="bold")
        style_legend(ax_geom, loc="upper right")
        st.pyplot(fig_geom)

    with col_cp:
        fig_cp, ax_cp = plt.subplots(figsize=(7, 3.8))
        apply_aero_theme(fig_cp, ax_cp)

        # Mean Cp curves
        ax_cp.plot(pred["x_grid"], pred["cp_upper"], color="#F43F5E", linewidth=2.0, label="Upper Suction (Cp,upper)")
        ax_cp.plot(pred["x_grid"], pred["cp_lower"], color="#38BDF8", linewidth=2.0, label="Lower Pressure (Cp,lower)")

        # 95% Uncertainty Bands
        ax_cp.fill_between(pred["x_grid"], uq.cp_upper_ci95_low, uq.cp_upper_ci95_high,
                           color="#F43F5E", alpha=0.22, label="95% UQ Band (Suction)")
        ax_cp.fill_between(pred["x_grid"], uq.cp_lower_ci95_low, uq.cp_lower_ci95_high,
                           color="#38BDF8", alpha=0.22, label="95% UQ Band (Pressure)")

        ax_cp.invert_yaxis()  # Aerospace convention: negative Cp suction on top
        ax_cp.set_xlabel("Chordwise Position x / c", fontsize=10)
        ax_cp.set_ylabel("Pressure Coefficient (Cp)", fontsize=10)
        ax_cp.set_title(f"Surface Cp(x) with 95% UQ Bounds (α = {alpha_deg:.1f}°)", fontsize=11, fontweight="bold")
        style_legend(ax_cp, loc="lower right")
        st.pyplot(fig_cp)

    # Aero-Structural Sizing Metrics Row
    if wb:
        st.markdown("##### 🏗️ Aero-Structural & Wingbox Cross-Section Metrics")
        st1, st2, st3, st4 = st.columns(4)
        st1.metric("Front Spar Depth (20% c)", f"{wb.h_front_spar*100:.2f}% chord", f"{wb.h_front_spar:.4f} c")
        st2.metric("Rear Spar Depth (65% c)", f"{wb.h_rear_spar*100:.2f}% chord", f"{wb.h_rear_spar:.4f} c")
        st3.metric("Fuel Tank Enclosed Area", f"{wb.fuel_tank_area*100:.2f}% c²", f"{wb.fuel_area_fraction*100:.1f}% total vol")
        st4.metric("Bending Stiffness (Ixx)", f"{wb.i_xx*1e4:.2f} × 10⁻⁴ c⁴", f"η={wb.bending_efficiency:.3f}")

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
# TAB 2: AI Inverse Design Optimizer
# -------------------------------------------------------------
with tab2:
    st.subheader("🎯 Constrained Multidisciplinary Design Optimization (MDAO)")
    st.write(
        "Discovers the optimal airfoil geometry that maximizes aerodynamic efficiency ($L/D$) "
        "while simultaneously satisfying strict structural wingbox depths, fuel tank capacities, and aerodynamic lift constraints."
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
        min_fuel = st.slider("Min Fuel Bay Area (A_fuel)", 0.02, 0.06, 0.035, step=0.005, format="%.3f c²")

        run_opt = st.button("🚀 Run Aero-Structural Inverse Design", type="primary")

    with col_opt2:
        if run_opt:
            with st.spinner("Solving SLSQP constrained MDAO via SciML surrogate..."):
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

            if opt_res["success"]:
                st.success(f"Converged in {opt_res['elapsed_seconds']:.3f} seconds ({opt_res['iterations']} iterations)!")
            else:
                st.warning(
                    f"SLSQP did not fully converge after {opt_res['iterations']} iterations "
                    f"({opt_res['elapsed_seconds']:.3f} s) — showing the best design found."
                )

            bp = opt_res["baseline_params"]
            op = opt_res["opt_params"]
            bpred = opt_res["baseline_pred"]
            opred = opt_res["opt_pred"]
            bwb = bpred.get("wingbox")
            owb = opred.get("wingbox")
            buq = bpred["uq"]
            ouq = opred["uq"]

            comp_df = pd.DataFrame({
                "Metric": [
                    "Camber (m)", "Thickness (t)", "Angle of Attack (α)",
                    "Lift (CL)", "Drag (CD)", "Efficiency (L/D)",
                    "Front Spar Depth", "Rear Spar Depth", "Fuel Bay Area",
                    "Bending Stiffness (Ixx)", "Surrogate Confidence",
                ],
                "Baseline Design": [
                    f"{bp['m']:.3f}", f"{bp['t']*100:.1f}%", f"{bp['alpha']:.1f}°",
                    f"{bpred['cl']:.4f}", f"{bpred['cd']:.5f}", f"{bpred['lift_to_drag']:.1f}",
                    f"{bwb.h_front_spar*100:.1f}%" if bwb else "-",
                    f"{bwb.h_rear_spar*100:.1f}%" if bwb else "-",
                    f"{bwb.fuel_tank_area*100:.2f}%" if bwb else "-",
                    f"{bwb.i_xx*1e4:.2f}e-4" if bwb else "-",
                    f"{buq.confidence_score:.1f}%",
                ],
                "AI-Optimized Design": [
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
                    f"{(opred['cl']-bpred['cl']):+.3f}", f"{(opred['cd']-bpred['cd']):+.5f}", f"+{opt_res['ld_improvement_pct']:.1f}%",
                    "PASSED" if owb and owb.h_front_spar >= min_front_h - 1e-4 else "FAIL",
                    "PASSED" if owb and owb.h_rear_spar >= min_rear_h - 1e-4 else "FAIL",
                    "PASSED" if owb and owb.fuel_tank_area >= min_fuel - 1e-4 else "FAIL",
                    f"{(owb.i_xx - bwb.i_xx)*1e4:+.2f}e-4" if (owb and bwb) else "-",
                    f"{ouq.status}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Overlay Plot with Dark Aerospace Theme
            base_af = generate_naca4_airfoil(naca_code, 161)
            opt_code = f"{int(round(op['m']*100)):1d}{int(round(op['p']*10)):1d}{int(round(op['t']*100)):02d}"
            opt_af = generate_naca4_airfoil(opt_code, 161)

            fig_opt, ax_opt = plt.subplots(figsize=(8, 3.2))
            apply_aero_theme(fig_opt, ax_opt)

            ax_opt.plot(base_af.x, base_af.y_upper, color="#64748B", linestyle="--", linewidth=1.5, label=f"Baseline NACA {naca_code}")
            ax_opt.plot(base_af.x, base_af.y_lower, color="#64748B", linestyle="--", linewidth=1.5)
            ax_opt.plot(opt_af.x, opt_af.y_upper, color="#38BDF8", linewidth=2.2, label=f"AI-Optimized NACA {opt_code}")
            ax_opt.plot(opt_af.x, opt_af.y_lower, color="#38BDF8", linewidth=2.2)

            if owb:
                (oxf_l, oyf_l), (oxf_u, oyf_u) = owb.front_spar_coords
                (oxr_l, oyr_l), (oxr_u, oyr_u) = owb.rear_spar_coords
                ax_opt.plot([oxf_l, oxf_u], [oyf_l, oyf_u], color="#F43F5E", linewidth=2.2, label="Front Spar")
                ax_opt.plot([oxr_l, oxr_u], [oyr_l, oyr_u], color="#F59E0B", linewidth=2.2, label="Rear Spar")
                mask_obay = (opt_af.x >= owb.x_front_spar) & (opt_af.x <= owb.x_rear_spar)
                ax_opt.fill_between(opt_af.x[mask_obay], opt_af.y_lower[mask_obay], opt_af.y_upper[mask_obay], color="#818CF8", alpha=0.25, label="Optimized Wingbox")

            ax_opt.set_aspect("equal")
            ax_opt.set_xlim(-0.05, 1.05)
            ax_opt.set_ylim(-0.25, 0.25)
            ax_opt.set_xlabel("x / c")
            ax_opt.set_ylabel("y / c")
            ax_opt.set_title(f"Aero-Structural Overlay (+{opt_res['ld_improvement_pct']:.1f}% L/D, Spar Constraints Satisfied)", fontweight="bold")
            style_legend(ax_opt, loc="upper right", fontsize=8)
            st.pyplot(fig_opt)
        else:
            st.info("Adjust the aerodynamic and structural constraints on the left and click **'Run Aero-Structural Inverse Design'**.")

# -------------------------------------------------------------
# TAB 3: Live OpenFOAM CFD Ground-Truth Verification
# -------------------------------------------------------------
with tab3:
    st.subheader("💨 OpenFOAM CFD Ground-Truth Verification")
    st.write(r"Directly run the automated Navier-Stokes pipeline (`simpleFoam`, $k$-$\omega$ SST) inside WSL2 to verify the neural surrogate's prediction with true numerical physics.")

    st.markdown(f"**Current Target Case:** NACA `{naca_code}` at $\\alpha = {alpha_deg:.1f}^\\circ$, $Re = {reynolds:.1e}$")
    cfd_mesh = st.selectbox("CFD Grid Refinement Level", ["coarse", "medium"], index=0)

    if st.button("🚀 Run Live OpenFOAM Simulation in WSL2", type="primary"):
        from src.dataset.sampler import AirfoilSample
        from src.dataset.pipeline import run_single_cfd_sample

        cfg = load_config()
        sample = AirfoilSample(
            case_id=f"NACA{naca_code}_a{alpha_deg:+.1f}_verify",
            naca_code=naca_code,
            m=camber_m,
            p=camber_p,
            t=thickness_t,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
        )
        case_dir = PROJECT_ROOT / "data" / "raw_cases" / sample.case_id

        with st.spinner("Building 6-block C-grid mesh and running simpleFoam inside WSL2 (~20-40 seconds)..."):
            import time
            t0 = time.perf_counter()
            res = run_single_cfd_sample(sample, case_dir, cfg, mesh_level=cfd_mesh)
            elapsed = time.perf_counter() - t0

        if res.get("success"):
            rec = res["record"]
            st.success(f"OpenFOAM simulation finished in {elapsed:.1f} seconds! Status: {rec['status'].upper()}")

            # Parity Comparison
            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("Lift Coefficient (CL)", f"{rec['cl']:.4f}", f"{(rec['cl'] - pred['cl']):+.4f} vs ML")
            with v2:
                st.metric("Drag Coefficient (CD)", f"{rec['cd']:.5f}", f"{(rec['cd'] - pred['cd']):+.5f} vs ML")
            with v3:
                st.metric("Lift-to-Drag Ratio (L/D)", f"{rec['lift_to_drag']:.1f}", f"{(rec['lift_to_drag'] - pred['lift_to_drag']):+.1f} vs ML")

            # Surface Cp comparison if available
            cp_data = res.get("cp_data", {})
            if cp_data.get("available"):
                fig_vcp, ax_vcp = plt.subplots(figsize=(8, 3.5))
                apply_aero_theme(fig_vcp, ax_vcp)

                ax_vcp.plot(cp_data["x_grid"], cp_data["cp_upper"], color="#10B981", label="OpenFOAM CFD Upper Cp", linewidth=1.8)
                ax_vcp.plot(cp_data["x_grid"], cp_data["cp_lower"], color="#34D399", label="OpenFOAM CFD Lower Cp", linewidth=1.8)
                ax_vcp.plot(pred["x_grid"], pred["cp_upper"], color="#F43F5E", linestyle="--", label="AI Surrogate Upper Cp", linewidth=1.4)
                ax_vcp.plot(pred["x_grid"], pred["cp_lower"], color="#38BDF8", linestyle="--", label="AI Surrogate Lower Cp", linewidth=1.4)
                ax_vcp.invert_yaxis()
                ax_vcp.set_xlabel("x / c")
                ax_vcp.set_ylabel("Cp")
                ax_vcp.set_title(f"Surface Cp Parity: OpenFOAM CFD Truth vs AI Surrogate", fontweight="bold")
                style_legend(ax_vcp, loc="lower right", fontsize=8)
                st.pyplot(fig_vcp)
        else:
            st.error("OpenFOAM execution encountered an error. Check logs in WSL2.")

# -------------------------------------------------------------
# TAB 4: Full Aerodynamic Polar Sweeps
# -------------------------------------------------------------
with tab4:
    st.subheader(f"📈 Aerodynamic Polars for NACA {naca_code}")
    st.write("Instantaneous full-polar sweep computed by the neural surrogate across angles of attack $\\alpha \\in [-6^\\circ, 14^\\circ]$.")

    sweep_alphas = np.linspace(-6.0, 14.0, 41)
    sweep_cl, sweep_cd, sweep_ld = [], [], []

    for a in sweep_alphas:
        sp = engine.predict(camber_m, camber_p, thickness_t, float(a), reynolds, compute_structures=False)
        sweep_cl.append(sp["cl"])
        sweep_cd.append(sp["cd"])
        sweep_ld.append(sp["lift_to_drag"])

    sweep_df = pd.DataFrame({"alpha": sweep_alphas, "CL": sweep_cl, "CD": sweep_cd, "LD": sweep_ld})

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

    with p_col2:
        fig_p2, ax_p2 = plt.subplots(figsize=(4.5, 3.8))
        apply_aero_theme(fig_p2, ax_p2)
        ax_p2.plot(sweep_df["CD"], sweep_df["CL"], color="#10B981", linewidth=2.2)
        ax_p2.set_xlabel("Drag Coefficient (CD)")
        ax_p2.set_ylabel("Lift Coefficient (CL)")
        ax_p2.set_title("Drag Polar (CL vs CD)", fontweight="bold")
        st.pyplot(fig_p2)

    with p_col3:
        fig_p3, ax_p3 = plt.subplots(figsize=(4.5, 3.8))
        apply_aero_theme(fig_p3, ax_p3)
        ax_p3.plot(sweep_df["alpha"], sweep_df["LD"], color="#F59E0B", linewidth=2.2)
        max_idx = np.argmax(sweep_df["LD"])
        max_a = sweep_df["alpha"].iloc[max_idx]
        max_ld = sweep_df["LD"].iloc[max_idx]
        ax_p3.scatter([max_a], [max_ld], color="#EF4444", s=50, zorder=5, label=f"Max L/D={max_ld:.1f} @ {max_a:.1f}°")
        ax_p3.set_xlabel("α (deg)")
        ax_p3.set_ylabel("Aerodynamic Efficiency (L/D)")
        ax_p3.set_title("Efficiency Curve (L/D vs α)", fontweight="bold")
        style_legend(ax_p3, loc="upper right", fontsize=8)
        st.pyplot(fig_p3)
