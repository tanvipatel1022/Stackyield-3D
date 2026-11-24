
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import load_or_generate_wafers, optimize_chiplet_pairing

st.set_page_config(
    page_title="StackYield 3D | Chiplet Stacking Copilot",
    page_icon="🧊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 14px 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

wafers = load_or_generate_wafers()
pattern_names = list(wafers.keys())

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🧊 StackYield 3D")
    st.caption("AI Pre-Assembly Die-Pairing Copilot for 3D Chiplet Packaging")
    st.divider()

    wafer_a_key = st.selectbox("Tier 1: Logic Wafer Defect Pattern", pattern_names, index=0)
    wafer_b_key = st.selectbox("Tier 2: Memory (HBM) Wafer Pattern", pattern_names, index=1)

    view_mode = st.radio(
        "Wafer Map Visualization Layer",
        ["Hidden Neighbor + Thermal Risk Score", "Raw 2D Wafer Probe Map"],
        index=0,
    )

    st.divider()
    st.subheader("Fab & ESG Impact Assumptions")
    monthly_pairs = st.slider("Monthly Wafer Pairs Bonded", 100, 2000, 500, step=100)
    pkg_value = st.slider("3D AI Package Value ($)", 500, 6000, 2500, step=250)
    thermal_coupling = st.slider("Cross-Tier Thermal Coupling Penalty", 0.05, 0.40, 0.24, step=0.01)

# --- Run Optimization ---
results = optimize_chiplet_pairing(
    wafers[wafer_a_key],
    wafers[wafer_b_key],
    thermal_coupling=thermal_coupling,
    monthly_wafer_pairs=monthly_pairs,
    package_value_usd=pkg_value,
)

# --- Header ---
st.title("StackYield 3D: Known-Good-Stack Optimization Engine")
st.markdown(
    "Legacy 2D wafer tests pick **Known-Good-Dies** in isolation—missing borderline dies adjacent to defect clusters "
    "and overlapping thermal hotspots. **StackYield 3D** scores latent cross-tier risk and pairs compatible dies "
    "*before* irreversible 3D bonding."
)

# --- KPI Row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "First-Pass 3D Stack Yield",
    f"{results['optimal_yield'] * 100:.1f}%",
    f"+{results['yield_lift'] * 100:.2f}% vs. Standard Stacking",
)
c2.metric(
    "High-Reliability Stacks (≥85%)",
    f"{results['high_rel_opt']} / {results['n_pairs']}",
    f"+{results['high_rel_opt'] - results['high_rel_base']} extra stacks / wafer pair",
)
c3.metric(
    "Annual Scrap Cost Avoided",
    f"${results['annual_usd_saved']:,.0f}",
    f"+{results['extra_stacks_per_pair']:.1f} rescued AI packages / pair",
)
c4.metric(
    "Cleanroom Energy & GHG Saved",
    f"{results['annual_mwh_saved']:,.0f} MWh/yr",
    f"-{results['annual_co2_tons_saved']:,.0f} tons CO₂e | -{results['annual_upw_kgal_saved']:,.0f}k gal UPW",
)

st.divider()

# --- Side-by-Side Wafer Heatmaps ---
def build_wafer_figure(wafer_dict: dict, risk_dict: dict, title: str, show_risk: bool) -> go.Figure:
    if show_risk:
        z = risk_dict["latent_risk"].copy()
        z[wafer_dict["wafer_map"] == 2] = 1.05
        fig = px.imshow(
            z,
            color_continuous_scale=[
                [0.0, "#10b981"],   # Healthy green
                [0.45, "#f59e0b"],  # Marginal amber neighbor-risk
                [0.85, "#ef4444"],  # High latent risk red
                [1.0, "#7f1d1d"],   # Failed probe die
            ],
            zmin=0.0,
            zmax=1.05,
            title=title,
        )
        fig.update_coloraxes(colorbar_title="Risk")
    else:
        z = wafer_dict["wafer_map"].astype(float)
        z[z == 0] = np.nan
        fig = px.imshow(
            z,
            color_continuous_scale=[[0.0, "#10b981"], [1.0, "#ef4444"]],
            zmin=1.0,
            zmax=2.0,
            title=title,
        )
        fig.update_coloraxes(showscale=False)

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


show_risk_layer = view_mode == "Hidden Neighbor + Thermal Risk Score"
col_w1, col_w2, col_dist = st.columns([1, 1, 1.25])

with col_w1:
    st.plotly_chart(
        build_wafer_figure(
            wafers[wafer_a_key],
            results["res_a"],
            f"Tier 1: Logic Wafer ({wafer_a_key})",
            show_risk_layer,
        ),
        use_container_width=True,
    )

with col_w2:
    st.plotly_chart(
        build_wafer_figure(
            wafers[wafer_b_key],
            results["res_b"],
            f"Tier 2: Memory Wafer ({wafer_b_key})",
            show_risk_layer,
        ),
        use_container_width=True,
    )

with col_dist:
    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Histogram(
            x=results["baseline_probs"] * 100,
            name="Standard Stacking",
            marker_color="#6b7280",
            opacity=0.65,
            nbinsx=20,
        )
    )
    dist_fig.add_trace(
        go.Histogram(
            x=results["optimal_probs"] * 100,
            name="StackYield 3D Pairing",
            marker_color="#10b981",
            opacity=0.78,
            nbinsx=20,
        )
    )
    dist_fig.update_layout(
        title="3D Stack Survival Distribution (%)",
        barmode="overlay",
        template="plotly_dark",
        height=390,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Predicted Post-Bonding Survival (%)",
        yaxis_title="Chiplet Stacks",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(dist_fig, use_container_width=True)

# --- Inspectable Optimal Pairing Table ---
st.subheader("Top Recommended Known-Good-Stack (KGS) Die Pairings")
st.dataframe(results["top_pairs_df"], use_container_width=True, hide_index=True)
