"""Model Comparison — select one model, compare across all 32 experiments."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")


@st.cache_data
def load_master():
    path = RESULTS_DIR / "master_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


master = load_master()
if master.empty:
    st.error("No results found.")
    st.stop()

# ------------------------------------------------------------------
# Model selector
# ------------------------------------------------------------------
model_names = list(master["model"].unique())
selected_model = st.selectbox("Select model", model_names)

model_rows = master[master["model"] == selected_model].copy()

# Description from config
MODEL_DESCRIPTIONS = {
    "univariate_r2_by_lag": "R² from OLS (flow ~ return_lag_k) for each lag k",
    "multilag_ols": "Multi-lag OLS: flow ~ Σ βₖ·return(t-k)",
    "multilag_ols_month_fe": "Multi-lag OLS + month dummies",
    "cross_correlation": "Pearson cross-correlation at lags -20..+20",
    "panel_pooled": "Pooled OLS (no fixed effects)",
    "panel_entity_fe": "Entity (ETF) fixed effects",
    "panel_entity_time_fe": "Entity + time fixed effects",
    "panel_entity_fe_excess": "Entity FE + excess return",
    "panel_entity_fe_controls": "Entity FE + volatility control",
    "asymmetry": "Piecewise regression: β⁺ vs β⁻",
    "relative_performance": "Absolute vs excess vs combined R² comparison",
    "granger": "Granger causality test (both directions)",
    "seasonality": "Average flow by calendar month",
    "drawdown": "Drawdown event study + flow regression",
}
st.caption(MODEL_DESCRIPTIONS.get(selected_model, ""))

st.divider()

# ------------------------------------------------------------------
# R² bar chart across experiments
# ------------------------------------------------------------------
st.subheader("R² across experiments")

exp_order = ["baseline"] + sorted(
    [e for e in model_rows["experiment_id"].unique() if e != "baseline"]
)
model_rows["experiment_id"] = pd.Categorical(
    model_rows["experiment_id"], categories=exp_order, ordered=True
)
model_rows = model_rows.sort_values("experiment_id")

r2_data = model_rows.dropna(subset=["r2"])
if not r2_data.empty:
    colors = ["#28a745" if gp else "#dc3545" for gp in r2_data["gate_pass"]]
    fig = go.Figure(
        go.Bar(
            x=r2_data["experiment_id"],
            y=r2_data["r2"],
            marker_color=colors,
            text=r2_data["r2"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=450,
        yaxis_title="R²",
        xaxis_tickangle=-45,
        title=f"{selected_model}: R² across experiments (green = gate pass)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# R² delta (bp) chart
# ------------------------------------------------------------------
delta_data = model_rows.dropna(subset=["r2_delta_bp"])
if not delta_data.empty and (delta_data["r2_delta_bp"] != 0).any():
    st.subheader("R² change vs baseline (basis points)")
    noise_only = delta_data[delta_data["experiment_id"] != "baseline"]
    if not noise_only.empty:
        colors_delta = [
            "#28a745" if v > 0 else "#dc3545" for v in noise_only["r2_delta_bp"]
        ]
        fig2 = go.Figure(
            go.Bar(
                x=noise_only["experiment_id"],
                y=noise_only["r2_delta_bp"],
                marker_color=colors_delta,
                text=noise_only["r2_delta_bp"].apply(lambda v: f"{v:+.1f}"),
                textposition="outside",
            )
        )
        fig2.update_layout(
            height=400,
            yaxis_title="Δ R² (bp)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Factor contribution analysis
# ------------------------------------------------------------------
st.subheader("Factor contribution analysis")
st.caption(
    "Average R² delta when each factor is included vs excluded "
    "— positive means the factor improves the model."
)

factors = ["A", "B", "C", "D", "E"]
factor_labels = {
    "A": "A (Macro events)",
    "B": "B (Market-wide flow)",
    "C": "C (VIX volatility)",
    "D": "D (Calendar dummies)",
    "E": "E (Peer aggregate flow)",
}

delta_rows = model_rows.dropna(subset=["r2_delta_bp"])
if not delta_rows.empty:
    contributions = {}
    for f in factors:
        with_f = delta_rows[delta_rows["factors"].str.contains(f, na=False)]
        without_f = delta_rows[~delta_rows["factors"].str.contains(f, na=False)]
        mean_with = with_f["r2_delta_bp"].mean() if not with_f.empty else 0
        mean_without = without_f["r2_delta_bp"].mean() if not without_f.empty else 0
        contributions[f] = mean_with - mean_without

    contrib_df = pd.DataFrame([
        {"Factor": factor_labels[f], "Marginal Δ (bp)": v}
        for f, v in contributions.items()
    ])
    contrib_df = contrib_df.sort_values("Marginal Δ (bp)", ascending=True)

    colors_c = [
        "#28a745" if v > 0 else "#dc3545" for v in contrib_df["Marginal Δ (bp)"]
    ]
    fig3 = go.Figure(
        go.Bar(
            x=contrib_df["Marginal Δ (bp)"],
            y=contrib_df["Factor"],
            orientation="h",
            marker_color=colors_c,
            text=contrib_df["Marginal Δ (bp)"].apply(lambda v: f"{v:+.1f}"),
            textposition="outside",
        )
    )
    fig3.update_layout(
        height=300,
        xaxis_title="Marginal Δ R² (bp)",
        title="Average marginal contribution of each noise factor",
        margin=dict(l=160),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Full results table for this model
# ------------------------------------------------------------------
st.subheader("Full results table")
display_cols = [
    "experiment_id", "factors", "r2", "beta_lag1", "beta_lag1_p",
    "gate_pass", "r2_delta_bp", "n_obs", "n_etfs",
]
available_cols = [c for c in display_cols if c in model_rows.columns]
st.dataframe(
    model_rows[available_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)
