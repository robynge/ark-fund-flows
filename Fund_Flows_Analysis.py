"""ETF Performance Chasing — Experiment Results Dashboard

Overview page: research summary + results heatmap (experiments × models).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "experiments" / "results"

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")


# ------------------------------------------------------------------
# Load master results
# ------------------------------------------------------------------
@st.cache_data
def load_master():
    path = RESULTS_DIR / "master_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


master = load_master()

if master.empty:
    st.error("No results found. Run `python -m experiments.runner` first.")
    st.stop()

# ------------------------------------------------------------------
# Research context
# ------------------------------------------------------------------
st.markdown("""
**Research question:** Do ETF investors chase past performance — i.e., do prior
returns predict subsequent fund flows?

**Method:** We test this across **14 statistical models** and **32 experiment
configurations** (1 baseline + 31 noise-factor combinations). Each noise factor
removes a potential confounder from the data:

| Factor | Removes |
|--------|---------|
| **A** — Macro events | Periods around FOMC, CPI, NFP releases |
| **B** — Market-wide flow | Cross-sectional mean flow on each date |
| **C** — VIX volatility | High-VIX regimes (> 80th percentile) |
| **D** — Calendar dummies | Month-of-year seasonal patterns |
| **E** — Peer aggregate flow | Peer-group average flow |

**Gate criterion:** β₁ > 0 *and* p < 0.05 — a positive, statistically
significant return→flow relationship.
""")

st.divider()

# ------------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------------
n_experiments = master["experiment_id"].nunique()
n_models = master["model"].nunique()
n_total = len(master)
n_pass = int(master["gate_pass"].sum())
pass_rate = n_pass / n_total * 100 if n_total else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments", n_experiments)
col2.metric("Models", n_models)
col3.metric("Total runs", n_total)
col4.metric("Gate pass", f"{n_pass} ({pass_rate:.1f}%)")

st.divider()

# ------------------------------------------------------------------
# Heatmap: experiments × models
# ------------------------------------------------------------------
st.subheader("Results Matrix")

metric_choice = st.radio(
    "Color metric",
    ["Gate pass", "R²", "R² delta (bp)", "β₁ p-value"],
    horizontal=True,
)

# Build experiment order: baseline first, then noise combos sorted
exp_order = ["baseline"] + sorted(
    [e for e in master["experiment_id"].unique() if e != "baseline"]
)
model_order = list(master["model"].unique())

pivot_kw = dict(index="experiment_id", columns="model")

if metric_choice == "Gate pass":
    piv = master.pivot_table(values="gate_pass", **pivot_kw, aggfunc="max")
    piv = piv.reindex(index=exp_order, columns=model_order).fillna(0).astype(int)
    colorscale = [[0, "#f8d7da"], [1, "#d4edda"]]
    zmin, zmax = 0, 1
    fmt = ".0f"
    hover_tmpl = "Experiment: %{y}<br>Model: %{x}<br>Pass: %{z}<extra></extra>"
elif metric_choice == "R²":
    piv = master.pivot_table(values="r2", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "Blues"
    zmin, zmax = 0, piv.max().max() if not piv.empty else 0.1
    fmt = ".4f"
    hover_tmpl = "Experiment: %{y}<br>Model: %{x}<br>R²: %{z:.4f}<extra></extra>"
elif metric_choice == "R² delta (bp)":
    piv = master.pivot_table(values="r2_delta_bp", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "RdBu"
    abs_max = max(abs(piv.min().min()), abs(piv.max().max())) if not piv.empty else 100
    zmin, zmax = -abs_max, abs_max
    fmt = ".1f"
    hover_tmpl = "Experiment: %{y}<br>Model: %{x}<br>Δbp: %{z:.1f}<extra></extra>"
else:  # p-value
    piv = master.pivot_table(values="beta_lag1_p", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "RdYlGn_r"
    zmin, zmax = 0, 1
    fmt = ".4f"
    hover_tmpl = "Experiment: %{y}<br>Model: %{x}<br>p: %{z:.4f}<extra></extra>"

fig = go.Figure(
    go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=piv.index.tolist(),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hovertemplate=hover_tmpl,
    )
)
fig.update_layout(
    height=max(400, len(exp_order) * 22),
    xaxis=dict(side="top", tickangle=-45),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=120, t=80, r=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Top results table
# ------------------------------------------------------------------
st.subheader("Top 20 results by R²")
top = (
    master.dropna(subset=["r2"])
    .nlargest(20, "r2")[
        ["experiment_id", "model", "r2", "beta_lag1", "beta_lag1_p",
         "gate_pass", "r2_delta_bp", "n_obs", "n_etfs"]
    ]
    .reset_index(drop=True)
)
st.dataframe(top, use_container_width=True, hide_index=True)
