"""Robustness Results — Specification curve, forest plot, and coefficient path
for noise-factor experiments."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import norm

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"

st.set_page_config(page_title="Robustness Results", layout="wide")
st.title("Robustness: Does Noise Removal Reveal the Signal?")

st.markdown("""
The baseline analysis shows a **weak return-flow relationship** in most per-ETF models.
Panel models (which pool across ETFs) detect significance, but individual ETF-level tests
often do not.

**Question:** Can we sharpen the signal by removing known confounders?
We systematically remove 5 noise factors (A-E) in all 31 non-empty combinations
and re-run 14 statistical models each time.

| Factor | What it removes |
|--------|----------------|
| **A** - Macro events | Periods around FOMC, CPI, NFP releases |
| **B** - Market-wide flow | Cross-sectional mean flow on each date |
| **C** - VIX volatility | High-VIX regimes (> 80th percentile) |
| **D** - Calendar dummies | Month-of-year seasonal patterns |
| **E** - Peer aggregate flow | Peer-group average flow |

**Gate criterion:** $\\beta_1 > 0$ *and* $p < 0.05$.
""")

st.divider()


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@st.cache_data
def load_master():
    path = RESULTS_DIR / "master_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


master = load_master()
if master.empty:
    st.error("No results found. Run `python -m experiments.runner` first.")
    st.stop()

FACTORS = ["A", "B", "C", "D", "E"]
FACTOR_LABELS = {
    "A": "Macro events", "B": "Market-wide flow", "C": "VIX volatility",
    "D": "Calendar dummies", "E": "Peer aggregate flow",
}
for f in FACTORS:
    master[f"has_{f}"] = master["factors"].str.contains(f, na=False)
master["n_factors"] = master["factors"].apply(
    lambda x: 0 if x == "(none)" else len([c for c in x if c in "ABCDE"])
)

# Summary metrics
n_experiments = master["experiment_id"].nunique()
n_models = master["model"].nunique()
n_total = len(master)
n_pass = int(master["gate_pass"].sum())
pass_rate = n_pass / n_total * 100 if n_total else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments", n_experiments)
col2.metric("Models", n_models)
col3.metric("Total tests", n_total)
col4.metric("Gate pass", f"{n_pass} ({pass_rate:.1f}%)")

st.divider()

# Model selector
model_names = sorted(master["model"].unique())
default_idx = model_names.index("panel_entity_fe") if "panel_entity_fe" in model_names else 0
selected_model = st.selectbox("Select model for detailed robustness plots", model_names, index=default_idx)

model_df = master[master["model"] == selected_model].copy()
bl_rows = model_df[model_df["experiment_id"] == "baseline"]
baseline_row = bl_rows.iloc[0] if len(bl_rows) > 0 else None

st.divider()


# ==================================================================
# 1. SPECIFICATION CURVE
# ==================================================================
st.subheader("1. Specification Curve")
st.caption(
    "Each column = one experiment. **Top**: $\\beta_1$ estimate "
    "(green = gate pass, red = fail, dashed = baseline). "
    "**Bottom**: which noise factors are removed (blue = removed)."
)

spec_df = model_df.dropna(subset=["beta_lag1"]).copy()
if len(spec_df) > 0:
    spec_df = spec_df.sort_values("beta_lag1", ascending=True).reset_index(drop=True)
    spec_df["x"] = range(len(spec_df))
    colors = ["#28a745" if gp else "#dc3545" for gp in spec_df["gate_pass"]]

    fig_spec = make_subplots(
        rows=2, cols=1, row_heights=[0.65, 0.35],
        shared_xaxes=True, vertical_spacing=0.02,
    )

    fig_spec.add_trace(go.Bar(
        x=spec_df["x"], y=spec_df["beta_lag1"],
        marker_color=colors, opacity=0.85,
        hovertemplate=(
            "Experiment: %{customdata[0]}<br>"
            "beta1: %{y:.2f}<br>"
            "p: %{customdata[1]:.4f}<br>"
            "R2: %{customdata[2]:.4f}<extra></extra>"
        ),
        customdata=spec_df[["experiment_id", "beta_lag1_p", "r2"]].values,
    ), row=1, col=1)

    if baseline_row is not None and pd.notna(baseline_row["beta_lag1"]):
        fig_spec.add_hline(
            y=baseline_row["beta_lag1"], line_dash="dash",
            line_color="#1f77b4", opacity=0.6, row=1, col=1,
            annotation_text="baseline", annotation_position="top right",
        )
    fig_spec.add_hline(y=0, line_color="gray", opacity=0.3, row=1, col=1)

    for i, f in enumerate(FACTORS):
        marker_colors = ["#1f77b4" if has else "#f0f0f0" for has in spec_df[f"has_{f}"]]
        fig_spec.add_trace(go.Scatter(
            x=spec_df["x"], y=[i] * len(spec_df),
            mode="markers",
            marker=dict(symbol="square", size=10, color=marker_colors,
                        line=dict(width=0.5, color="#ccc")),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)

    fig_spec.update_yaxes(title_text="beta_1", row=1, col=1)
    fig_spec.update_yaxes(
        tickvals=list(range(len(FACTORS))),
        ticktext=[f"{f} ({FACTOR_LABELS[f]})" for f in FACTORS],
        row=2, col=1,
    )
    fig_spec.update_xaxes(showticklabels=False)
    fig_spec.update_layout(
        height=550, showlegend=False,
        margin=dict(l=160, r=30, t=30, b=20),
    )
    st.plotly_chart(fig_spec, use_container_width=True)
else:
    st.info("No data available for this model.")


# ==================================================================
# 2. FOREST PLOT
# ==================================================================
st.divider()
st.subheader("2. Forest Plot")
st.caption(
    "$\\beta_1$ point estimate with approximate 95% CI for each experiment, "
    "sorted by number of factors removed. Vertical dashed line = zero."
)

forest_df = model_df.dropna(subset=["beta_lag1", "beta_lag1_p"]).copy()
if len(forest_df) > 0:
    forest_df["z_score"] = forest_df["beta_lag1_p"].clip(1e-10, 1).apply(
        lambda p: norm.ppf(1 - p / 2)
    )
    forest_df["se"] = (forest_df["beta_lag1"].abs() / forest_df["z_score"]).replace(
        [np.inf, -np.inf], np.nan
    )
    forest_df["ci_lo"] = forest_df["beta_lag1"] - 1.96 * forest_df["se"]
    forest_df["ci_hi"] = forest_df["beta_lag1"] + 1.96 * forest_df["se"]

    forest_df["sort_key"] = forest_df.apply(
        lambda r: (0, "") if r["experiment_id"] == "baseline" else (r["n_factors"], r["experiment_id"]),
        axis=1,
    )
    forest_df = forest_df.sort_values("sort_key", ascending=False).reset_index(drop=True)

    colors_f = ["#28a745" if gp else "#dc3545" for gp in forest_df["gate_pass"]]
    labels = [
        f"{'>> ' if gp else ''}{eid}"
        for eid, gp in zip(forest_df["experiment_id"], forest_df["gate_pass"])
    ]

    fig_forest = go.Figure()
    fig_forest.add_trace(go.Scatter(
        x=forest_df["beta_lag1"], y=labels,
        mode="markers",
        marker=dict(size=7, color=colors_f),
        error_x=dict(
            type="data", symmetric=False,
            array=(forest_df["ci_hi"] - forest_df["beta_lag1"]).tolist(),
            arrayminus=(forest_df["beta_lag1"] - forest_df["ci_lo"]).tolist(),
            color="#999", thickness=1,
        ),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "beta1: %{x:.2f}, p: %{customdata[1]:.4f}<extra></extra>"
        ),
        customdata=forest_df[["experiment_id", "beta_lag1_p"]].values,
    ))
    fig_forest.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_forest.update_layout(
        height=max(400, len(forest_df) * 22),
        xaxis_title="beta_1",
        showlegend=False,
        margin=dict(l=160, r=30, t=30, b=40),
    )
    st.plotly_chart(fig_forest, use_container_width=True)
else:
    st.info("No data available for this model.")


# ==================================================================
# 3. COEFFICIENT PATH (cumulative noise removal)
# ==================================================================
st.divider()
st.subheader("3. Coefficient Path")
st.caption(
    "Cumulative noise removal: starting from baseline, we progressively remove "
    "factors A through E and track how $\\beta_1$ and $R^2$ respond. "
    "An upward trend means removing that noise reveals more signal."
)

path_ids = ["baseline", "N-A", "N-AB", "N-ABC", "N-ABCD", "N-ABCDE"]
path_labels = ["Baseline", "+Remove A", "+B", "+C", "+D", "+E"]
path_df = model_df[model_df["experiment_id"].isin(path_ids)].copy()
path_df["experiment_id"] = pd.Categorical(path_df["experiment_id"], categories=path_ids, ordered=True)
path_df = path_df.sort_values("experiment_id")

if len(path_df) > 1:
    fig_path = make_subplots(rows=1, cols=2, subplot_titles=["beta_1 (coefficient)", "R-squared"])

    path_colors = ["#28a745" if gp else "#dc3545" for gp in path_df["gate_pass"]]
    x_labs = [path_labels[path_ids.index(eid)] for eid in path_df["experiment_id"]]

    fig_path.add_trace(go.Scatter(
        x=x_labs, y=path_df["beta_lag1"],
        mode="lines+markers",
        marker=dict(size=10, color=path_colors),
        line=dict(color="#666", width=2), name="beta_1",
    ), row=1, col=1)
    fig_path.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=1, col=1)

    r2_valid = path_df.dropna(subset=["r2"])
    if len(r2_valid) > 0:
        x_labs_r2 = [path_labels[path_ids.index(eid)] for eid in r2_valid["experiment_id"]]
        r2_colors = ["#28a745" if gp else "#dc3545" for gp in r2_valid["gate_pass"]]
        fig_path.add_trace(go.Scatter(
            x=x_labs_r2, y=r2_valid["r2"],
            mode="lines+markers",
            marker=dict(size=10, color=r2_colors),
            line=dict(color="#666", width=2), name="R2",
        ), row=1, col=2)

    fig_path.update_layout(height=380, showlegend=False, margin=dict(l=60, r=30, t=40, b=30))
    st.plotly_chart(fig_path, use_container_width=True)
else:
    st.info("Not enough data for coefficient path.")


# ==================================================================
# 4. FACTOR CONTRIBUTION
# ==================================================================
st.divider()
st.subheader("4. Factor Contribution")
st.caption(
    "Marginal $\\Delta R^2$ (basis points) when each factor is removed vs not removed, "
    "averaged across all experiment configurations. "
    "Positive = removing this noise source improves the model."
)

delta_rows = model_df.dropna(subset=["r2_delta_bp"])
if not delta_rows.empty:
    contributions = {}
    for f in FACTORS:
        with_f = delta_rows[delta_rows[f"has_{f}"]]
        without_f = delta_rows[~delta_rows[f"has_{f}"]]
        mean_with = with_f["r2_delta_bp"].mean() if not with_f.empty else 0
        mean_without = without_f["r2_delta_bp"].mean() if not without_f.empty else 0
        contributions[f] = mean_with - mean_without

    contrib_df = pd.DataFrame([
        {"Factor": f"{f} ({FACTOR_LABELS[f]})", "delta": v}
        for f, v in contributions.items()
    ]).sort_values("delta", ascending=True)

    colors_c = ["#28a745" if v > 0 else "#dc3545" for v in contrib_df["delta"]]
    fig_contrib = go.Figure(go.Bar(
        x=contrib_df["delta"], y=contrib_df["Factor"],
        orientation="h", marker_color=colors_c,
        text=contrib_df["delta"].apply(lambda v: f"{v:+.1f}"),
        textposition="outside",
    ))
    fig_contrib.update_layout(
        height=280, xaxis_title="Marginal delta R2 (bp)",
        margin=dict(l=180, r=60, t=20, b=30),
    )
    fig_contrib.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    st.plotly_chart(fig_contrib, use_container_width=True)


# ==================================================================
# 5. FULL HEATMAP
# ==================================================================
st.divider()
st.subheader("5. Results Matrix")

metric_choice = st.radio(
    "Color metric", ["Gate pass", "R2", "R2 delta (bp)", "p-value"], horizontal=True,
)

exp_order = ["baseline"] + sorted(
    [e for e in master["experiment_id"].unique() if e != "baseline"]
)
model_order = sorted(master["model"].unique())
pivot_kw = dict(index="experiment_id", columns="model")

if metric_choice == "Gate pass":
    piv = master.pivot_table(values="gate_pass", **pivot_kw, aggfunc="max")
    piv = piv.reindex(index=exp_order, columns=model_order).fillna(0).astype(int)
    colorscale = [[0, "#f8d7da"], [1, "#d4edda"]]
    zmin, zmax = 0, 1
    htmpl = "Experiment: %{y}<br>Model: %{x}<br>Pass: %{z}<extra></extra>"
elif metric_choice == "R2":
    piv = master.pivot_table(values="r2", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "Blues"
    zmin, zmax = 0, piv.max().max() if not piv.empty else 0.1
    htmpl = "Experiment: %{y}<br>Model: %{x}<br>R2: %{z:.4f}<extra></extra>"
elif metric_choice == "R2 delta (bp)":
    piv = master.pivot_table(values="r2_delta_bp", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "RdBu"
    abs_max = max(abs(piv.min().min()), abs(piv.max().max())) if not piv.empty else 100
    zmin, zmax = -abs_max, abs_max
    htmpl = "Experiment: %{y}<br>Model: %{x}<br>delta bp: %{z:.1f}<extra></extra>"
else:
    piv = master.pivot_table(values="beta_lag1_p", **pivot_kw, aggfunc="mean")
    piv = piv.reindex(index=exp_order, columns=model_order)
    colorscale = "RdYlGn_r"
    zmin, zmax = 0, 1
    htmpl = "Experiment: %{y}<br>Model: %{x}<br>p: %{z:.4f}<extra></extra>"

fig_hm = go.Figure(go.Heatmap(
    z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
    colorscale=colorscale, zmin=zmin, zmax=zmax, hovertemplate=htmpl,
))
fig_hm.update_layout(
    height=max(400, len(exp_order) * 22),
    xaxis=dict(side="top", tickangle=-45),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=120, t=80, r=20, b=20),
)
st.plotly_chart(fig_hm, use_container_width=True)
