"""Robustness Results — Specification curve, forest plot, coefficient path,
GMM comparison, bootstrap CI, and panel diagnostics."""
import json
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


# ==================================================================
# 6. GMM COMPARISON
# ==================================================================
st.divider()
st.subheader("6. GMM Model Comparison")
st.caption(
    "Arellano-Bond (AB) and Blundell-Bond (BB) GMM results vs standard FE. "
    "GMM addresses Nickell bias from including lagged dependent variable."
)

gmm_models = ["panel_entity_fe", "panel_gmm_ab", "panel_gmm_bb", "panel_entity_fe_trend"]
gmm_df = master[
    (master["model"].isin(gmm_models)) & (master["experiment_id"] == "baseline")
].copy()

if not gmm_df.empty:
    # Parse GMM diagnostics from extra_json
    gmm_rows = []
    for _, row in gmm_df.iterrows():
        r = {
            "Model": row["model"],
            "Freq": row.get("freq", ""),
            "beta_lag1": row["beta_lag1"],
            "p_value": row["beta_lag1_p"],
            "R2": row["r2"],
            "N": row["n_obs"],
        }
        try:
            extra = json.loads(row.get("extra_json", "{}"))
            r["Sargan_p"] = extra.get("sargan_p", np.nan)
            r["AR1_p"] = extra.get("ar1_p", np.nan)
            r["AR2_p"] = extra.get("ar2_p", np.nan)
        except (json.JSONDecodeError, TypeError):
            pass
        gmm_rows.append(r)

    gmm_table = pd.DataFrame(gmm_rows)

    # Display table
    st.dataframe(
        gmm_table.style.format({
            "beta_lag1": "{:.2f}",
            "p_value": "{:.4f}",
            "R2": "{:.4f}",
            "Sargan_p": "{:.4f}",
            "AR1_p": "{:.4f}",
            "AR2_p": "{:.4f}",
        }, na_rep="-"),
        use_container_width=True,
    )

    # Bar chart comparing betas
    fig_gmm = go.Figure()
    for _, row in gmm_table.iterrows():
        color = "#28a745" if pd.notna(row["p_value"]) and row["p_value"] < 0.05 else "#dc3545"
        fig_gmm.add_trace(go.Bar(
            x=[f"{row['Model']}\n({row['Freq']})"],
            y=[row["beta_lag1"]],
            marker_color=color,
            text=f"p={row['p_value']:.3f}" if pd.notna(row["p_value"]) else "",
            textposition="outside",
            showlegend=False,
        ))
    fig_gmm.update_layout(
        height=350, yaxis_title="beta(Return_lag1)",
        margin=dict(l=60, r=30, t=30, b=80),
    )
    fig_gmm.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    st.plotly_chart(fig_gmm, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - **Sargan test** (p > 0.05): instruments are valid (overidentification not rejected)
    - **AR(1)** (p < 0.05): expected — first-order serial correlation in differences
    - **AR(2)** (p > 0.05): no second-order serial correlation — GMM assumptions hold
    - With only 9 ETFs, GMM may be unstable; treat as supplementary evidence
    """)
else:
    st.info("No GMM results available. Run experiments with panel_gmm_ab/bb models.")


# ==================================================================
# 7. BOOTSTRAP CI VISUALIZATION
# ==================================================================
st.divider()
st.subheader("7. Cluster Bootstrap Confidence Intervals")
st.caption(
    "With only 9 clusters (ETFs), asymptotic cluster SE may be unreliable. "
    "Pairs cluster bootstrap resamples entire ETFs to construct robust CIs."
)

# Check if bootstrap results exist in any detail CSV
boot_results = []
baseline_dir = RESULTS_DIR / "baseline"
for csv_path in sorted(RESULTS_DIR.glob("**/panel_entity_fe.csv")):
    try:
        detail = pd.read_csv(csv_path)
        if "_SUMMARY_" in detail.get("Variable", pd.Series()).values:
            summary = detail[detail["Variable"] == "_SUMMARY_"].iloc[0]
            boot_results.append(summary.to_dict())
    except Exception:
        pass

# Show bootstrap info from r_engine if available
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from r_engine import R_AVAILABLE as _r_ok
except ImportError:
    _r_ok = False

if _r_ok:
    st.markdown(
        "Bootstrap validation is available via `r_engine.cluster_bootstrap()`. "
        "Run on your top specification to verify cluster SE."
    )

    if st.button("Run bootstrap on baseline Entity FE (ME/raw/SPY)"):
        with st.spinner("Running 199 bootstrap replications..."):
            try:
                from r_engine import cluster_bootstrap
                from data_loader import get_prepared_data_with_peers
                df_boot = get_prepared_data_with_peers(freq="ME", zscore_type="full", benchmark="SPY")
                boot = cluster_bootstrap(df_boot, "Flow_Sum", "Return_Cum", lags=[1], n_boot=199)
                if boot:
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("Original beta", f"{boot['original_beta']:.2f}")
                    col_b2.metric("Bootstrap SE", f"{boot['boot_se']:.2f}")
                    col_b3.metric("Bootstrap p", f"{boot['boot_p']:.4f}")

                    ci_lo, ci_hi = boot["ci_percentile"]
                    st.markdown(f"**95% Percentile CI:** [{ci_lo:.2f}, {ci_hi:.2f}]")

                    fig_boot = go.Figure()
                    fig_boot.add_shape(type="rect", x0=ci_lo, x1=ci_hi, y0=0, y1=1,
                                       fillcolor="lightblue", opacity=0.3, line_width=0)
                    fig_boot.add_vline(x=boot["original_beta"], line_color="blue", line_width=2)
                    fig_boot.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_boot.update_layout(
                        height=200, xaxis_title="beta(Return_lag1)",
                        yaxis=dict(visible=False),
                        margin=dict(l=60, r=30, t=20, b=40),
                        annotations=[
                            dict(x=boot["original_beta"], y=0.8, text="Point estimate",
                                 showarrow=False, font=dict(size=11)),
                            dict(x=(ci_lo + ci_hi) / 2, y=0.3, text="95% CI",
                                 showarrow=False, font=dict(size=11, color="steelblue")),
                        ]
                    )
                    st.plotly_chart(fig_boot, use_container_width=True)
                else:
                    st.warning("Bootstrap returned no results.")
            except Exception as e:
                st.error(f"Bootstrap failed: {e}")
else:
    st.info("R not available. Install rpy2 + R to enable bootstrap validation.")


# ==================================================================
# 8. PANEL DIAGNOSTICS SUMMARY
# ==================================================================
st.divider()
st.subheader("8. Panel Diagnostics Summary")
st.caption(
    "Key diagnostic tests that guide standard error selection. "
    "Run via `r_engine.diagnostic_tests()` on different frequencies."
)

if _r_ok:
    if st.button("Run diagnostics (ME/raw/SPY)"):
        with st.spinner("Running 6 panel diagnostic tests..."):
            try:
                from r_engine import diagnostic_tests, variance_decomposition
                from data_loader import get_prepared_data_with_peers
                df_diag = get_prepared_data_with_peers(freq="ME", zscore_type="full", benchmark="SPY")
                diag = diagnostic_tests(df_diag, "Flow_Sum", "Return_Cum", lags=[1])

                diag_rows = []
                test_labels = {
                    "hausman": ("Hausman (FE vs RE)", "p < 0.05 => Use FE"),
                    "bp_lm_test": ("BP LM (Pooled vs Panel)", "p < 0.05 => Panel effects exist"),
                    "f_test_individual_effects": ("F-test (individual effects)", "p < 0.05 => Individual effects"),
                    "serial_correlation": ("Breusch-Godfrey (serial corr)", "p < 0.05 => Serial correlation"),
                    "cross_sectional_dependence": ("Pesaran CD (cross-section dep)", "p < 0.05 => CSD exists"),
                    "breusch_pagan": ("Breusch-Pagan (heterosked)", "p < 0.05 => Heteroskedastic"),
                }
                for key, (label, rule) in test_labels.items():
                    if key in diag:
                        stat = diag[key]["statistic"]
                        pval = diag[key]["p_value"]
                        sig = pval < 0.05 if pd.notna(pval) else None
                        diag_rows.append({
                            "Test": label,
                            "Statistic": stat,
                            "p-value": pval,
                            "Significant": "Yes" if sig else ("No" if sig is not None else "-"),
                            "Implication": rule,
                        })

                diag_table = pd.DataFrame(diag_rows)
                st.dataframe(
                    diag_table.style.format({"Statistic": "{:.4f}", "p-value": "{:.6f}"}, na_rep="-"),
                    use_container_width=True,
                )

                # SE recommendation
                sc = diag.get("serial_correlation", {}).get("p_value", 1)
                csd = diag.get("cross_sectional_dependence", {}).get("p_value", 1)
                het = diag.get("breusch_pagan", {}).get("p_value", 1)

                if sc < 0.05 and csd < 0.05:
                    st.success("Recommended SE: **Driscoll-Kraay** (serial correlation + CSD detected)")
                elif sc < 0.05 and het < 0.05:
                    st.warning("Recommended SE: **Clustered by ETF** (serial correlation + heteroskedasticity)")
                else:
                    st.info("Recommended SE: **Clustered by ETF** (default robust choice)")
                st.markdown("*Note: With only 9 clusters, wild cluster bootstrap is recommended for verification.*")

                # Variance decomposition
                vd = variance_decomposition(df_diag, ["Flow_Sum", "Return_Cum"])
                if not vd.empty:
                    st.markdown("**Between/Within Variance Decomposition:**")
                    st.dataframe(
                        vd.style.format({
                            "overall_sd": "{:.4f}", "between_sd": "{:.4f}",
                            "within_sd": "{:.4f}", "within_pct": "{:.1f}%",
                        }),
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Diagnostics failed: {e}")
else:
    st.info("R not available. Install rpy2 + R to enable panel diagnostics.")
