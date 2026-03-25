"""Experiment Viewer — select one experiment, see all 14 models' detailed output."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"

st.set_page_config(page_title="Experiment Viewer", layout="wide")
st.title("Experiment Viewer")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_data
def load_master():
    path = RESULTS_DIR / "master_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_detail(experiment_id: str, model_name: str) -> pd.DataFrame | None:
    if experiment_id == "baseline":
        path = RESULTS_DIR / "baseline" / f"{model_name}.csv"
    else:
        path = RESULTS_DIR / "noise" / experiment_id / f"{model_name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def fmt_pval(p):
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def gate_badge(passed: bool):
    if passed:
        return ":green[PASS]"
    return ":red[FAIL]"


master = load_master()
if master.empty:
    st.error("No results found.")
    st.stop()

# ------------------------------------------------------------------
# Experiment selector
# ------------------------------------------------------------------
exp_ids = ["baseline"] + sorted(
    [e for e in master["experiment_id"].unique() if e != "baseline"]
)
selected_exp = st.selectbox("Select experiment", exp_ids)

exp_rows = master[master["experiment_id"] == selected_exp]
factors = exp_rows["factors"].iloc[0] if not exp_rows.empty else "(none)"
st.caption(f"Factors removed: **{factors}** · Freq: ME · Flow: raw · Benchmark: SPY")

st.divider()

# ------------------------------------------------------------------
# Summary bar chart: R² across models for this experiment
# ------------------------------------------------------------------
r2_data = exp_rows.dropna(subset=["r2"])[["model", "r2", "gate_pass"]].copy()
if not r2_data.empty:
    r2_data = r2_data.sort_values("r2", ascending=True)
    colors = ["#28a745" if gp else "#dc3545" for gp in r2_data["gate_pass"]]
    fig_bar = go.Figure(
        go.Bar(
            x=r2_data["r2"],
            y=r2_data["model"],
            orientation="h",
            marker_color=colors,
            text=r2_data["r2"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="R² by model (green = gate pass, red = fail)",
        height=400,
        xaxis_title="R²",
        margin=dict(l=180),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Detailed output per model — one expander each
# ------------------------------------------------------------------
st.subheader("Detailed model outputs")

for _, row in exp_rows.iterrows():
    model_name = row["model"]
    r2_val = row["r2"] if pd.notna(row["r2"]) else None
    p_val = row["beta_lag1_p"]
    gate = bool(row["gate_pass"])

    header = f"{model_name}  ·  R²={r2_val:.4f}" if r2_val else model_name
    header += f"  ·  {gate_badge(gate)}"

    with st.expander(header, expanded=False):
        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{r2_val:.4f}" if r2_val else "—")
        c2.metric("β₁", f"{row['beta_lag1']:.2f}" if pd.notna(row["beta_lag1"]) else "—")
        c3.metric("p-value", fmt_pval(p_val))
        c4.metric("N obs", f"{int(row['n_obs']):,}" if pd.notna(row["n_obs"]) else "—")

        # Extra JSON info
        extra = {}
        if pd.notna(row.get("extra_json")):
            try:
                extra = json.loads(row["extra_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        if extra:
            st.caption("Summary statistics")
            st.json(extra)

        # Detail CSV
        detail = load_detail(selected_exp, model_name)
        if detail is not None and not detail.empty:
            st.caption(f"Detail table ({len(detail)} rows)")

            # Model-specific visualizations
            if model_name == "univariate_r2_by_lag":
                # R² by lag heatmap (ETF × lag)
                if "lag" in detail.columns and "ETF" in detail.columns:
                    piv = detail.pivot_table(
                        index="ETF", columns="lag", values="r_squared"
                    )
                    fig = px.imshow(
                        piv,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title="R² by lag × ETF",
                    )
                    fig.update_layout(height=max(300, len(piv) * 20))
                    st.plotly_chart(fig, use_container_width=True)

            elif model_name == "cross_correlation":
                if "lag" in detail.columns and "ETF" in detail.columns:
                    piv = detail.pivot_table(
                        index="ETF", columns="lag", values="correlation"
                    )
                    fig = px.imshow(
                        piv,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        zmid=0,
                        title="Cross-correlation by lag × ETF",
                    )
                    fig.update_layout(height=max(300, len(piv) * 20))
                    st.plotly_chart(fig, use_container_width=True)

            elif model_name in ("multilag_ols", "multilag_ols_month_fe"):
                if "ETF" in detail.columns and "R²" in detail.columns:
                    sorted_df = detail.sort_values("R²", ascending=True)
                    fig = go.Figure(
                        go.Bar(
                            x=sorted_df["R²"],
                            y=sorted_df["ETF"],
                            orientation="h",
                            text=sorted_df["R²"].apply(lambda v: f"{v:.4f}"),
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        title="R² by ETF",
                        height=max(300, len(sorted_df) * 20),
                        margin=dict(l=100),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif model_name.startswith("panel_"):
                # Coefficient table
                if "Variable" in detail.columns:
                    st.caption("Coefficient table")

            elif model_name == "asymmetry":
                if "ETF" in detail.columns and "Beta_Pos" in detail.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="β⁺ (positive returns)",
                        x=detail["ETF"], y=detail["Beta_Pos"],
                    ))
                    fig.add_trace(go.Bar(
                        name="β⁻ (negative returns)",
                        x=detail["ETF"], y=detail["Beta_Neg"],
                    ))
                    fig.update_layout(
                        title="Asymmetry: β⁺ vs β⁻ by ETF",
                        barmode="group",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif model_name == "relative_performance":
                if "ETF" in detail.columns:
                    r2_cols = [c for c in detail.columns if c.startswith("R²")]
                    if r2_cols:
                        melted = detail.melt(
                            id_vars="ETF", value_vars=r2_cols,
                            var_name="Type", value_name="R²",
                        )
                        fig = px.bar(
                            melted, x="ETF", y="R²", color="Type",
                            barmode="group",
                            title="R² comparison: Absolute vs Excess vs Combined",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

            elif model_name == "granger":
                if "ETF" in detail.columns and "direction" in detail.columns:
                    ret_to_flow = detail[
                        detail["direction"] == "Returns → Flows"
                    ]
                    if not ret_to_flow.empty and "lag" in ret_to_flow.columns:
                        piv = ret_to_flow.pivot_table(
                            index="ETF", columns="lag", values="p_value"
                        )
                        fig = px.imshow(
                            piv,
                            aspect="auto",
                            color_continuous_scale="RdYlGn_r",
                            zmin=0, zmax=0.1,
                            title="Granger p-values (Returns → Flows)",
                        )
                        fig.update_layout(height=max(300, len(piv) * 20))
                        st.plotly_chart(fig, use_container_width=True)

            elif model_name == "seasonality":
                if "Month_Name" in detail.columns and "Mean" in detail.columns:
                    fig = go.Figure(
                        go.Bar(
                            x=detail["Month_Name"],
                            y=detail["Mean"],
                            marker_color=[
                                "#28a745" if v > 0 else "#dc3545"
                                for v in detail["Mean"]
                            ],
                        )
                    )
                    fig.update_layout(
                        title="Mean flow by calendar month",
                        yaxis_title="Mean flow",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif model_name == "drawdown":
                if "Horizon" in detail.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=detail["Horizon"],
                        y=detail["β_Depth"],
                        name="β_Depth",
                        error_y=dict(
                            type="constant",
                            value=0,
                        ),
                    ))
                    fig.update_layout(
                        title="Drawdown β by horizon",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Always show the raw table
            st.dataframe(detail, use_container_width=True, hide_index=True)
        else:
            st.info("No detail CSV available for this model.")
