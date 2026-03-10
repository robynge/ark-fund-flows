"""Advanced Analysis: Relative Performance, Asymmetry, Panel Regression"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES, PEER_ETF_NAMES, ALL_ETF_NAMES,
)
from analysis import (
    relative_performance_regression, relative_performance_all_etfs,
    asymmetry_regression, asymmetry_all_etfs,
    panel_regression, panel_regression_comparison,
)

st.set_page_config(page_title="Advanced Analysis", layout="wide")
st.title("Advanced Analysis: Relative Performance, Asymmetry & Panel")

# --- Sidebar ---
with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2,
    )
    lag_opts = {"D": [1, 5, 21], "W": [1, 4, 13], "ME": [1, 3, 6], "QE": [1, 2]}
    max_lag_default = {"D": 5, "W": 4, "ME": 3, "QE": 2}[freq]
    max_lag = st.slider("Max Lag", 1, 12, max_lag_default)
    lags = list(range(1, max_lag + 1))

    selected_etf = st.selectbox("Per-ETF View", ALL_ETF_NAMES, index=0)

    st.markdown("---")
    st.caption("35 ETFs: 9 ARK + 26 peers")


@st.cache_data(show_spinner="Loading 35 ETFs...")
def load_peer_data(freq):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full")


df = load_peer_data(freq)

if freq == "D":
    fc, rc = "Fund_Flow", "Return"
else:
    fc, rc = "Flow_Sum", "Return_Cum"

exc_col = "Excess_Return"

# Only keep ETFs that have fund flow data (non-null)
etfs_with_flows = df.groupby("ETF")[fc].apply(lambda x: x.notna().sum())
valid_etfs = etfs_with_flows[etfs_with_flows > 20].index.tolist()
df_valid = df[df["ETF"].isin(valid_etfs)].copy()

n_valid = len(valid_etfs)
n_ark = len([e for e in valid_etfs if e in ETF_NAMES])
n_peer = n_valid - n_ark
st.caption(f"ETFs with sufficient flow data: {n_valid} ({n_ark} ARK + {n_peer} peers)")

# ============================================================
# Section 1: Relative Performance
# ============================================================
st.header("1. Relative Performance: Absolute vs Excess Returns")
st.markdown("""
Do investors respond to **absolute** returns or **peer-relative** (excess) returns?
Benchmark = cross-sectional mean return of all 35 ETFs each period.
""")

with st.spinner("Running relative performance regressions..."):
    rp_summary = relative_performance_all_etfs(df_valid, fc, rc, exc_col, lags)

if len(rp_summary) > 0:
    # Add source flag
    rp_summary["Source"] = rp_summary["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Peer")

    # Bar chart: R² comparison
    rp_melt = rp_summary.melt(
        id_vars=["ETF", "Source"],
        value_vars=["R²_Absolute", "R²_Excess", "R²_Combined"],
        var_name="Model", value_name="R²",
    )
    fig_rp = px.bar(
        rp_melt, x="ETF", y="R²", color="Model", barmode="group",
        color_discrete_map={"R²_Absolute": "#1f77b4", "R²_Excess": "#ff7f0e",
                            "R²_Combined": "#2ca02c"},
        title="R² Comparison: Absolute vs Excess Return Models",
    )
    fig_rp.update_layout(height=450, xaxis_tickangle=-45,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_rp, use_container_width=True)

    # Summary table
    st.dataframe(
        rp_summary[["ETF", "Source", "R²_Absolute", "R²_Excess", "R²_Combined", "N"]]
        .sort_values("R²_Combined", ascending=False)
        .style.format({"R²_Absolute": "{:.4f}", "R²_Excess": "{:.4f}",
                        "R²_Combined": "{:.4f}"}),
        hide_index=True, use_container_width=True,
    )

    # Per-ETF detail
    with st.expander(f"Per-ETF Detail: {selected_etf}"):
        etf_df = df_valid[df_valid["ETF"] == selected_etf]
        if not etf_df[exc_col].dropna().empty:
            rp_detail = relative_performance_regression(
                etf_df, fc, rc, exc_col, lags)
            if rp_detail:
                cols = st.columns(3)
                for i, (name, label) in enumerate([
                    ("absolute", "Absolute Return"),
                    ("excess", "Excess Return"),
                    ("combined", "Combined"),
                ]):
                    with cols[i]:
                        st.markdown(f"**{label}** (R²={rp_detail[name]['r_squared']:.4f})")
                        st.dataframe(
                            rp_detail[name]["coefficients"]
                            [["Variable", "Coefficient", "p_value"]]
                            .style.format({"Coefficient": "{:.4f}", "p_value": "{:.4f}"}),
                            hide_index=True, use_container_width=True,
                        )

                # Scatter: excess return vs flow
                scatter_df = etf_df.dropna(subset=[fc, exc_col])
                if len(scatter_df) > 10:
                    fig_sc = px.scatter(
                        scatter_df, x=exc_col, y=fc,
                        trendline="ols", opacity=0.5,
                        title=f"{selected_etf}: Excess Return vs Flow",
                        labels={exc_col: "Excess Return", fc: "Flow ($M)"},
                    )
                    fig_sc.update_layout(height=350)
                    st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info(f"No excess return data for {selected_etf}.")
else:
    st.warning("Not enough data for relative performance analysis.")

# ============================================================
# Section 2: Asymmetry
# ============================================================
st.markdown("---")
st.header("2. Asymmetric Flow Response")
st.markdown("""
Do investors react more strongly to **positive** returns (performance chasing)
or **negative** returns (panic selling)?
Asymmetry ratio > 1 means stronger response to gains.
""")

with st.spinner("Running asymmetry regressions..."):
    asym_summary = asymmetry_all_etfs(df_valid, fc, rc, lags)

if len(asym_summary) > 0:
    asym_summary["Source"] = asym_summary["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Peer")

    # Grouped bar: β_pos vs |β_neg|
    asym_bar = asym_summary.copy()
    asym_bar["|β_neg|"] = asym_bar["Beta_Neg"].abs()

    fig_asym = go.Figure()
    fig_asym.add_trace(go.Bar(
        x=asym_bar["ETF"], y=asym_bar["Beta_Pos"],
        name="β_pos (gains)", marker_color="#2ca02c",
    ))
    fig_asym.add_trace(go.Bar(
        x=asym_bar["ETF"], y=asym_bar["|β_neg|"],
        name="|β_neg| (losses)", marker_color="#d62728",
    ))
    fig_asym.update_layout(
        barmode="group", height=450, title="Asymmetric Response: β_pos vs |β_neg|",
        xaxis_tickangle=-45, yaxis_title="Coefficient Magnitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_asym, use_container_width=True)

    # Summary table
    display_cols = ["ETF", "Source", "Beta_Pos", "Beta_Neg",
                    "Asymmetry_Ratio", "Wald_P", "R²", "N"]
    st.dataframe(
        asym_summary[display_cols]
        .sort_values("Asymmetry_Ratio", ascending=False)
        .style.format({
            "Beta_Pos": "{:.2f}", "Beta_Neg": "{:.2f}",
            "Asymmetry_Ratio": "{:.2f}", "Wald_P": "{:.4f}", "R²": "{:.4f}",
        }),
        hide_index=True, use_container_width=True,
    )

    # Per-ETF coefficient plot
    with st.expander(f"Per-ETF Detail: {selected_etf}"):
        etf_df = df_valid[df_valid["ETF"] == selected_etf]
        asym_detail = asymmetry_regression(etf_df, fc, rc, lags)
        if asym_detail:
            coef = asym_detail["coefficients"]
            coef = coef[coef["Variable"] != "const"]
            coef["CI_lower"] = coef["Coefficient"] - 1.96 * coef["Std_Error"]
            coef["CI_upper"] = coef["Coefficient"] + 1.96 * coef["Std_Error"]
            coef["Color"] = coef["Variable"].apply(
                lambda x: "#2ca02c" if "pos" in x else "#d62728")

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=coef["Variable"], y=coef["Coefficient"],
                marker_color=coef["Color"],
                error_y=dict(type="data",
                             symmetric=False,
                             array=coef["CI_upper"] - coef["Coefficient"],
                             arrayminus=coef["Coefficient"] - coef["CI_lower"]),
            ))
            fig_ci.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ci.update_layout(
                height=350, title=f"{selected_etf}: Asymmetric Coefficients with 95% CI",
                yaxis_title="Coefficient",
            )
            st.plotly_chart(fig_ci, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Asymmetry Ratio", f"{asym_detail['asymmetry_ratio']:.2f}")
            c2.metric("Wald p-value", f"{asym_detail['wald_p']:.4f}")
            c3.metric("R²", f"{asym_detail['r_squared']:.4f}")
else:
    st.warning("Not enough data for asymmetry analysis.")

# ============================================================
# Section 3: Panel Regression
# ============================================================
st.markdown("---")
st.header("3. Panel Regression (All ETFs)")
st.markdown("""
Fixed-effects panel regression across all ETFs with fund flow data.
Entity effects capture ETF-specific intercepts; clustered SEs account for within-ETF correlation.
""")

# Only use ETFs with both flows and excess returns for panel
panel_df = df_valid.dropna(subset=[fc, rc])

with st.spinner("Running panel regressions (5 specifications)..."):
    try:
        panel_comp = panel_regression_comparison(
            panel_df, fc, rc, excess_return_col=exc_col, lags=lags)

        if len(panel_comp) > 0:
            # Model comparison table
            st.subheader("Model Comparison")
            fmt_dict = {"R²_within": "{:.4f}", "R²_overall": "{:.4f}"}
            # Format coefficient columns
            for col in panel_comp.columns:
                if col.endswith("_coef"):
                    fmt_dict[col] = "{:.4f}"
                elif col.endswith("_pval"):
                    fmt_dict[col] = "{:.4f}"
            st.dataframe(
                panel_comp.style.format(fmt_dict, na_rep="—"),
                hide_index=True, use_container_width=True,
            )

            # Entity fixed effects
            st.subheader("Entity Fixed Effects")
            fe_result = panel_regression(
                panel_df, fc, rc, lags=lags,
                entity_effects=True, time_effects=False)
            if fe_result and "entity_effects" in fe_result:
                fe_df = fe_result["entity_effects"].sort_values("Fixed_Effect")
                fe_df["Source"] = fe_df["ETF"].apply(
                    lambda x: "ARK" if x in ETF_NAMES else "Peer")
                colors = fe_df["Source"].map({"ARK": "#1f77b4", "Peer": "#aec7e8"})

                fig_fe = go.Figure(go.Bar(
                    x=fe_df["ETF"], y=fe_df["Fixed_Effect"],
                    marker_color=colors,
                    text=fe_df["Source"],
                    hovertemplate="ETF: %{x}<br>α_i: %{y:.2f}<br>%{text}<extra></extra>",
                ))
                fig_fe.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_fe.update_layout(
                    height=400,
                    title="Entity Fixed Effects (α_i) — sorted",
                    yaxis_title="Fixed Effect (α_i)",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_fe, use_container_width=True)

            # Data coverage heatmap
            st.subheader("Data Coverage")
            coverage = panel_df.groupby(["ETF", pd.Grouper(key="Date", freq="YS")])[fc].count().reset_index()
            coverage.columns = ["ETF", "Year", "Observations"]
            coverage["Year"] = coverage["Year"].dt.year
            pivot = coverage.pivot(index="ETF", columns="Year", values="Observations").fillna(0)

            fig_cov = px.imshow(
                pivot, aspect="auto",
                color_continuous_scale="Blues",
                title="Observations per ETF per Year",
                labels=dict(color="Obs"),
            )
            fig_cov.update_layout(height=max(400, len(pivot) * 18))
            st.plotly_chart(fig_cov, use_container_width=True)
        else:
            st.warning("Panel regression returned no results.")

    except ImportError:
        st.error("Please install `linearmodels`: `pip install linearmodels>=6.0`")
    except Exception as e:
        st.error(f"Panel regression error: {e}")
