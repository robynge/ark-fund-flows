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
    auto_lags, r_squared_by_lag,
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
    selected_etf = st.selectbox("Per-ETF View", ALL_ETF_NAMES, index=0)

    st.markdown("---")
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading 38 ETFs...")
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
**What we're testing:** When investors decide to buy/sell an ETF, do they look at the fund's
own return ("it went up 10%") or how it performed *relative to peers* ("it beat the group average by 5%")?

We run three regressions for each ETF — predicting fund flows using lagged returns:
- **R² Absolute** — only uses the ETF's own return → how much of flow variation is explained by "did it go up or down?"
- **R² Excess** — only uses return minus the peer-group average → how much is explained by "did it beat peers?"
- **R² Combined** — uses both together → which matters more when we include both?

If R² Excess > R² Absolute, investors care more about **relative ranking** than raw performance.
""")


with st.spinner("Running relative performance regressions..."):
    rp_summary = relative_performance_all_etfs(df_valid, fc, rc, exc_col)

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
    st.plotly_chart(fig_rp, width="stretch")

    # Summary table
    st.dataframe(
        rp_summary[["ETF", "Source", "R²_Absolute", "R²_Excess", "R²_Combined", "N"]]
        .sort_values("R²_Combined", ascending=False)
        .style.format({"R²_Absolute": "{:.4f}", "R²_Excess": "{:.4f}",
                        "R²_Combined": "{:.4f}"}),
        hide_index=True, width="stretch",
    )

    # Per-ETF detail
    with st.expander(f"Per-ETF Detail: {selected_etf}"):
        etf_df = df_valid[df_valid["ETF"] == selected_etf]
        if not etf_df[exc_col].dropna().empty:
            # R² by lag profile: absolute vs excess
            n_obs = len(etf_df[fc].dropna())
            max_lag_profile = max(1, min(n_obs // 2, 24))
            lag_range = range(1, max_lag_profile + 1)

            r2_abs = r_squared_by_lag(etf_df, fc, rc, lag_range)
            r2_exc = r_squared_by_lag(etf_df, fc, exc_col, lag_range)

            if len(r2_abs) > 0 and len(r2_exc) > 0:
                freq_label = {"D": "days", "W": "weeks", "ME": "months",
                              "QE": "quarters"}[freq]
                fig_lag = go.Figure()
                fig_lag.add_trace(go.Scatter(
                    x=r2_abs["lag"], y=r2_abs["r_squared"],
                    name="Absolute Return", mode="lines+markers",
                    line=dict(color="#1f77b4", width=2),
                ))
                fig_lag.add_trace(go.Scatter(
                    x=r2_exc["lag"], y=r2_exc["r_squared"],
                    name="Excess Return", mode="lines+markers",
                    line=dict(color="#ff7f0e", width=2),
                ))
                # Mark peak lags
                abs_peak = r2_abs.loc[r2_abs["r_squared"].idxmax()]
                exc_peak = r2_exc.loc[r2_exc["r_squared"].idxmax()]
                fig_lag.add_annotation(
                    x=abs_peak["lag"], y=abs_peak["r_squared"],
                    text=f"peak lag {int(abs_peak['lag'])}",
                    showarrow=True, arrowhead=2, font=dict(color="#1f77b4"),
                )
                fig_lag.add_annotation(
                    x=exc_peak["lag"], y=exc_peak["r_squared"],
                    text=f"peak lag {int(exc_peak['lag'])}",
                    showarrow=True, arrowhead=2, font=dict(color="#ff7f0e"),
                )
                fig_lag.update_layout(
                    height=400,
                    title=f"{selected_etf}: R² by Lag — Absolute vs Excess Return",
                    xaxis_title=f"Lag ({freq_label})",
                    yaxis_title="R² (single-lag simple regression)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_lag, width="stretch")

                st.caption(
                    f"Each point = R² from a simple regression Flow(t) ~ Return(t-k) "
                    f"using one lag at a time. Shows how far back the effect extends "
                    f"and at which lag it peaks."
                )

            # Regression tables
            etf_lags = auto_lags(n_obs)
            rp_detail = relative_performance_regression(
                etf_df, fc, rc, exc_col, etf_lags)
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
                            hide_index=True, width="stretch",
                        )
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
**What we're testing:** When a fund goes up 10%, does the same amount of money flow in
as flows out when it drops 10%? Or is the reaction lopsided?

We split past returns into a **positive part** (gains) and **negative part** (losses), and estimate
a separate coefficient for each:
- **β_pos** — how much money flows in per unit of positive return (chasing gains)
- **β_neg** — how much money flows out per unit of negative return (fleeing losses)
- **Asymmetry Ratio** = β_pos / |β_neg| — values > 1 mean investors chase gains more aggressively than they flee losses
- **Wald P** — p-value testing whether the two responses are statistically different (< 0.05 = significantly asymmetric)
""")

with st.spinner("Running asymmetry regressions..."):
    asym_summary = asymmetry_all_etfs(df_valid, fc, rc)

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
    st.plotly_chart(fig_asym, width="stretch")

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
        hide_index=True, width="stretch",
    )

    # Per-ETF coefficient plot
    with st.expander(f"Per-ETF Detail: {selected_etf}"):
        etf_df = df_valid[df_valid["ETF"] == selected_etf]
        etf_lags = auto_lags(len(etf_df[fc].dropna()))
        asym_detail = asymmetry_regression(etf_df, fc, rc, etf_lags)
        if asym_detail:
            coef = asym_detail["coefficients"]
            coef = coef[coef["Variable"] != "const"].copy()
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
            st.plotly_chart(fig_ci, width="stretch")

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
**What we're doing:** Instead of running separate regressions per ETF, we stack all ETFs together
into one big regression. This gives us much more statistical power and lets us answer:
"Across the entire universe of tech ETFs, do past returns predict future flows?"

Five model specifications, from simplest to most controlled:
- **Pooled OLS** — treats all observations the same, ignoring ETF identity
- **Entity FE** — adds a "fixed effect" per ETF (controls for the fact that XLK naturally attracts more money than a niche fund)
- **Entity+Time FE** — also controls for market-wide time trends (e.g., everyone pulled money in March 2020)
- **Entity FE + Excess** — adds excess return (relative performance) alongside absolute return
- **Entity FE + Controls** — adds rolling volatility as a control variable

The **Entity Fixed Effects chart** shows each ETF's baseline flow level (α_i) after controlling for returns —
positive = this fund attracts more money than average, negative = less.
""")

# Only use ETFs with both flows and excess returns for panel
panel_df = df_valid.dropna(subset=[fc, rc])

with st.spinner("Running panel regressions (5 specifications)..."):
    try:
        panel_comp = panel_regression_comparison(
            panel_df, fc, rc, excess_return_col=exc_col)

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
                hide_index=True, width="stretch",
            )

            # Entity fixed effects
            st.subheader("Entity Fixed Effects")
            min_n = panel_df.groupby("ETF")[fc].apply(
                lambda x: x.notna().sum()).min()
            panel_lags = auto_lags(min_n)
            fe_result = panel_regression(
                panel_df, fc, rc, lags=panel_lags,
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
                st.plotly_chart(fig_fe, width="stretch")

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
            st.plotly_chart(fig_cov, width="stretch")
        else:
            st.warning("Panel regression returned no results.")

    except ImportError:
        st.error("Please install `linearmodels`: `pip install linearmodels>=6.0`")
    except Exception as e:
        st.error(f"Panel regression error: {e}")
