"""ETF Performance Chasing: A Research Dashboard
Single-page narrative exploring whether ETF investors chase past performance."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES, PEER_ETF_NAMES, ALL_ETF_NAMES,
)
from analysis import (
    auto_lags, cross_correlation, r_squared_by_lag, r_squared_by_lag_all_etfs,
    relative_performance_regression, relative_performance_all_etfs,
    asymmetry_regression, asymmetry_all_etfs,
    panel_regression, panel_regression_comparison,
)

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")

with st.expander("About this dashboard"):
    st.markdown("""
This dashboard investigates whether ETF investors chase past performance across
**38 tech ETFs** (9 ARK + 29 peers). Each section's conclusion raises the next
section's question, building a complete research narrative.

**Data**: Daily fund flow and price data (2014–2026), aggregated to chosen frequency.

**Sections**:
1. **Does Performance Chasing Exist?** — Do past returns predict future flows?
2. **How Long Does the Effect Last?** — Lag profile across all 38 ETFs
3. **Absolute vs Relative Performance** — Own return vs peer-relative return?
4. **Asymmetric Response** — Do gains and losses trigger equal reactions?
5. **Panel Regression** — Robustness check across all ETFs simultaneously
""")

# --- Sidebar ---
BENCHMARK_OPTIONS = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2,
    )
    benchmark_label = st.selectbox("Benchmark", list(BENCHMARK_OPTIONS.keys()))
    benchmark = BENCHMARK_OPTIONS[benchmark_label]
    selected_etf = st.selectbox("Per-ETF View", ALL_ETF_NAMES, index=0)
    st.markdown("---")
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading 38 ETFs...")
def load_peer_data(freq, benchmark):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark=benchmark)


df = load_peer_data(freq, benchmark)

if freq == "D":
    fc, rc, fc_z, rc_z = "Fund_Flow", "Return", "Fund_Flow_Z", "Return_Z"
else:
    fc, rc, fc_z, rc_z = "Flow_Sum", "Return_Cum", "Flow_Sum_Z", "Return_Cum_Z"

exc_col = "Excess_Return"
freq_label = {"D": "days", "W": "weeks", "ME": "months", "QE": "quarters"}[freq]

# Filter to ETFs with enough flow data
etfs_with_flows = df.groupby("ETF")[fc].apply(lambda x: x.notna().sum())
valid_etfs = etfs_with_flows[etfs_with_flows > 20].index.tolist()
df_valid = df[df["ETF"].isin(valid_etfs)].copy()
n_valid = len(valid_etfs)
n_ark = len([e for e in valid_etfs if e in ETF_NAMES])
n_peer = n_valid - n_ark


# ============================================================
# Section 1 — Does Performance Chasing Exist?
# ============================================================
st.header("1. Does Performance Chasing Exist?")
st.markdown("""
**Question**: Do past returns predict future fund flows?

We test this with two approaches: an **R² by lag curve** (how much variance does
a single lagged return explain?) and a **cross-correlogram** (raw correlation at
each lag). If performance chasing exists, we expect significant positive
correlation at positive lags (past returns → current flows).
""")

etf_df = df_valid[df_valid["ETF"] == selected_etf].copy().sort_values("Date")

if len(etf_df) > 0:
    # R² by lag curve for selected ETF
    n_obs = len(etf_df[fc].dropna())
    max_lag_profile = max(1, min(n_obs // 2, 24))
    lag_range = range(1, max_lag_profile + 1)

    r2_abs = r_squared_by_lag(etf_df, fc, rc, lag_range)

    col_r2, col_cc = st.columns(2)

    with col_r2:
        st.subheader(f"{selected_etf}: R² by Lag")
        if len(r2_abs) > 0:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=r2_abs["lag"], y=r2_abs["r_squared"],
                mode="lines+markers", line=dict(color="#1f77b4", width=2),
                hovertemplate="Lag %{x} " + freq_label + "<br>R²: %{y:.4f}<extra></extra>",
            ))
            peak = r2_abs.loc[r2_abs["r_squared"].idxmax()]
            fig_r2.add_annotation(
                x=peak["lag"], y=peak["r_squared"],
                text=f"peak: lag {int(peak['lag'])}, R²={peak['r_squared']:.3f}",
                showarrow=True, arrowhead=2,
            )
            fig_r2.update_layout(
                height=380, xaxis_title=f"Lag ({freq_label})",
                yaxis_title="R²", margin=dict(l=60, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        else:
            st.info("Not enough data for R² profile.")

    # Cross-correlogram
    with col_cc:
        st.subheader(f"{selected_etf}: Cross-Correlogram")
        etf_ts = etf_df.set_index("Date").sort_index()
        max_lag_cc = min(max_lag_profile, 20)
        cc = cross_correlation(etf_ts[fc_z], etf_ts[rc_z], max_lag_cc)

        if len(cc) > 0:
            colors = ["#2ca02c" if p < 0.05 else "#c7c7c7" for p in cc["p_value"]]
            hover_texts = [
                f"Lag: {int(row.lag)} {freq_label}<br>"
                f"Corr: {row.correlation:.4f}<br>"
                f"p: {row.p_value:.4f}<br>"
                f"{'Past ret → flow' if row.lag > 0 else ('Flow → future ret' if row.lag < 0 else 'Same period')}"
                for row in cc.itertuples()
            ]
            fig_cc = go.Figure(go.Bar(
                x=cc["lag"], y=cc["correlation"], marker_color=colors,
                hovertext=hover_texts, hoverinfo="text",
            ))
            n_cc = len(etf_ts[[fc_z, rc_z]].dropna())
            if n_cc > 0:
                ci = 1.96 / n_cc ** 0.5
                fig_cc.add_hline(y=ci, line_dash="dot", line_color="red", opacity=0.5)
                fig_cc.add_hline(y=-ci, line_dash="dot", line_color="red", opacity=0.5)
            fig_cc.add_hline(y=0, line_color="gray", opacity=0.4)
            fig_cc.update_layout(
                height=380, xaxis_title=f"Lag ({freq_label})",
                yaxis_title="Correlation",
                margin=dict(l=60, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_cc, use_container_width=True)
        else:
            st.info("Not enough data for cross-correlation.")

    # Summary table: all ETFs
    st.subheader("Summary: All ETFs")
    st.caption(f"{n_valid} ETFs with sufficient data ({n_ark} ARK + {n_peer} peers)")

    @st.cache_data(show_spinner="Computing R² profiles...")
    def compute_all_r2(freq):
        return r_squared_by_lag_all_etfs(df_valid, fc, rc)

    all_r2 = compute_all_r2(freq)
    if len(all_r2) > 0:
        summary_rows = []
        for etf in all_r2["ETF"].unique():
            etf_r2 = all_r2[all_r2["ETF"] == etf]
            peak_row = etf_r2.loc[etf_r2["r_squared"].idxmax()]
            summary_rows.append({
                "ETF": etf,
                "Source": "ARK" if etf in ETF_NAMES else "Peer",
                "Peak Lag": int(peak_row["lag"]),
                "Peak R²": peak_row["r_squared"],
                "Peak p-value": peak_row["p_value"],
                "N": len(df_valid[df_valid["ETF"] == etf][fc].dropna()),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("Peak R²", ascending=False)
        st.dataframe(
            summary_df.style.format({"Peak R²": "{:.4f}", "Peak p-value": "{:.4f}"}),
            hide_index=True, use_container_width=True,
        )
else:
    st.warning(f"No data for {selected_etf}.")


# ============================================================
# Section 2 — How Long Does the Effect Last?
# ============================================================
st.markdown("---")
st.header("2. How Long Does the Effect Last?")
st.markdown("""
**Question**: At which lag is the effect strongest, and how far back does it extend?

The heatmap below shows R² at each lag for all 38 ETFs simultaneously. Bright
cells = strong predictive power. This reveals whether performance chasing is a
short-memory (1-period) or long-memory (multi-period) phenomenon, and whether
ARK ETFs differ from peers.
""")

if len(all_r2) > 0:
    # Heatmap: ETF × lag → R²
    pivot = all_r2.pivot(index="ETF", columns="lag", values="r_squared")

    # Sort: ARK first, then peers, each sorted by peak R²
    etf_peak = all_r2.groupby("ETF")["r_squared"].max().reset_index()
    etf_peak.columns = ["ETF", "peak_r2"]
    etf_peak["is_ark"] = etf_peak["ETF"].isin(ETF_NAMES)
    etf_peak = etf_peak.sort_values(
        ["is_ark", "peak_r2"], ascending=[False, False])
    ordered_etfs = [e for e in etf_peak["ETF"] if e in pivot.index]
    pivot = pivot.loc[ordered_etfs]

    # Mark ARK vs Peer in row labels
    labels = [f"{'★ ' if e in ETF_NAMES else ''}{e}" for e in pivot.index]

    fig_hm = px.imshow(
        pivot.values, x=[str(c) for c in pivot.columns], y=labels,
        aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="Lag", y="ETF", color="R²"),
    )
    fig_hm.update_layout(
        height=max(450, len(pivot) * 20),
        margin=dict(l=100, r=30, t=30, b=40),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("★ = ARK ETF. Sorted: ARK funds first, then peers, each by descending peak R².")

    # Peak lag distribution
    peak_lags = all_r2.loc[all_r2.groupby("ETF")["r_squared"].idxmax()]
    peak_lags["Source"] = peak_lags["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Peer")

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        fig_pk = px.histogram(
            peak_lags, x="lag", color="Source", barmode="overlay",
            nbins=max_lag_profile, color_discrete_map={"ARK": "#1f77b4", "Peer": "#aec7e8"},
            title="Peak Lag Distribution",
        )
        fig_pk.update_layout(height=300, xaxis_title=f"Peak Lag ({freq_label})")
        st.plotly_chart(fig_pk, use_container_width=True)

    with col_dist2:
        fig_box = px.box(
            peak_lags, x="Source", y="r_squared", color="Source",
            color_discrete_map={"ARK": "#1f77b4", "Peer": "#aec7e8"},
            title="Peak R² Distribution",
            points="all",
        )
        fig_box.update_layout(height=300, yaxis_title="Peak R²", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
else:
    st.warning("Not enough data for lag profile heatmap.")


# ============================================================
# Section 3 — Absolute vs Relative Performance
# ============================================================
st.markdown("---")
st.header("3. Absolute vs Relative Performance")
_bench_desc = {
    "SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "peer_avg": "peer-group average",
}
st.markdown(f"""
**Question**: Do investors react to the fund's own return, or how it performed
relative to the market?

Benchmark: **{_bench_desc[benchmark]}**

For each ETF we run three models predicting flows from lagged returns:
- **Absolute** — the ETF's own return
- **Excess** — return minus {_bench_desc[benchmark]} (relative performance)
- **Combined** — both together

If R² Excess > R² Absolute, investors care more about **relative performance**
than raw return.
""")

# Per-ETF: R² by lag, absolute vs excess
etf_df_sec3 = df_valid[df_valid["ETF"] == selected_etf]
if len(etf_df_sec3) > 0 and not etf_df_sec3[exc_col].dropna().empty:
    n_obs3 = len(etf_df_sec3[fc].dropna())
    max_lag3 = max(1, min(n_obs3 // 2, 24))
    lag_range3 = range(1, max_lag3 + 1)

    r2_abs3 = r_squared_by_lag(etf_df_sec3, fc, rc, lag_range3)
    r2_exc3 = r_squared_by_lag(etf_df_sec3, fc, exc_col, lag_range3)

    if len(r2_abs3) > 0 and len(r2_exc3) > 0:
        st.subheader(f"{selected_etf}: R² by Lag — Absolute vs Excess")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=r2_abs3["lag"], y=r2_abs3["r_squared"],
            name="Absolute Return", mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Lag %{x}<br>Absolute R²: %{y:.4f}<extra></extra>",
        ))
        fig_cmp.add_trace(go.Scatter(
            x=r2_exc3["lag"], y=r2_exc3["r_squared"],
            name="Excess Return", mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="Lag %{x}<br>Excess R²: %{y:.4f}<extra></extra>",
        ))
        abs_peak = r2_abs3.loc[r2_abs3["r_squared"].idxmax()]
        exc_peak = r2_exc3.loc[r2_exc3["r_squared"].idxmax()]
        fig_cmp.add_annotation(
            x=abs_peak["lag"], y=abs_peak["r_squared"],
            text=f"Abs peak: lag {int(abs_peak['lag'])}",
            showarrow=True, arrowhead=2, font=dict(color="#1f77b4"),
        )
        fig_cmp.add_annotation(
            x=exc_peak["lag"], y=exc_peak["r_squared"],
            text=f"Exc peak: lag {int(exc_peak['lag'])}",
            showarrow=True, arrowhead=2, font=dict(color="#ff7f0e"),
        )
        fig_cmp.update_layout(
            height=400, xaxis_title=f"Lag ({freq_label})",
            yaxis_title="R²",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# All ETFs bar chart
with st.spinner("Running relative performance regressions..."):
    rp_summary = relative_performance_all_etfs(df_valid, fc, rc, exc_col)

if len(rp_summary) > 0:
    rp_summary["Source"] = rp_summary["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Peer")

    st.subheader("R² Comparison: All ETFs")
    rp_melt = rp_summary.melt(
        id_vars=["ETF", "Source"],
        value_vars=["R²_Absolute", "R²_Excess", "R²_Combined"],
        var_name="Model", value_name="R²",
    )
    fig_rp = px.bar(
        rp_melt, x="ETF", y="R²", color="Model", barmode="group",
        color_discrete_map={"R²_Absolute": "#1f77b4", "R²_Excess": "#ff7f0e",
                            "R²_Combined": "#2ca02c"},
    )
    fig_rp.update_layout(
        height=450, xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=30, b=80),
    )
    st.plotly_chart(fig_rp, use_container_width=True)

    st.dataframe(
        rp_summary[["ETF", "Source", "R²_Absolute", "R²_Excess", "R²_Combined", "N"]]
        .sort_values("R²_Combined", ascending=False)
        .style.format({"R²_Absolute": "{:.4f}", "R²_Excess": "{:.4f}",
                        "R²_Combined": "{:.4f}"}),
        hide_index=True, use_container_width=True,
    )
else:
    st.warning("Not enough data for relative performance analysis.")


# ============================================================
# Section 4 — Asymmetric Response
# ============================================================
st.markdown("---")
st.header("4. Asymmetric Response")
st.markdown("""
**Question**: Do investors react equally to gains and losses?

We split past returns into positive (gains) and negative (losses) components and
estimate separate coefficients:
- **β_pos** — flow response per unit of positive return (chasing gains)
- **β_neg** — flow response per unit of negative return (fleeing losses)
- **Asymmetry Ratio** = β_pos / |β_neg| — values > 1 mean gain-chasing dominates
- **Wald P** — tests whether the asymmetry is statistically significant
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
        barmode="group", height=450,
        xaxis_tickangle=-45, yaxis_title="Coefficient Magnitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=30, b=80),
    )
    st.plotly_chart(fig_asym, use_container_width=True)

    # Summary table
    st.dataframe(
        asym_summary[["ETF", "Source", "Beta_Pos", "Beta_Neg",
                       "Asymmetry_Ratio", "Wald_P", "R²", "N"]]
        .sort_values("Asymmetry_Ratio", ascending=False)
        .style.format({
            "Beta_Pos": "{:.2f}", "Beta_Neg": "{:.2f}",
            "Asymmetry_Ratio": "{:.2f}", "Wald_P": "{:.4f}", "R²": "{:.4f}",
        }),
        hide_index=True, use_container_width=True,
    )

    # Per-ETF coefficient CI plot
    with st.expander(f"Per-ETF Detail: {selected_etf}"):
        etf_df_a = df_valid[df_valid["ETF"] == selected_etf]
        etf_lags_a = auto_lags(len(etf_df_a[fc].dropna()))
        asym_detail = asymmetry_regression(etf_df_a, fc, rc, etf_lags_a)
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
                error_y=dict(type="data", symmetric=False,
                             array=coef["CI_upper"] - coef["Coefficient"],
                             arrayminus=coef["Coefficient"] - coef["CI_lower"]),
            ))
            fig_ci.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ci.update_layout(
                height=350,
                title=f"{selected_etf}: Asymmetric Coefficients with 95% CI",
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
# Section 5 — Panel Regression (Robustness)
# ============================================================
st.markdown("---")
st.header("5. Panel Regression (Robustness)")
st.markdown("""
**Question**: Are these findings robust across all 38 ETFs simultaneously?

Instead of separate regressions per ETF, we stack all ETFs into one panel regression.
Five specifications from simplest to most controlled:
- **Pooled OLS** — ignores ETF identity
- **Entity FE** — controls for each ETF's baseline flow level
- **Entity+Time FE** — also controls for market-wide time trends
- **Entity FE + Excess** — adds relative performance
- **Entity FE + Controls** — adds rolling volatility
""")

panel_df = df_valid.dropna(subset=[fc, rc])

with st.spinner("Running panel regressions (5 specifications)..."):
    try:
        panel_comp = panel_regression_comparison(
            panel_df, fc, rc, excess_return_col=exc_col)

        if len(panel_comp) > 0:
            # Model comparison table
            st.subheader("5-Specification Comparison")
            fmt_dict = {"R²_within": "{:.4f}", "R²_overall": "{:.4f}"}
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
                    yaxis_title="Fixed Effect (α_i)",
                    xaxis_tickangle=-45,
                    margin=dict(l=60, r=30, t=30, b=80),
                )
                st.plotly_chart(fig_fe, use_container_width=True)

            # Data coverage heatmap
            st.subheader("Data Coverage")
            coverage = panel_df.groupby(
                ["ETF", pd.Grouper(key="Date", freq="YS")]
            )[fc].count().reset_index()
            coverage.columns = ["ETF", "Year", "Observations"]
            coverage["Year"] = coverage["Year"].dt.year
            pivot_cov = coverage.pivot(
                index="ETF", columns="Year", values="Observations").fillna(0)

            fig_cov = px.imshow(
                pivot_cov, aspect="auto", color_continuous_scale="Blues",
                labels=dict(color="Obs"),
            )
            fig_cov.update_layout(height=max(400, len(pivot_cov) * 18))
            st.plotly_chart(fig_cov, use_container_width=True)
        else:
            st.warning("Panel regression returned no results.")

    except ImportError:
        st.error("Please install `linearmodels`: `pip install linearmodels>=6.0`")
    except Exception as e:
        st.error(f"Panel regression error: {e}")
