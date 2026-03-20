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
    seasonality_analysis, seasonality_inflow_outflow,
    compute_etf_drawdowns, drawdown_flow_analysis, drawdown_flow_regression,
)
from scipy import stats

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.markdown(
    '<style>[data-testid="stMetricValue"] {font-size: 1.2rem;}</style>',
    unsafe_allow_html=True,
)
st.title("Do ETF Investors Chase Past Performance?")

st.markdown("""
This dashboard investigates whether ETF investors chase past performance across
**38 tech ETFs** (9 ARK + 29 tech peers). We use Bloomberg daily net fund flow
(creation/redemption) data, daily OHLCV prices, and monthly total net assets (AUM)
covering 2014–2026. Benchmarks include SPY (S&P 500), QQQ (Nasdaq-100), and a
peer-group average (equal-weighted cross-sectional mean return of all tech peer ETFs on each
date; dates with fewer than 10 reporting ETFs are excluded). When expressed as % of AUM, flows are computed
as the aggregate flow over the period divided by beginning-of-period AUM × 100.
""")

# --- Sidebar ---
ETF_FULL_NAMES = {
    "ARKK": "ARK Innovation",
    "ARKF": "ARK Fintech Innovation",
    "ARKG": "ARK Genomic Revolution",
    "ARKX": "ARK Space & Defense Innovation",
    "ARKB": "ARK 21Shares Bitcoin",
    "ARKQ": "ARK Autonomous Tech & Robotics",
    "ARKW": "ARK Next Generation Internet",
    "PRNT": "The 3D Printing ETF",
    "IZRL": "ARK Israel Innovative Technology",
    "FTXL": "First Trust Nasdaq Semiconductor",
    "PSI":  "Invesco Semiconductors",
    "SMH":  "VanEck Semiconductor",
    "SOXX": "iShares Semiconductor",
    "PTF":  "Invesco DW Tech Momentum",
    "XSD":  "SPDR S&P Semiconductor",
    "PSCT": "Invesco S&P SmallCap IT",
    "IGPT": "Invesco AI & Next Gen Software",
    "KNCT": "Invesco Next Gen Connectivity",
    "IXN":  "iShares Global Tech",
    "IGM":  "iShares Expanded Tech Sector",
    "IYW":  "iShares U.S. Technology",
    "XLK":  "Technology Select Sector SPDR",
    "FTEC": "Fidelity MSCI IT Index",
    "VGT":  "Vanguard Information Technology",
    "TDIV": "First Trust NASDAQ Tech Dividend",
    "QTEC": "First Trust NASDAQ-100 Tech",
    "FID":  "First Trust S&P Intl Dividend",
    "FXL":  "First Trust Tech AlphaDEX",
    "ERTH": "Invesco MSCI Sustainable Future",
    "XT":   "iShares Exponential Technologies",
    "GAMR": "Amplify Video Game Tech",
    "CQQQ": "Invesco China Technology",
    "FDN":  "First Trust DJ Internet Index",
    "HACK": "Amplify Cybersecurity",
    "PNQI": "Invesco NASDAQ Internet",
    "SKYY": "First Trust Cloud Computing",
    "CIBR": "First Trust NASDAQ Cybersecurity",
    "SOCL": "Global X Social Media",
}

FLOW_UNIT_OPTIONS = {"Raw $ (millions)": "raw", "% of AUM": "pct"}

TIME_RANGE_OPTIONS = ["All Time", "Last 5 Years", "Since Pandemic (2020-03)", "Custom"]

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2,
    )
    flow_unit_label = st.selectbox("Flow Unit", list(FLOW_UNIT_OPTIONS.keys()))
    flow_unit = FLOW_UNIT_OPTIONS[flow_unit_label]
    selected_etf = st.selectbox(
        "Per-ETF View", ALL_ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_FULL_NAMES.get(x, x)}",
    )

    st.markdown("---")
    time_range = st.selectbox("Time Range", TIME_RANGE_OPTIONS, index=0)
    if time_range == "Custom":
        date_start = st.date_input("Start date", value=pd.Timestamp("2014-01-01"))
        date_end = st.date_input("End date", value=pd.Timestamp("2026-12-31"))
    elif time_range == "Last 5 Years":
        date_end = pd.Timestamp("today")
        date_start = date_end - pd.DateOffset(years=5)
    elif time_range == "Since Pandemic (2020-03)":
        date_start = pd.Timestamp("2020-03-01")
        date_end = pd.Timestamp("today")
    else:
        date_start = None
        date_end = None

    st.markdown("---")
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading 38 ETFs...")
def load_peer_data(freq, benchmark):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark=benchmark)


df = load_peer_data(freq, "peer_avg")

# Apply global time filter
if date_start is not None and date_end is not None:
    _start = pd.Timestamp(date_start)
    _end = pd.Timestamp(date_end)
    df = df[(df["Date"] >= _start) & (df["Date"] <= _end)]

if freq == "D":
    fc_raw, rc = "Fund_Flow", "Return"
    fc_z_raw, rc_z = "Fund_Flow_Z", "Return_Z"
else:
    fc_raw, rc = "Flow_Sum", "Return_Cum"
    fc_z_raw, rc_z = "Flow_Sum_Z", "Return_Cum_Z"

# Switch flow column based on unit selector
if flow_unit == "pct" and "Flow_Pct" in df.columns and df["Flow_Pct"].notna().any():
    fc = "Flow_Pct"
    fc_z = "Flow_Pct_Z" if "Flow_Pct_Z" in df.columns else fc
    flow_ylabel = "Flow (% of AUM)"
else:
    fc = fc_raw
    fc_z = fc_z_raw
    flow_ylabel = "Flow ($M)"
    if flow_unit == "pct":
        st.sidebar.warning("AUM data unavailable — showing raw $ flows.")

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
# Section 1 — Fund Flow Distribution
# ============================================================
st.header("1. Fund Flow Distribution")
st.markdown(f"""
This section characterizes the **empirical distribution of net fund flows** across
38 tech ETFs. Understanding the shape, scale, and dispersion of flows is essential
before any regression analysis — it defines what constitutes a "normal" vs
"extreme" flow event and reveals whether ARK funds differ structurally from peers.
""")

# --- Per-ETF histogram + stats ---
etf_dist = df_valid[df_valid["ETF"] == selected_etf][fc].dropna()

if len(etf_dist) > 0:
    col_hist, col_stats = st.columns([2, 1])

    with col_hist:
        st.subheader(f"{selected_etf}: Flow Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=etf_dist, nbinsx=50,
            marker_color="#1f77b4", opacity=0.8,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig_dist.add_vline(x=etf_dist.median(), line_dash="dash",
                           line_color="red", opacity=0.7,
                           annotation_text="Median",
                           annotation_position="top right")
        fig_dist.add_vline(x=0, line_color="gray", opacity=0.4)
        fig_dist.update_layout(
            height=380, xaxis_title=flow_ylabel, yaxis_title="Count",
            margin=dict(l=60, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_dist, width="stretch")
        st.caption(
            "Skewness > 0 (right-skewed): more frequent small outflows offset by occasional large inflows. "
            "P25 < 0 in the percentile table means the fund experienced net outflows in at least 25% of periods."
        )

    with col_stats:
        st.subheader("Percentiles")
        pcts = [5, 10, 25, 50, 75, 90, 95]
        pct_vals = np.percentile(etf_dist, pcts)
        pct_df = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pcts],
            "Value": pct_vals,
        })
        pct_df_display = pct_df.copy()
        pct_df_display["Value"] = pct_df_display["Value"].map(
            lambda v: f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}")
        st.dataframe(pct_df_display, hide_index=True, width="stretch")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mean", f"{etf_dist.mean():.2f}")
        mc2.metric("Std Dev", f"{etf_dist.std():.2f}")
        mc3.metric("Skewness", f"{etf_dist.skew():.2f}")
        mc4.metric("N", f"{len(etf_dist):,}")

# Warn about extreme Flow_Pct observations for the selected ETF
if fc == "Flow_Pct":
    _etf_flow_pct = df_valid[df_valid["ETF"] == selected_etf][["Date", "Flow_Pct", "AUM"]].dropna()
    _extreme = _etf_flow_pct[_etf_flow_pct["Flow_Pct"].abs() > 100].sort_values("Flow_Pct", key=abs, ascending=False)
    if len(_extreme) > 0:
        _etf_first_date = df_valid[df_valid["ETF"] == selected_etf]["Date"].min()
        _lines = []
        for _, _row in _extreme.iterrows():
            _months_in = (_row["Date"].year - _etf_first_date.year) * 12 + (_row["Date"].month - _etf_first_date.month)
            _reason = "inception month — seed AUM" if _months_in <= 2 else "single-month inflow exceeded beginning-of-period AUM"
            _lines.append(
                f"- **{_row['Date'].strftime('%Y-%m')}**: Flow_Pct = {_row['Flow_Pct']:.0f}%, "
                f"AUM = ${_row['AUM']:.0f}M ({_reason})"
            )
        st.info(
            f"**{selected_etf}** has {len(_extreme)} month(s) where |Flow % of AUM| > 100%. "
            "This occurs when net flows in a single month exceed the fund's entire beginning-of-period AUM — "
            "either because the ETF had just launched (small seed AUM) or because of a genuine massive "
            "capital reallocation event.\n\n" + "\n".join(_lines)
        )

# --- Monthly time series: flow bars + excess return line ---
etf_ts_s1 = df_valid[df_valid["ETF"] == selected_etf].copy().sort_values("Date")
if len(etf_ts_s1) > 5 and exc_col in etf_ts_s1.columns:
    st.subheader(f"{selected_etf}: Monthly Flow & Excess Return")

    bench_s1 = st.radio(
        "Benchmark", ["SPY", "QQQ", "Peer Average"],
        horizontal=True, index=2, key="sec1_benchmark",
    )
    bench_s1_key = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}[bench_s1]

    if bench_s1_key != "peer_avg":
        _df_s1 = load_peer_data(freq, bench_s1_key)
        if date_start is not None and date_end is not None:
            _df_s1 = _df_s1[(_df_s1["Date"] >= pd.Timestamp(date_start)) & (_df_s1["Date"] <= pd.Timestamp(date_end))]
        etf_ts_s1 = _df_s1[_df_s1["ETF"] == selected_etf].copy().sort_values("Date")

    bar_colors = ["#2ca02c" if v >= 0 else "#d62728"
                  for v in etf_ts_s1[fc].fillna(0)]

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(
        go.Bar(x=etf_ts_s1["Date"], y=etf_ts_s1[fc],
               marker_color=bar_colors, name=flow_ylabel, opacity=0.7),
        secondary_y=False,
    )
    fig_dual.add_trace(
        go.Scatter(x=etf_ts_s1["Date"], y=etf_ts_s1[exc_col],
                   mode="lines", name="Excess Return",
                   line=dict(color="#1f77b4", width=1.5)),
        secondary_y=True,
    )
    fig_dual.update_yaxes(title_text=flow_ylabel, secondary_y=False)
    fig_dual.update_yaxes(title_text="Excess Return", secondary_y=True)
    fig_dual.update_layout(
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=60, t=40, b=30),
    )
    # Annotate top inflow and top outflow months
    flow_series = etf_ts_s1[[fc, "Date"]].dropna(subset=[fc])
    if len(flow_series) > 0:
        idx_max = flow_series[fc].idxmax()
        idx_min = flow_series[fc].idxmin()
        for idx, label_prefix in [(idx_max, "Peak inflow"), (idx_min, "Peak outflow")]:
            dt = flow_series.loc[idx, "Date"]
            val = flow_series.loc[idx, fc]
            fig_dual.add_annotation(
                x=dt, y=val, secondary_y=False,
                text=f"{label_prefix}<br>{dt.strftime('%Y-%m')}: {val:,.1f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                ax=0, ay=-35 if val >= 0 else 35,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderpad=3,
            )
    st.plotly_chart(fig_dual, width="stretch")
    st.caption(
        "Excess return = ETF return − benchmark return in the same period. "
        "If performance chasing exists, large positive excess returns should precede inflow bars in subsequent periods — "
        "look for the line leading the bars by 1–2 periods."
    )

# --- All-ETF comparison ---
st.subheader("Cross-ETF Comparison")

all_stats = []
for etf in valid_etfs:
    s = df_valid[df_valid["ETF"] == etf][fc].dropna()
    if len(s) < 5:
        continue
    all_stats.append({
        "ETF": etf,
        "Source": "ARK" if etf in ETF_NAMES else "Tech Peers",
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(),
        "P5": np.percentile(s, 5),
        "P95": np.percentile(s, 95),
        "Skew": s.skew(),
        "N": len(s),
    })
all_stats_df = pd.DataFrame(all_stats)

if len(all_stats_df) > 0:
    col_box, col_vol = st.columns(2)

    with col_box:
        # Violin plot: flow distributions, ARK vs Peers
        flow_long = df_valid[df_valid["ETF"].isin(valid_etfs)][[fc, "ETF", "Is_ARK"]].dropna()
        flow_long["Source"] = flow_long["Is_ARK"].map({True: "ARK", False: "Tech Peers"})
        fig_box_dist = go.Figure()
        for source, color in [("ARK", "#1f77b4"), ("Tech Peers", "#aec7e8")]:
            subset = flow_long[flow_long["Source"] == source][fc]
            fig_box_dist.add_trace(go.Violin(
                y=subset, name=source,
                box_visible=True, meanline_visible=True,
                line_color="black", fillcolor=color, opacity=0.85,
                marker=dict(color=color, outliercolor=color),
                points=False,
            ))
        fig_box_dist.update_layout(
            height=380, title="Flow Distribution: ARK vs Peers",
            yaxis_title=flow_ylabel,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="white",
        )
        fig_box_dist.update_xaxes(
            showline=True, linewidth=2, linecolor="black", showgrid=False,
        )
        fig_box_dist.update_yaxes(
            showline=True, linewidth=2, linecolor="black",
            showgrid=True, gridcolor="rgba(200,200,200,0.4)", gridwidth=0.5,
        )
        st.plotly_chart(fig_box_dist, width="stretch")
        st.caption(
            "All period-level flow observations pooled by group. Compare interquartile range and tail extent "
            "to assess whether ARK funds experience more extreme flow events than passive tech peers."
        )

    with col_vol:
        # Bar chart: std dev per ETF (flow volatility)
        vol_df = all_stats_df.sort_values("Std", ascending=False)
        colors = vol_df["Source"].map({"ARK": "#1f77b4", "Tech Peers": "#aec7e8"})
        fig_vol = go.Figure(go.Bar(
            x=vol_df["ETF"], y=vol_df["Std"], marker_color=colors,
            hovertemplate="ETF: %{x}<br>Std: %{y:.2f}<extra></extra>",
        ))
        fig_vol.update_layout(
            height=380, title="Flow Volatility by ETF",
            yaxis_title=f"Std Dev ({flow_ylabel})",
            xaxis_tickangle=-45,
            margin=dict(l=60, r=30, t=40, b=80),
        )
        st.plotly_chart(fig_vol, width="stretch")
        st.caption(
            "Flow volatility = standard deviation of period-level net fund flows across the full sample for each ETF. "
            "Higher σ implies more extreme inflow/outflow episodes, "
            "likely reflecting higher retail ownership, media attention, or concentrated investor bases."
        )

    # Summary table
    st.dataframe(
        all_stats_df.sort_values("Std", ascending=False)
        .style.format({
            "Mean": "{:.2f}", "Median": "{:.2f}", "Std": "{:.2f}",
            "P5": "{:.2f}", "P95": "{:.2f}", "Skew": "{:.2f}",
        }),
        hide_index=True, width="stretch",
    )


# ============================================================
# Section 2 — Does Performance Chasing Exist?
# ============================================================
st.markdown("---")
st.header("2. Does Performance Chasing Exist?")
st.markdown("""
We test whether lagged returns have **predictive power** for current-period fund flows.
For each lag *k*, we estimate the univariate OLS regression below and record the R²
(fraction of flow variance explained) and the slope coefficient β (flow sensitivity to
a one-unit return *k* periods ago).
""")
st.latex(r"Flow_{i,t} = \alpha + \beta \cdot Return_{i,t-k} + \varepsilon_{i,t}")
st.markdown("""
- **Dependent variable**: net fund flow (% of AUM or $M, per sidebar selection).
- **Independent variable**: the ETF's own return *k* periods ago.
- **Interpretation**: a positive, statistically significant β means inflows tend to follow positive past returns — evidence of performance chasing.
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
            st.plotly_chart(fig_r2, width="stretch")
            st.caption(
                "Each point = R² from a separate OLS: Flow(t) = α + β·Return(t−k) + ε. "
                "The peak identifies the lag at which past returns have the strongest univariate "
                "predictive power for current-period flows."
            )
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
            st.plotly_chart(fig_cc, width="stretch")
            st.caption(
                "Unlike the R² profile (left panel), this is not a regression — it shows the raw "
                "Pearson correlation between z-scored flows and z-scored returns at each lag. "
                "corr(z-flow(t), z-return(t−k)) at each lag k. "
                "**Positive lags** (right): does past return predict current flow? "
                "**Negative lags** (left): does current flow predict future return? "
                "Bars exceeding the dashed confidence band (±1.96/√N) are significant at the 5% level."
            )
        else:
            st.info("Not enough data for cross-correlation.")

    # Summary table: all ETFs
    st.subheader("Summary: All ETFs")
    st.caption(
        f"{n_valid} ETFs with sufficient data ({n_ark} ARK + {n_peer} peers). "
        "Peak Lag = the lag k* that maximizes R². F-stat tests H₀: β=0 (the slope is zero); "
        "a large F-stat means the relationship is statistically distinguishable from zero even if R² is low. "
        "Peak p-value = two-sided p-value for the slope coefficient at the optimal lag. "
        "Note: R² below 5% is typical in fund-flow prediction — flows are inherently noisy. "
        "The F-statistic is a more appropriate test: a large F (small p) confirms the relationship is "
        "statistically real even when R² is modest."
    )

    @st.cache_data(show_spinner="Computing R² profiles...")
    def compute_all_r2(freq, _time_key=None):
        return r_squared_by_lag_all_etfs(df_valid, fc, rc)

    _time_key = f"{date_start}_{date_end}" if date_start else "all"
    all_r2 = compute_all_r2(freq, _time_key=_time_key)
    if len(all_r2) > 0:
        summary_rows = []
        for etf in all_r2["ETF"].unique():
            etf_r2 = all_r2[all_r2["ETF"] == etf]
            peak_row = etf_r2.loc[etf_r2["r_squared"].idxmax()]
            summary_rows.append({
                "ETF": etf,
                "Source": "ARK" if etf in ETF_NAMES else "Tech Peers",
                "Peak Lag": int(peak_row["lag"]),
                "Peak R²": peak_row["r_squared"],
                "F-stat": peak_row.get("f_statistic", np.nan),
                "Peak p-value": peak_row["p_value"],
                "N": len(df_valid[df_valid["ETF"] == etf][fc].dropna()),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("Peak R²", ascending=False)
        st.dataframe(
            summary_df.style.format({
                "Peak R²": "{:.4f}", "F-stat": "{:.2f}", "Peak p-value": "{:.4f}",
            }),
            hide_index=True, width="stretch",
        )
else:
    st.warning(f"No data for {selected_etf}.")


# ============================================================
# Section 3 — Drawdown Analysis
# ============================================================
st.markdown("---")
st.header("3. Drawdown Analysis")
st.markdown("""
We identify **non-overlapping drawdown episodes** (peak-to-trough declines ≥ 10%)
for each ETF using an iterative deepest-first algorithm on a cumulative-return price index.
For each episode, we measure cumulative net flows over the 1, 2, 3, and 6 months following
the trough, then regress those post-drawdown flows on the drawdown's depth and duration:
""")
st.latex(r"CumFlow_{i,[t,t+h]} = \alpha + \beta_1 \cdot DrawdownDepth_i + \beta_2 \cdot Duration_i + \varepsilon")
st.markdown("""
- **CumFlow**: sum of net flows from trough date *t* to *t + h* months.
- **DrawdownDepth**: peak-to-trough decline in %, negative (e.g., −30% means a 30% drop).
- **Duration**: number of trading days from peak to trough.
- **Interpretation**: β₁ < 0 would mean deeper drawdowns lead to *larger outflows* (panic selling);
  β₁ > 0 would suggest contrarian buying after steep declines.
""")

@st.cache_data(show_spinner="Computing drawdowns...")
def compute_drawdowns(_df, return_col, _time_key=None):
    return compute_etf_drawdowns(_df, return_col, min_depth_pct=10.0)

dd_all = compute_drawdowns(df_valid, rc, _time_key=_time_key)

if len(dd_all) > 0:
    # Chart 1: price index with drawdown shading for selected ETF
    etf_dd = dd_all[dd_all["ETF"] == selected_etf]
    etf_prices = df_valid[df_valid["ETF"] == selected_etf].copy().sort_values("Date")

    if len(etf_prices) > 10:
        price_idx = (1 + etf_prices.set_index("Date")[rc].dropna()).cumprod() * 100

        fig_dd_price = go.Figure()
        fig_dd_price.add_trace(go.Scatter(
            x=price_idx.index, y=price_idx.values,
            mode="lines", name="Price Index", line=dict(color="#1f77b4", width=1.5),
        ))
        for _, dd_row in etf_dd.iterrows():
            fig_dd_price.add_vrect(
                x0=dd_row["peak_date"], x1=dd_row["trough_date"],
                fillcolor="red", opacity=0.15, line_width=0,
                annotation_text=f"{dd_row['depth_pct']:.0f}%",
                annotation_position="top left",
                annotation_font_size=9,
            )
        fig_dd_price.update_layout(
            height=400, yaxis_title="Price Index (base=100)",
            title=f"{selected_etf}: Price Index with Drawdown Periods",
            margin=dict(l=60, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_dd_price, width="stretch")
        st.caption(
            "Price index constructed from cumulative returns (base = 100). Drawdown episodes identified "
            "using an iterative deepest-first algorithm: find the largest peak-to-trough decline (≥ 10%), "
            "remove that segment, repeat on remaining periods. Labels show depth (%)."
        )

    # Compute flow analysis
    dd_flow = drawdown_flow_analysis(df_valid, dd_all, fc)

    if len(dd_flow) > 0:
        # Chart 2: scatter — drawdown depth vs cumulative flow (1m)
        col_scat1, col_scat2 = st.columns(2)
        with col_scat1:
            if "CumFlow_1m" in dd_flow.columns:
                fig_scat = px.scatter(
                    dd_flow, x="depth_pct", y="CumFlow_1m",
                    color="ETF", hover_data=["trough_date", "duration_days"],
                    title="Drawdown Depth vs 1-Month Cumulative Flow",
                )
                fig_scat.update_layout(
                    height=400, xaxis_title="Drawdown Depth (%)",
                    yaxis_title=f"Cumulative Flow 1m ({flow_ylabel})",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat, width="stretch")
                st.caption(
                    "Each dot = one drawdown episode across all 38 ETFs. A negative slope would indicate deeper "
                    "drawdowns trigger larger outflows (panic selling); a positive slope suggests contrarian buying."
                )

        with col_scat2:
            if "CumFlow_3m" in dd_flow.columns:
                fig_scat3 = px.scatter(
                    dd_flow, x="depth_pct", y="CumFlow_3m",
                    color="ETF", hover_data=["trough_date", "duration_days"],
                    title="Drawdown Depth vs 3-Month Cumulative Flow",
                )
                fig_scat3.update_layout(
                    height=400, xaxis_title="Drawdown Depth (%)",
                    yaxis_title=f"Cumulative Flow 3m ({flow_ylabel})",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat3, width="stretch")
                st.caption(
                    "3-month forward window captures delayed investor reactions beyond the immediate post-trough period."
                )

        # Regression table
        dd_reg = drawdown_flow_regression(dd_flow)
        if len(dd_reg) > 0:
            st.subheader("Post-Drawdown Flow Regression")
            st.caption(
                "OLS regression of cumulative post-trough flows on drawdown characteristics, pooled across all ETFs. "
                "Each row = a different forward horizon (1m, 2m, 3m, 6m). "
                "β_Depth: change in cumulative flow per 1 pp deeper drawdown. "
                "β_Duration: change in cumulative flow per additional trading day of drawdown. "
                "Significant β_Depth_p < 0.05 confirms that drawdown severity predicts subsequent flow behavior."
            )
            st.dataframe(
                dd_reg.style.format({
                    "β_Depth": "{:.4f}", "β_Depth_p": "{:.4f}",
                    "β_Duration": "{:.4f}", "β_Duration_p": "{:.4f}",
                    "R²": "{:.4f}",
                }),
                hide_index=True, width="stretch",
            )
    else:
        st.info("Not enough post-drawdown flow data for analysis.")
else:
    st.info("No drawdowns ≥ 10% found in the selected time range.")


# ============================================================
# Section 4 — How Long Does the Effect Last?
# ============================================================
st.markdown("---")
st.header("4. How Long Does the Effect Last?")
st.markdown("""
Section 2 tested performance chasing for one ETF at a time. Here we visualize the
**lag structure across all 38 ETFs simultaneously** to answer: is the effect concentrated
at short lags (e.g., last month's return drives this month's flow) or does it persist
over multiple periods? The heatmap maps every (ETF, lag) pair to its univariate R².
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
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "Each cell: R² from OLS Flow(t) = α + β·Return(t−k) + ε. ★ = ARK ETF. "
        "Warmer colors (darker red) = higher R². "
        "A bright vertical band at one lag across many ETFs indicates a common chasing horizon; "
        "scattered bright cells suggest heterogeneous investor behavior across funds."
    )

    # Peak lag distribution
    peak_lags = all_r2.loc[all_r2.groupby("ETF")["r_squared"].idxmax()]
    peak_lags["Source"] = peak_lags["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        fig_pk = px.histogram(
            peak_lags, x="lag", color="Source", barmode="overlay",
            nbins=max_lag_profile, color_discrete_map={"ARK": "#1f77b4", "Tech Peers": "#aec7e8"},
            title="Peak Lag Distribution",
        )
        fig_pk.update_layout(height=300, xaxis_title=f"Peak Lag ({freq_label})")
        st.plotly_chart(fig_pk, width="stretch")
        st.caption(
            f"Distribution of optimal lag k* across ETFs. If clustered at lag 1, investors react within one {freq_label[:-1]}. "
            "A wider spread implies heterogeneous investor horizons."
        )

    with col_dist2:
        fig_box = px.box(
            peak_lags, x="Source", y="r_squared", color="Source",
            color_discrete_map={"ARK": "#1f77b4", "Tech Peers": "#aec7e8"},
            title="Peak R² Distribution",
            points="all",
        )
        fig_box.update_layout(height=300, yaxis_title="Peak R²", showlegend=False)
        st.plotly_chart(fig_box, width="stretch")
        st.caption(
            "Compares the strength of performance chasing (peak R²) between groups. "
            "A systematically higher distribution for ARK would suggest their investors are more return-sensitive."
        )
else:
    st.warning("Not enough data for lag profile heatmap.")


# ============================================================
# Section 5 — Seasonality
# ============================================================
st.markdown("---")
st.header("5. Seasonality")
st.markdown("""
If fund flows are driven by **calendar effects** (e.g., tax-loss harvesting in
December, reallocation in January), the performance-chasing signal from Sections 2–4
could be a confound. This section tests for monthly seasonality using daily flow data,
which provides more granularity than the aggregated data used elsewhere.
""")

# Always use daily data for seasonality analysis
@st.cache_data(show_spinner="Loading daily data for seasonality...")
def load_daily_data():
    return get_prepared_data_with_peers(freq="D", zscore_type="full", benchmark="peer_avg")

daily_df = load_daily_data()

# Apply time filter to daily data
if date_start is not None and date_end is not None:
    daily_df = daily_df[(daily_df["Date"] >= pd.Timestamp(date_start)) & (daily_df["Date"] <= pd.Timestamp(date_end))]

etf_daily = daily_df[daily_df["ETF"] == selected_etf]

daily_fc = "Flow_Pct" if flow_unit == "pct" and "Flow_Pct" in daily_df.columns and daily_df["Flow_Pct"].notna().any() else "Fund_Flow"
seasonal = seasonality_analysis(etf_daily, daily_fc)

if len(seasonal) > 0:
    st.subheader(f"{selected_etf} — Average Flow by Month")
    m_colors = ["#2ca02c" if m >= 0 else "#d62728" for m in seasonal["Mean"]]
    fig_s = go.Figure(go.Bar(
        x=seasonal["Month_Name"], y=seasonal["Mean"], marker_color=m_colors,
        error_y=dict(type="data",
                     array=seasonal["Std"] / seasonal["Count"] ** 0.5,
                     visible=True),
        hovertemplate="Month: %{x}<br>Avg Flow: %{y:.2f}<extra></extra>",
    ))
    daily_ylabel = "Avg Daily Flow (% AUM)" if daily_fc == "Flow_Pct" else "Avg Daily Flow ($M)"
    fig_s.update_layout(height=380, yaxis_title=daily_ylabel,
                        margin=dict(l=60, r=40, t=30, b=30))
    fig_s.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    st.plotly_chart(fig_s, width="stretch")
    st.caption(
        "Mean daily flow by calendar month, pooled across all years. Error bars = ± one standard error (σ/√N). "
        "A negative December bar combined with a positive January bar would support the "
        "tax-loss harvesting / January reallocation hypothesis."
    )

    # January t-test
    jan = etf_daily[etf_daily["Date"].dt.month == 1][daily_fc].dropna()
    other = etf_daily[etf_daily["Date"].dt.month != 1][daily_fc].dropna()
    if len(jan) > 5 and len(other) > 5:
        t_stat, p_val = stats.ttest_ind(jan, other, equal_var=False)
        c1, c2, c3 = st.columns(3)
        jan_unit = "% AUM/day" if daily_fc == "Flow_Pct" else "$M/day"
        c1.metric(f"Jan Avg ({jan_unit})", f"{jan.mean():.2f}")
        c2.metric("Other Months Avg", f"{other.mean():.2f}")
        c3.metric("Jan vs Others p-value", f"{p_val:.4f}")
        st.caption(
            "Welch's two-sample t-test (unequal variances) comparing mean daily flow in January vs all "
            "other months. p < 0.05 indicates January flows are statistically different from the annual baseline."
        )

    # Inflow / Outflow breakdown
    io_data = seasonality_inflow_outflow(etf_daily, daily_fc)
    if len(io_data) > 0:
        st.subheader(f"{selected_etf} — Inflow vs Outflow by Month")
        fig_io = go.Figure()
        fig_io.add_trace(go.Bar(
            x=io_data["Month_Name"], y=io_data["Avg_Inflow"],
            name="Avg Inflow", marker_color="#2ca02c",
            hovertemplate="Month: %{x}<br>Avg Inflow: %{y:.2f}<extra></extra>",
        ))
        fig_io.add_trace(go.Bar(
            x=io_data["Month_Name"], y=io_data["Avg_Outflow"],
            name="Avg Outflow", marker_color="#d62728",
            hovertemplate="Month: %{x}<br>Avg Outflow: %{y:.2f}<extra></extra>",
        ))
        fig_io.update_layout(
            barmode="group", height=380, yaxis_title=daily_ylabel,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=40, t=40, b=30),
        )
        fig_io.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        st.plotly_chart(fig_io, width="stretch")
        st.caption(
            "Conditional on flow direction: average magnitude on inflow days vs outflow days per month. "
            "This isolates whether seasonal patterns are driven by larger inflows, larger outflows, or both. "
            "A deeper outflow bar in December = heavier selling on outflow days, consistent with tax-loss harvesting."
        )
else:
    st.warning(f"Not enough daily data for {selected_etf} seasonality analysis.")


# ============================================================
# Section 6 — Absolute vs Relative Performance
# ============================================================
st.markdown("---")
st.header("6. Absolute vs Relative Performance")

_bench_desc = {
    "SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "peer_avg": "peer-group average",
}
benchmark = st.radio(
    "Benchmark for excess return",
    ["SPY", "QQQ", "Peer Average"],
    horizontal=True, index=2,
    key="sec6_benchmark",
)
benchmark = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}[benchmark]

# Load data with selected benchmark (cached per benchmark)
df_bench = load_peer_data(freq, benchmark)
if date_start is not None and date_end is not None:
    df_bench = df_bench[(df_bench["Date"] >= pd.Timestamp(date_start)) & (df_bench["Date"] <= pd.Timestamp(date_end))]
df_bench_valid = df_bench[df_bench["ETF"].isin(valid_etfs)].copy()

st.markdown(f"""
Section 2 used the ETF's **own return** as the predictor. But investors may benchmark
their ETF against the market — reacting to **relative outperformance or underperformance**
rather than absolute gains/losses. We compare three OLS specifications:

Benchmark: **{_bench_desc[benchmark]}** | Excess Return = ETF Return − Benchmark Return
""")
st.latex(r"\text{Model 1 (Absolute): } Flow_{i,t} = \alpha + \sum_k \beta_k \cdot Return_{i,t-k} + \varepsilon")
st.latex(r"\text{Model 2 (Excess): } Flow_{i,t} = \alpha + \sum_k \gamma_k \cdot ExcessReturn_{i,t-k} + \varepsilon")
st.latex(r"\text{Model 3 (Combined): } Flow_{i,t} = \alpha + \sum_k \beta_k \cdot Return_{i,t-k} + \sum_k \gamma_k \cdot ExcessReturn_{i,t-k} + \varepsilon")
st.markdown("""
- If **R²(Excess) > R²(Absolute)**: investors care more about relative performance.
- If **R²(Combined) ≫ R²(Absolute)**: excess return contains additional information beyond own return.
- **F-statistic**: tests H₀ that all slope coefficients are jointly zero. Large F = model has predictive power.
""")

# Per-ETF: R² by lag, absolute vs excess
etf_df_sec6 = df_bench_valid[df_bench_valid["ETF"] == selected_etf]
if len(etf_df_sec6) > 0 and not etf_df_sec6[exc_col].dropna().empty:
    n_obs6 = len(etf_df_sec6[fc].dropna())
    max_lag6 = max(1, min(n_obs6 // 2, 24))
    lag_range6 = range(1, max_lag6 + 1)

    r2_abs6 = r_squared_by_lag(etf_df_sec6, fc, rc, lag_range6)
    r2_exc6 = r_squared_by_lag(etf_df_sec6, fc, exc_col, lag_range6)

    if len(r2_abs6) > 0 and len(r2_exc6) > 0:
        st.subheader(f"{selected_etf}: R² by Lag — Absolute vs Excess")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=r2_abs6["lag"], y=r2_abs6["r_squared"],
            name="Absolute Return", mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Lag %{x}<br>Absolute R²: %{y:.4f}<extra></extra>",
        ))
        fig_cmp.add_trace(go.Scatter(
            x=r2_exc6["lag"], y=r2_exc6["r_squared"],
            name="Excess Return", mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="Lag %{x}<br>Excess R²: %{y:.4f}<extra></extra>",
        ))
        abs_peak = r2_abs6.loc[r2_abs6["r_squared"].idxmax()]
        exc_peak = r2_exc6.loc[r2_exc6["r_squared"].idxmax()]
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
        st.plotly_chart(fig_cmp, width="stretch")
        st.caption(
            "R² from two univariate regressions: Flow ~ Return(t−k) vs Flow ~ ExcessReturn(t−k). "
            "If the excess curve dominates, investors respond more to relative performance than absolute gains/losses."
        )

# All ETFs bar chart
with st.spinner("Running relative performance regressions..."):
    rp_summary = relative_performance_all_etfs(df_bench_valid, fc, rc, exc_col)

if len(rp_summary) > 0:
    rp_summary["Source"] = rp_summary["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")

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
    st.plotly_chart(fig_rp, width="stretch")
    st.caption(
        "R² from three OLS specifications per ETF (automatic lag selection). ETFs where excess R² exceeds "
        "absolute R² show stronger sensitivity to relative performance. The combined model tests whether both "
        "absolute and relative returns contribute independently."
    )

    display_cols = ["ETF", "Source", "R²_Absolute", "R²_Excess", "R²_Combined"]
    fmt = {"R²_Absolute": "{:.4f}", "R²_Excess": "{:.4f}", "R²_Combined": "{:.4f}"}
    if "F_Abs" in rp_summary.columns:
        display_cols += ["F_Abs", "F_Exc", "F_Comb"]
        fmt.update({"F_Abs": "{:.2f}", "F_Exc": "{:.2f}", "F_Comb": "{:.2f}"})
    display_cols.append("N")

    st.dataframe(
        rp_summary[display_cols]
        .sort_values("R²_Combined", ascending=False)
        .style.format(fmt),
        hide_index=True, width="stretch",
    )
else:
    st.warning("Not enough data for relative performance analysis.")


# ============================================================
# Section 7 — Asymmetric Response
# ============================================================
st.markdown("---")
st.header("7. Asymmetric Response")
st.markdown("""
Sections 2–6 assume a **symmetric** relationship: a +10% return has the same magnitude effect
on flows as a −10% return. Here we relax that assumption by decomposing lagged returns
into positive and negative components and estimating separate slope coefficients:
""")
st.latex(r"Flow_{i,t} = \alpha + \sum_k \beta^+_k \cdot Return^+_{i,t-k} + \sum_k \beta^-_k \cdot Return^-_{i,t-k} + \varepsilon")
st.markdown(r"""
- **Return⁺ = max(Return, 0)**: the gain component (positive months only, zero otherwise).
- **Return⁻ = min(Return, 0)**: the loss component (negative months only, zero otherwise).
- **β⁺**: flow response per unit of positive return — measures **gain chasing**.
- **β⁻**: flow response per unit of negative return — measures **loss fleeing** (expected negative: losses cause outflows).
- **Asymmetry Ratio** = β⁺ / |β⁻|: values > 1 mean investors chase gains more than they flee losses.
- **Wald test**: H₀: β⁺ + β⁻ = 0 (symmetric response). p < 0.05 rejects symmetry.
""")

with st.spinner("Running asymmetry regressions..."):
    asym_summary = asymmetry_all_etfs(df_valid, fc, rc)

if len(asym_summary) > 0:
    asym_summary["Source"] = asym_summary["ETF"].apply(
        lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")

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
    st.plotly_chart(fig_asym, width="stretch")
    st.caption(
        "Compares gain sensitivity (β⁺) to loss sensitivity (|β⁻|) per ETF. When β⁺ > |β⁻|, investors chase "
        "gains more aggressively than they flee losses — asymmetric performance chasing."
    )

    # Summary table
    st.dataframe(
        asym_summary[["ETF", "Source", "Beta_Pos", "Beta_Neg",
                       "Asymmetry_Ratio", "Wald_P", "R²", "N"]]
        .sort_values("Asymmetry_Ratio", ascending=False)
        .style.format({
            "Beta_Pos": "{:.2f}", "Beta_Neg": "{:.2f}",
            "Asymmetry_Ratio": "{:.2f}", "Wald_P": "{:.4f}", "R²": "{:.4f}",
        }),
        hide_index=True, width="stretch",
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
            st.plotly_chart(fig_ci, width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Asymmetry Ratio", f"{asym_detail['asymmetry_ratio']:.2f}")
            c2.metric("Wald p-value", f"{asym_detail['wald_p']:.4f}")
            c3.metric("R²", f"{asym_detail['r_squared']:.4f}")
else:
    st.warning("Not enough data for asymmetry analysis.")


# ============================================================
# Section 8 — Panel Regression (Robustness)
# ============================================================
st.markdown("---")
st.header("8. Panel Regression (Robustness)")

benchmark_panel = st.radio(
    "Benchmark for excess return",
    ["SPY", "QQQ", "Peer Average"],
    horizontal=True, index=2,
    key="sec8_benchmark",
)
benchmark_panel = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}[benchmark_panel]

df_panel_bench = load_peer_data(freq, benchmark_panel)
if date_start is not None and date_end is not None:
    df_panel_bench = df_panel_bench[(df_panel_bench["Date"] >= pd.Timestamp(date_start)) & (df_panel_bench["Date"] <= pd.Timestamp(date_end))]
df_panel_bench_valid = df_panel_bench[df_panel_bench["ETF"].isin(valid_etfs)].copy()

st.markdown(f"""
Sections 2–7 ran separate OLS regressions per ETF. Here we **pool all 38 ETFs** into a
single panel regression to test whether the performance-chasing relationship holds across
the full cross-section simultaneously. Five specifications, from simplest to most controlled:
""")
st.latex(r"Flow_{i,t} = \alpha_i + \lambda_t + \sum_k \beta_k \cdot Return_{i,t-k} + \gamma \cdot ExcessReturn_{i,t-k} + \delta \cdot Volatility_{i,t} + \varepsilon_{i,t}")
st.markdown(f"""
| Specification | αᵢ (entity FE) | λₜ (time FE) | Excess Return | Volatility |
|---|---|---|---|---|
| Pooled OLS | — | — | — | — |
| Entity FE | ✓ | — | — | — |
| Entity+Time FE | ✓ | ✓ | — | — |
| Entity FE + Excess | ✓ | — | ✓ ({_bench_desc[benchmark_panel]}) | — |
| Entity FE + Controls | ✓ | — | — | ✓ (5-period rolling σ) |

Standard errors clustered by entity to account for within-ETF serial correlation.
""")

panel_df = df_panel_bench_valid.dropna(subset=[fc, rc])

with st.spinner("Running panel regressions (5 specifications)..."):
    try:
        panel_comp = panel_regression_comparison(
            panel_df, fc, rc, excess_return_col=exc_col)

        if len(panel_comp) > 0:
            # Model comparison table
            st.subheader("5-Specification Comparison")
            st.caption(
                "R²_within: fraction of within-entity (demeaned) variation explained — measures how well lagged returns "
                "predict flows after removing each ETF's average flow level. R²_overall: fraction of total variation explained. "
                "F_stat: joint test that all slope coefficients = 0; large F with small F_pval confirms the model has "
                "explanatory power. Coefficient columns show the estimated β and its p-value for each regressor."
            )
            fmt_dict = {
                "R²_within": "{:.4f}", "R²_overall": "{:.4f}",
                "F_stat": "{:.2f}", "F_pval": "{:.4f}",
            }
            for col in panel_comp.columns:
                if col.endswith("_coef"):
                    fmt_dict[col] = "{:.4f}"
                elif col.endswith("_pval") and col != "F_pval":
                    fmt_dict[col] = "{:.4f}"
            st.dataframe(
                panel_comp.style.format(fmt_dict, na_rep="—"),
                hide_index=True, width="stretch",
            )

            # Entity fixed effects
            st.subheader("Entity Fixed Effects")
            st.caption(
                "Estimated entity fixed effects (αᵢ) from the Entity FE model. Each bar represents an ETF's "
                "baseline flow level after controlling for lagged returns. Positive αᵢ = the ETF attracts "
                "flows beyond what its past returns predict (strong brand/momentum); negative αᵢ = structural outflows "
                "unexplained by returns."
            )
            min_n = panel_df.groupby("ETF")[fc].apply(
                lambda x: x.notna().sum()).min()
            panel_lags = auto_lags(min_n)
            fe_result = panel_regression(
                panel_df, fc, rc, lags=panel_lags,
                entity_effects=True, time_effects=False)
            if fe_result and "entity_effects" in fe_result:
                fe_df = fe_result["entity_effects"].sort_values("Fixed_Effect")
                fe_df["Source"] = fe_df["ETF"].apply(
                    lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")
                fig_fe = go.Figure()
                for source, color in [("ARK", "#1f77b4"), ("Tech Peers", "#aec7e8")]:
                    subset = fe_df[fe_df["Source"] == source]
                    fig_fe.add_trace(go.Bar(
                        x=subset["ETF"], y=subset["Fixed_Effect"],
                        marker_color=color, name=source,
                        hovertemplate="ETF: %{x}<br>α_i: %{y:.2f}<extra></extra>",
                    ))
                fig_fe.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_fe.update_layout(
                    height=400,
                    yaxis_title="Fixed Effect (α_i)",
                    xaxis_tickangle=-45,
                    margin=dict(l=60, r=30, t=30, b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    barmode="relative",
                )
                st.plotly_chart(fig_fe, width="stretch")

            # Data coverage heatmap
            st.subheader("Data Coverage")
            st.caption(
                "Non-missing observations per ETF per year. "
                "Sparse coverage for newer ETFs in early years may reduce panel regression precision for those entities."
            )
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
            st.plotly_chart(fig_cov, width="stretch")
        else:
            st.warning("Panel regression returned no results.")

    except ImportError:
        st.error("Please install `linearmodels`: `pip install linearmodels>=6.0`")
    except Exception as e:
        st.error(f"Panel regression error: {e}")
