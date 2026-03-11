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
    seasonality_analysis,
)
from scipy import stats

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")

with st.expander("About this dashboard"):
    st.markdown("""
This dashboard investigates whether ETF investors chase past performance across
**38 tech ETFs** (9 ARK + 29 peers). Each section's conclusion raises the next
section's question, building a complete research narrative.

**Data**: Daily fund flow and price data (2014–2026), aggregated to chosen frequency.

**Sections**:
1. **Fund Flow Distribution** — What does a typical flow look like? What's big vs small?
2. **Does Performance Chasing Exist?** — Do past returns predict future flows?
3. **How Long Does the Effect Last?** — Lag profile across all 38 ETFs
4. **Seasonality** — Calendar effects in fund flows (January reallocation, etc.)
5. **Absolute vs Relative Performance** — Own return vs market-relative return?
6. **Asymmetric Response** — Do gains and losses trigger equal reactions?
7. **Panel Regression** — Robustness check across all ETFs simultaneously
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
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading 38 ETFs...")
def load_peer_data(freq, benchmark):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark=benchmark)


df = load_peer_data(freq, "peer_avg")

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
**Question**: What does a typical fund flow look like? What counts as a large
inflow or outflow?

Before testing for performance chasing, we need to understand the data.
This section shows the distribution of {flow_ylabel.lower()} at the chosen
frequency, with percentile breakdowns and cross-ETF comparisons.
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
        st.plotly_chart(fig_dist, use_container_width=True)

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
        st.dataframe(pct_df_display, hide_index=True, use_container_width=True)

        st.metric("Mean", f"{etf_dist.mean():.2f}")
        st.metric("Std Dev", f"{etf_dist.std():.2f}")
        st.metric("Skewness", f"{etf_dist.skew():.2f}")
        st.metric("Observations", f"{len(etf_dist):,}")

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
        # Box plot: flow distributions, ARK vs Peers
        flow_long = df_valid[df_valid["ETF"].isin(valid_etfs)][[fc, "ETF", "Is_ARK"]].dropna()
        flow_long["Source"] = flow_long["Is_ARK"].map({True: "ARK", False: "Tech Peers"})
        fig_box_dist = px.box(
            flow_long, x="Source", y=fc, color="Source",
            color_discrete_map={"ARK": "#1f77b4", "Tech Peers": "#aec7e8"},
            title="Flow Distribution: ARK vs Peers",
            points=False,
        )
        fig_box_dist.update_layout(height=380, yaxis_title=flow_ylabel,
                                   showlegend=False)
        st.plotly_chart(fig_box_dist, use_container_width=True)

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
        st.plotly_chart(fig_vol, use_container_width=True)

    # Summary table
    st.dataframe(
        all_stats_df.sort_values("Std", ascending=False)
        .style.format({
            "Mean": "{:.2f}", "Median": "{:.2f}", "Std": "{:.2f}",
            "P5": "{:.2f}", "P95": "{:.2f}", "Skew": "{:.2f}",
        }),
        hide_index=True, use_container_width=True,
    )


# ============================================================
# Section 2 — Does Performance Chasing Exist?
# ============================================================
st.markdown("---")
st.header("2. Does Performance Chasing Exist?")
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
                "Source": "ARK" if etf in ETF_NAMES else "Tech Peers",
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
# Section 3 — How Long Does the Effect Last?
# ============================================================
st.markdown("---")
st.header("3. How Long Does the Effect Last?")
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
        lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        fig_pk = px.histogram(
            peak_lags, x="lag", color="Source", barmode="overlay",
            nbins=max_lag_profile, color_discrete_map={"ARK": "#1f77b4", "Tech Peers": "#aec7e8"},
            title="Peak Lag Distribution",
        )
        fig_pk.update_layout(height=300, xaxis_title=f"Peak Lag ({freq_label})")
        st.plotly_chart(fig_pk, use_container_width=True)

    with col_dist2:
        fig_box = px.box(
            peak_lags, x="Source", y="r_squared", color="Source",
            color_discrete_map={"ARK": "#1f77b4", "Tech Peers": "#aec7e8"},
            title="Peak R² Distribution",
            points="all",
        )
        fig_box.update_layout(height=300, yaxis_title="Peak R²", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
else:
    st.warning("Not enough data for lag profile heatmap.")


# ============================================================
# Section 4 — Seasonality
# ============================================================
st.markdown("---")
st.header("4. Seasonality")
st.markdown("""
**Question**: Are there calendar effects in fund flows?

Some months (e.g., January) may see systematic reallocation. If flows are
seasonally driven, the performance-chasing signal from Sections 2–3 could be
a confound. This section checks whether calendar effects exist and how large
they are.
""")

# Always use daily data for seasonality analysis
@st.cache_data(show_spinner="Loading daily data for seasonality...")
def load_daily_data():
    return get_prepared_data_with_peers(freq="D", zscore_type="full", benchmark="peer_avg")

daily_df = load_daily_data()
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
    st.plotly_chart(fig_s, use_container_width=True)

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
else:
    st.warning(f"Not enough daily data for {selected_etf} seasonality analysis.")


# ============================================================
# Section 5 — Absolute vs Relative Performance
# ============================================================
st.markdown("---")
st.header("5. Absolute vs Relative Performance")

_bench_desc = {
    "SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "peer_avg": "peer-group average",
}
benchmark = st.radio(
    "Benchmark for excess return",
    ["SPY", "QQQ", "Peer Average"],
    horizontal=True, index=2,
    key="sec5_benchmark",
)
benchmark = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}[benchmark]

# Load data with selected benchmark (cached per benchmark)
df_bench = load_peer_data(freq, benchmark)
df_bench_valid = df_bench[df_bench["ETF"].isin(valid_etfs)].copy()

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
etf_df_sec4 = df_bench_valid[df_bench_valid["ETF"] == selected_etf]
if len(etf_df_sec4) > 0 and not etf_df_sec4[exc_col].dropna().empty:
    n_obs4 = len(etf_df_sec4[fc].dropna())
    max_lag4 = max(1, min(n_obs4 // 2, 24))
    lag_range4 = range(1, max_lag4 + 1)

    r2_abs4 = r_squared_by_lag(etf_df_sec4, fc, rc, lag_range4)
    r2_exc4 = r_squared_by_lag(etf_df_sec4, fc, exc_col, lag_range4)

    if len(r2_abs4) > 0 and len(r2_exc4) > 0:
        st.subheader(f"{selected_etf}: R² by Lag — Absolute vs Excess")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=r2_abs4["lag"], y=r2_abs4["r_squared"],
            name="Absolute Return", mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Lag %{x}<br>Absolute R²: %{y:.4f}<extra></extra>",
        ))
        fig_cmp.add_trace(go.Scatter(
            x=r2_exc4["lag"], y=r2_exc4["r_squared"],
            name="Excess Return", mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="Lag %{x}<br>Excess R²: %{y:.4f}<extra></extra>",
        ))
        abs_peak = r2_abs4.loc[r2_abs4["r_squared"].idxmax()]
        exc_peak = r2_exc4.loc[r2_exc4["r_squared"].idxmax()]
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
# Section 6 — Asymmetric Response
# ============================================================
st.markdown("---")
st.header("6. Asymmetric Response")
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
# Section 7 — Panel Regression (Robustness)
# ============================================================
st.markdown("---")
st.header("7. Panel Regression (Robustness)")

benchmark_panel = st.radio(
    "Benchmark for excess return",
    ["SPY", "QQQ", "Peer Average"],
    horizontal=True, index=2,
    key="sec7_benchmark",
)
benchmark_panel = {"SPY": "SPY", "QQQ": "QQQ", "Peer Average": "peer_avg"}[benchmark_panel]

df_panel_bench = load_peer_data(freq, benchmark_panel)
df_panel_bench_valid = df_panel_bench[df_panel_bench["ETF"].isin(valid_etfs)].copy()

st.markdown(f"""
**Question**: Are these findings robust across all 38 ETFs simultaneously?

Instead of separate regressions per ETF, we stack all ETFs into one panel regression.
Five specifications from simplest to most controlled:
- **Pooled OLS** — ignores ETF identity
- **Entity FE** — controls for each ETF's baseline flow level
- **Entity+Time FE** — also controls for market-wide time trends
- **Entity FE + Excess** — adds relative performance (benchmark: **{_bench_desc[benchmark_panel]}**)
- **Entity FE + Controls** — adds rolling volatility
""")

panel_df = df_panel_bench_valid.dropna(subset=[fc, rc])

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
                    lambda x: "ARK" if x in ETF_NAMES else "Tech Peers")
                colors = fe_df["Source"].map({"ARK": "#1f77b4", "Tech Peers": "#aec7e8"})

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
