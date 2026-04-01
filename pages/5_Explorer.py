"""Page 5: Explorer — Per-ETF interactive analysis for curious readers."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES, PEER_ETF_NAMES, ALL_ETF_NAMES,
)
from analysis import (
    auto_lags, cross_correlation, r_squared_by_lag, r_squared_by_lag_all_etfs,
    seasonality_analysis, seasonality_inflow_outflow,
)

st.set_page_config(page_title="Explorer", layout="wide")
st.title("Explorer: Dive Into Individual ETFs")
st.markdown("""
The previous pages presented aggregate evidence. Here you can **explore each ETF
individually** — its flow distribution, lag structure, and seasonal patterns.
Use the sidebar to select an ETF, frequency, and time range.
""")

# --- Sidebar ---
ETF_FULL_NAMES = {
    "ARKK": "ARK Innovation", "ARKF": "ARK Fintech", "ARKG": "ARK Genomic",
    "ARKX": "ARK Space & Defense", "ARKB": "ARK Bitcoin", "ARKQ": "ARK Autonomous Tech",
    "ARKW": "ARK Next Gen Internet", "PRNT": "3D Printing", "IZRL": "ARK Israel",
}

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2, key="exp_freq")
    selected_etf = st.selectbox(
        "Select ETF", ALL_ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_FULL_NAMES.get(x, x)}",
        key="exp_etf")
    st.markdown("---")
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading data...")
def load_data(freq):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark="peer_avg")


df = load_data(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]

fc = "Fund_Flow" if freq == "D" else "Flow_Sum"
rc = "Return" if freq == "D" else "Return_Cum"
flow_ylabel = "Fund Flow ($M)"
freq_label = {"D": "days", "W": "weeks", "ME": "months", "QE": "quarters"}[freq]

valid_etfs = df.groupby("ETF")[fc].apply(lambda x: x.notna().sum())
valid_etfs = valid_etfs[valid_etfs > 20].index.tolist()
df_valid = df[df["ETF"].isin(valid_etfs)].copy()

# ============================================================
# 1. Flow Distribution
# ============================================================
st.header(f"1. {selected_etf}: Flow Distribution")

etf_dist = df_valid[df_valid["ETF"] == selected_etf][fc].dropna()
if len(etf_dist) > 0:
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=etf_dist, nbinsx=50, marker_color="#1f77b4", opacity=0.8))
        fig.add_vline(x=etf_dist.median(), line_dash="dash", line_color="red",
                      annotation_text="Median", annotation_position="top right")
        fig.add_vline(x=0, line_color="gray", opacity=0.4)
        fig.update_layout(height=350, xaxis_title=flow_ylabel, yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("Mean", f"{etf_dist.mean():.2f}")
        st.metric("Std Dev", f"{etf_dist.std():.2f}")
        st.metric("Skewness", f"{etf_dist.skew():.2f}")
        st.metric("N", f"{len(etf_dist):,}")

# ============================================================
# 2. Cross-Correlogram
# ============================================================
st.header(f"2. {selected_etf}: Cross-Correlogram")
st.markdown(r"""
Pearson correlation between fund flow at time $t$ and return at time $t-k$
for lags $k = -20, \ldots, +20$. Positive values at **negative lags** (return
leads flow) = performance chasing.
""")

etf_data = df_valid[df_valid["ETF"] == selected_etf].copy()
if len(etf_data) > 30:
    etf_sorted = etf_data.set_index("Date").sort_index()
    n_obs = len(etf_sorted[fc].dropna())
    lag_list = auto_lags(n_obs)
    cc = cross_correlation(etf_sorted[fc], etf_sorted[rc], max_lag=max(lag_list))

    fig_cc = go.Figure()
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in cc["correlation"]]
    fig_cc.add_trace(go.Bar(x=cc["lag"], y=cc["correlation"], marker_color=colors))
    fig_cc.add_hline(y=0, line_color="black", line_width=0.5)
    n_obs = len(etf_data[fc].dropna())
    se = 1.96 / np.sqrt(n_obs)
    fig_cc.add_hline(y=se, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cc.add_hline(y=-se, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cc.update_layout(height=350, xaxis_title=f"Lag ({freq_label})",
                         yaxis_title="Correlation",
                         title=f"{selected_etf}: Flow-Return Cross-Correlogram")
    st.plotly_chart(fig_cc, width="stretch")

# ============================================================
# 3. R² by Lag: All ETFs Heatmap
# ============================================================
st.header("3. Lag Structure Heatmap: All ETFs")
st.markdown("""
R² from univariate regressions `Flow(t) ~ Return(t-k)` for each ETF and lag.
Darker colors = stronger predictive power. Look for which ETFs and which lags
show the most concentration.
""")

@st.cache_data(show_spinner="Computing R² by lag...")
def compute_all_r2(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc = "Fund_Flow" if freq == "D" else "Flow_Sum"
    _rc = "Return" if freq == "D" else "Return_Cum"
    _valid = _df.groupby("ETF")[_fc].apply(lambda x: x.notna().sum())
    _valid = _valid[_valid > 20].index.tolist()
    _df = _df[_df["ETF"].isin(_valid)]
    return r_squared_by_lag_all_etfs(_df, _fc, _rc)


all_r2 = compute_all_r2(freq)
if not all_r2.empty and "lag" in all_r2.columns:
    # Pivot long-form (ETF, lag, r_squared) to wide for heatmap
    heatmap_data = all_r2.pivot_table(index="ETF", columns="lag", values="r_squared")
    heatmap_data.columns = [str(int(c)) for c in heatmap_data.columns]

    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
        colorscale="YlOrRd", hovertemplate="ETF: %{y}<br>Lag: %{x}<br>R²: %{z:.4f}<extra></extra>",
    ))
    fig_hm.update_layout(height=max(400, len(heatmap_data) * 18),
                         xaxis_title=f"Lag ({freq_label})", yaxis_title="ETF",
                         title="R² by Lag: All ETFs")
    st.plotly_chart(fig_hm, width="stretch")

# ============================================================
# 4. Seasonality
# ============================================================
st.header(f"4. {selected_etf}: Seasonal Patterns")
st.markdown("""
Average fund flow by calendar month. The **January effect** in fund flows is
well-documented — do ARK ETFs exhibit similar patterns?
""")

if len(etf_data) > 30:
    seas = seasonality_analysis(etf_data, fc)
    if seas is not None and not seas.empty:
        month_names = seas["Month_Name"].tolist()
        mean_vals = seas["Mean"].tolist()
        colors = ["#d62728" if m == "Jan" else "#1f77b4" for m in month_names]
        fig_seas = go.Figure(go.Bar(x=month_names, y=mean_vals, marker_color=colors))
        fig_seas.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_seas.update_layout(height=350, yaxis_title=flow_ylabel,
                               title=f"{selected_etf}: Average Flow by Calendar Month")
        st.plotly_chart(fig_seas, width="stretch")

        # January vs rest t-test
        jan_row = seas[seas["Month"] == 1]
        other_rows = seas[seas["Month"] != 1]
        if not jan_row.empty and not other_rows.empty:
            jan_mean = jan_row.iloc[0]["Mean"]
            other_mean = other_rows["Mean"].mean()
            col1, col2 = st.columns(2)
            col1.metric("January avg", f"{jan_mean:.2f}")
            col2.metric("Other months avg", f"{other_mean:.2f}")
