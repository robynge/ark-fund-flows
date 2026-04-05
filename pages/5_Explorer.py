"""Page 5: Explorer — Per-ETF interactive analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (sidebar_freq, sidebar_etf, load_data, get_cols,
                      ETF_NAMES, FREQ_LABELS)

st.set_page_config(page_title="Explorer", layout="wide")
st.title("Explorer: Dive Into Individual ETFs")

freq = sidebar_freq(key="exp_freq")
selected_etf = st.sidebar.selectbox(
    "Select ETF",
    list(load_data(freq)["ETF"].unique()),
    index=0, key="exp_etf")

df = load_data(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]
fc, rc = get_cols(freq)
period = FREQ_LABELS[freq]

valid = df.groupby("ETF")[fc].apply(lambda x: x.notna().sum())
valid = valid[valid > 20].index.tolist()
df_valid = df[df["ETF"].isin(valid)].copy()

from analysis import auto_lags, cross_correlation, r_squared_by_lag_all_etfs, seasonality_analysis

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
        fig.update_layout(height=350, xaxis_title="Fund Flow ($M)", yaxis_title="Count")
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
etf_data = df_valid[df_valid["ETF"] == selected_etf].copy()
if len(etf_data) > 30:
    etf_sorted = etf_data.set_index("Date").sort_index()
    n_obs = len(etf_sorted[fc].dropna())
    lag_list = auto_lags(n_obs)
    cc = cross_correlation(etf_sorted[fc], etf_sorted[rc], max_lag=max(lag_list))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in cc["correlation"]]
    fig_cc = go.Figure()
    fig_cc.add_trace(go.Bar(x=cc["lag"], y=cc["correlation"], marker_color=colors,
                            hovertemplate="Lag: %{x}<br>Corr: %{y:.4f}<extra></extra>"))
    fig_cc.add_hline(y=0, line_color="black", line_width=0.5)
    se = 1.96 / np.sqrt(n_obs)
    fig_cc.add_hline(y=se, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cc.add_hline(y=-se, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cc.update_layout(height=350, xaxis_title=f"Lag ({period})",
                         yaxis_title="Correlation",
                         title=f"{selected_etf}: Flow-Return Cross-Correlogram")
    st.plotly_chart(fig_cc, width="stretch")

# ============================================================
# 3. Lag Heatmap
# ============================================================
st.header("3. Lag Structure Heatmap: All ETFs")

@st.cache_data(show_spinner="Computing R² by lag...")
def compute_heatmap(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    _valid = _df.groupby("ETF")[_fc].apply(lambda x: x.notna().sum())
    _valid = _valid[_valid > 20].index.tolist()
    _df = _df[_df["ETF"].isin(_valid)]
    return r_squared_by_lag_all_etfs(_df, _fc, _rc)

all_r2 = compute_heatmap(freq)
if not all_r2.empty and "lag" in all_r2.columns:
    hm = all_r2.pivot_table(index="ETF", columns="lag", values="r_squared")
    hm.columns = [str(int(c)) for c in hm.columns]
    fig_hm = go.Figure(data=go.Heatmap(
        z=hm.values, x=hm.columns, y=hm.index, colorscale="YlOrRd",
        hovertemplate="ETF: %{y}<br>Lag: %{x}<br>R²: %{z:.4f}<extra></extra>"))
    fig_hm.update_layout(height=max(400, len(hm) * 18),
                         xaxis_title=f"Lag ({period})", yaxis_title="ETF")
    st.plotly_chart(fig_hm, width="stretch")

# ============================================================
# 4. Seasonality
# ============================================================
st.header(f"4. {selected_etf}: Seasonal Patterns")
if len(etf_data) > 30:
    seas = seasonality_analysis(etf_data, fc)
    if seas is not None and not seas.empty:
        month_names = seas["Month_Name"].tolist()
        mean_vals = seas["Mean"].tolist()
        colors = ["#d62728" if m == "Jan" else "#1f77b4" for m in month_names]
        fig_seas = go.Figure(go.Bar(x=month_names, y=mean_vals, marker_color=colors,
                                     hovertemplate="%{x}<br>Avg flow: %{y:.2f}<extra></extra>"))
        fig_seas.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_seas.update_layout(height=350, yaxis_title="Fund Flow ($M)",
                               title=f"{selected_etf}: Average Flow by Calendar Month")
        st.plotly_chart(fig_seas, width="stretch")
