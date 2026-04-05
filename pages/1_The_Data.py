"""Page 1: The Data — Price-flow time series, cumulative flows, summary stats."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (sidebar_freq, sidebar_etf, load_data, get_cols,
                      ETF_NAMES, FREQ_LABELS)

st.set_page_config(page_title="The Data", layout="wide")
st.title("The Data")
st.markdown("""
Before testing any hypothesis, let's look at the raw data. We study **6 ARK ETFs**
and their **peer ETFs** using Bloomberg daily net fund flows (creation/redemption),
OHLCV prices, and monthly AUM from **2014 to 2026**.
""")

freq = sidebar_freq(key="data_freq")
selected_etf = sidebar_etf(key="data_etf")

df = load_data(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]
fc, rc = get_cols(freq)
period = FREQ_LABELS[freq]

# ============================================================
# 1. Price and Fund Flow
# ============================================================
st.header("1. Price and Fund Flow")
st.info("""
**Fund Flow Definition**: Net capital entering or leaving an ETF each day through
the creation/redemption process, in **millions of USD**, from Bloomberg.
Positive = net inflows (new shares created), negative = net outflows (shares redeemed).
""")

etf_ts = df[df["ETF"] == selected_etf].sort_values("Date")

if len(etf_ts) > 0 and "Close_Last" in etf_ts.columns:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in etf_ts[fc].fillna(0)]
    fig.add_trace(
        go.Bar(x=etf_ts["Date"], y=etf_ts[fc], marker_color=bar_colors,
               name="Fund Flow", opacity=0.7),
        secondary_y=False)
    if all(c in etf_ts.columns for c in ["Open_First", "High_Max", "Low_Min", "Close_Last"]):
        fig.add_trace(
            go.Candlestick(x=etf_ts["Date"], open=etf_ts["Open_First"],
                           high=etf_ts["High_Max"], low=etf_ts["Low_Min"],
                           close=etf_ts["Close_Last"], name="Price",
                           increasing_line_color="#1f77b4",
                           decreasing_line_color="#aec7e8"),
            secondary_y=True)
    fig.update_yaxes(title_text="Fund Flow ($M)", secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True)
    fig.update_layout(
        height=500, hovermode="x unified", xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=60, t=40, b=30))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 1b. Cumulative Fund Flow
# ============================================================
st.subheader("Cumulative Fund Flow")
if len(etf_ts) > 0:
    etf_cum = etf_ts.copy()
    etf_cum["Cumulative_Flow"] = etf_cum[fc].cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=etf_cum["Date"], y=etf_cum["Cumulative_Flow"],
        mode="lines", line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%Y-%m}<br>Cumulative: %{y:,.0f}M<extra></extra>"))
    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_cum.update_layout(height=350, yaxis_title="Cumulative Fund Flow ($M)",
                          title=f"{selected_etf}: Cumulative Net Fund Flow")
    st.plotly_chart(fig_cum, width="stretch")

# ============================================================
# 2. Summary Statistics
# ============================================================
st.header("2. Summary Statistics")

from summary_stats import panel_summary
ark_df = df[df["ETF"].isin(ETF_NAMES)]
if len(ark_df) > 0:
    stats = panel_summary(ark_df, flow_col=fc, return_col=rc)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(stats["overall"], width="stretch", hide_index=True)
    with col2:
        st.subheader("Between/Within Decomposition")
        st.dataframe(stats["between_within"], width="stretch", hide_index=True)

# ============================================================
# 3. ARK vs Peers
# ============================================================
st.header("3. ARK vs. Peer Fund Flows")

flow_display = st.radio("Flow metric", ["Raw $ (millions)", "% of AUM"],
                         horizontal=True, key="ark_vs_peer_flow")
fc_comp = fc if flow_display == "Raw $ (millions)" else "Flow_Pct"
fc_label = "Fund Flow ($M)" if flow_display == "Raw $ (millions)" else "Fund Flow (% of AUM)"

# For % of AUM, filter out extreme values from tiny-AUM ETFs (e.g. VOLT AUM=$25)
df_comp = df.copy()
if fc_comp == "Flow_Pct":
    df_comp = df_comp[df_comp["Flow_Pct"].between(-100, 100)]

ark_flows = df_comp[df_comp["ETF"].isin(ETF_NAMES)].groupby("Date")[fc_comp].mean()
peer_etfs = [e for e in df_comp["ETF"].unique() if e not in ETF_NAMES]
if peer_etfs:
    peer_flows = df_comp[df_comp["ETF"].isin(peer_etfs)].groupby("Date")[fc_comp].mean()
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=ark_flows.index, y=ark_flows.rolling(20, min_periods=5).mean(),
        name="ARK ETFs (20-period MA)", line=dict(color="#d62728", width=2),
        hovertemplate="%{x|%Y-%m}<br>Avg flow: %{y:.2f}<extra>ARK</extra>"))
    fig_comp.add_trace(go.Scatter(
        x=peer_flows.index, y=peer_flows.rolling(20, min_periods=5).mean(),
        name="Peer ETFs (20-period MA)", line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%Y-%m}<br>Avg flow: %{y:.2f}<extra>Peers</extra>"))
    fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_comp.update_layout(height=400, yaxis_title=f"Average {fc_label}",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_comp, width="stretch")

st.info(f"**Next** → *The Evidence*: Do these patterns hold up statistically? (currently showing **{FREQ_LABELS[freq]}** data)")
