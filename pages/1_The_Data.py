"""Page 1: The Data — Price-flow time series, summary statistics, ARK vs peers."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import get_prepared_data_with_peers, ETF_NAMES, ALL_ETF_NAMES

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="The Data", layout="wide")
st.title("The Data")
st.markdown("""
Before testing any hypothesis, let's look at the raw data. We study **9 ARK ETFs**
and **29 tech peer ETFs** using Bloomberg daily net fund flows (creation/redemption),
OHLCV prices, and monthly AUM from **2014 to 2026**.
""")

# --- Sidebar ---
ETF_NAMES_DISPLAY = {
    "ARKK": "ARK Innovation", "ARKF": "ARK Fintech", "ARKG": "ARK Genomic",
    "ARKX": "ARK Space & Defense", "ARKB": "ARK Bitcoin", "ARKQ": "ARK Autonomous Tech",
    "ARKW": "ARK Next Gen Internet", "PRNT": "3D Printing", "IZRL": "ARK Israel",
}

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2,
    )
    selected_etf = st.selectbox(
        "Select ETF", ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_NAMES_DISPLAY.get(x, x)}",
    )


@st.cache_data(show_spinner="Loading data...")
def load_data(freq):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark="SPY")


df = load_data(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]

fc = "Fund_Flow" if freq == "D" else "Flow_Sum"
rc = "Return" if freq == "D" else "Return_Cum"

# ============================================================
# 1. Price + Fund Flow Chart
# ============================================================
st.header("1. Price and Fund Flow")
st.markdown(f"""
The chart below shows **{selected_etf}**'s price (candlestick, right axis) alongside
net fund flows (bars, left axis). Green = inflows, red = outflows.
This is the raw phenomenon we're investigating: **do capital flows follow price movements?**
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
# 2. Summary Statistics
# ============================================================
st.header("2. Summary Statistics")
st.markdown(r"""
Descriptive statistics for the daily panel of 9 ARK ETFs. The **between/within
decomposition** shows how much variation comes from differences *across* ETFs
vs. changes *within* each ETF over time. High within-% justifies entity fixed effects.
""")

summary_f = RESULTS / "table_1_summary.csv"
bw_f = RESULTS / "table_1_bw.csv"

col1, col2 = st.columns(2)
if summary_f.exists():
    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(pd.read_csv(summary_f), width="stretch", hide_index=True)
if bw_f.exists():
    with col2:
        st.subheader("Between/Within Decomposition")
        st.dataframe(pd.read_csv(bw_f), width="stretch", hide_index=True)

# ============================================================
# 3. ARK vs Peers
# ============================================================
st.header("3. ARK vs. Peer Fund Flows")
st.markdown("""
Are ARK flows different from peers? The plot below compares the average fund flow
for ARK ETFs vs. tech peer ETFs (20-period moving average).
""")

ark_flows = df[df["ETF"].isin(ETF_NAMES)].groupby("Date")[fc].mean()
peer_etfs = [e for e in df["ETF"].unique() if e not in ETF_NAMES]
if peer_etfs:
    peer_flows = df[df["ETF"].isin(peer_etfs)].groupby("Date")[fc].mean()
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=ark_flows.index, y=ark_flows.rolling(20, min_periods=5).mean(),
        name="ARK ETFs (20-period MA)", line=dict(color="#d62728", width=2)))
    fig_comp.add_trace(go.Scatter(
        x=peer_flows.index, y=peer_flows.rolling(20, min_periods=5).mean(),
        name="Peer ETFs (20-period MA)", line=dict(color="#1f77b4", width=2)))
    fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_comp.update_layout(
        height=400, yaxis_title="Average Fund Flow ($M)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_comp, width="stretch")

st.info("**Next →** *The Evidence*: Do these visual patterns hold up statistically?")
