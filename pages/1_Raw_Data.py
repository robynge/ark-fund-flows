"""Step 1: Look at the raw data."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_all_etfs, add_returns, ETF_NAMES

st.set_page_config(page_title="Step 1: Raw Data", layout="wide")

st.title("Step 1: Look at the Raw Data")
st.markdown("""
Before doing any analysis, let's first understand what the data looks like.

We have **9 ARK ETFs** with daily data: closing price, fund flow (in $M), and volume.
""")


@st.cache_data
def get_data():
    return add_returns(load_all_etfs())


df = get_data()

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)

etf_df = df[df["ETF"] == selected_etf].copy()

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Date Range", f"{etf_df['Date'].min().strftime('%Y-%m')} → {etf_df['Date'].max().strftime('%Y-%m')}")
col2.metric("Trading Days", f"{len(etf_df):,}")
col3.metric("Total Net Flow", f"${etf_df['Fund_Flow'].sum():,.0f}M")
total_ret = (etf_df["Close"].iloc[-1] / etf_df["Close"].iloc[0] - 1) * 100
col4.metric("Total Return", f"{total_ret:+.1f}%")

st.markdown("---")

# --- Chart 1: Price over time ---
st.subheader(f"{selected_etf} — Closing Price")

fig1 = go.Figure(go.Scatter(
    x=etf_df["Date"], y=etf_df["Close"],
    line=dict(color="#1f77b4", width=2),
))
fig1.update_layout(height=350, yaxis_title="Price ($)", margin=dict(l=60, r=40, t=20, b=40))
st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Daily fund flows ---
st.subheader(f"{selected_etf} — Daily Fund Flows")
st.markdown("Each bar is one day's net inflow (green) or outflow (red) in millions of dollars.")

colors = ["#2ca02c" if v >= 0 else "#d62728" for v in etf_df["Fund_Flow"].fillna(0)]
fig2 = go.Figure(go.Bar(
    x=etf_df["Date"], y=etf_df["Fund_Flow"],
    marker_color=colors,
))
fig2.update_layout(height=350, yaxis_title="Fund Flow ($M)", margin=dict(l=60, r=40, t=20, b=40))
st.plotly_chart(fig2, use_container_width=True)

# --- Chart 3: Overlay ---
st.subheader(f"{selected_etf} — Price vs Cumulative Fund Flows")
st.markdown("Cumulative flows show the total money that has entered or left the fund over time.")

etf_df["Cum_Flow"] = etf_df["Fund_Flow"].cumsum()
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Scatter(x=etf_df["Date"], y=etf_df["Close"], name="Price",
                          line=dict(color="#1f77b4", width=2)), secondary_y=False)
fig3.add_trace(go.Scatter(x=etf_df["Date"], y=etf_df["Cum_Flow"], name="Cumulative Flow ($M)",
                          line=dict(color="#ff7f0e", width=2),
                          fill="tozeroy", fillcolor="rgba(255,127,14,0.1)"), secondary_y=True)
fig3.update_layout(height=400, hovermode="x unified",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02),
                   margin=dict(l=60, r=60, t=40, b=40))
fig3.update_yaxes(title_text="Price ($)", secondary_y=False)
fig3.update_yaxes(title_text="Cumulative Flow ($M)", secondary_y=True)
st.plotly_chart(fig3, use_container_width=True)

# --- Raw data table ---
st.subheader("Data Table")
st.dataframe(
    etf_df[["Date", "Fund_Flow", "Open", "High", "Low", "Close", "Volume", "Return"]].sort_values("Date", ascending=False).style.format({
        "Fund_Flow": "{:.2f}", "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
        "Close": "{:.2f}", "Volume": "{:,.0f}", "Return": "{:.4%}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True, height=400,
)

# --- All ETFs summary ---
st.markdown("---")
st.subheader("All ETFs at a Glance")

rows = []
for etf in ETF_NAMES:
    edf = df[df["ETF"] == etf]
    if len(edf) == 0:
        continue
    rows.append({
        "ETF": etf,
        "Start": edf["Date"].min().strftime("%Y-%m-%d"),
        "End": edf["Date"].max().strftime("%Y-%m-%d"),
        "Days": len(edf),
        "Total Flow ($M)": round(edf["Fund_Flow"].sum(), 1),
        "Total Return (%)": round((edf["Close"].iloc[-1] / edf["Close"].iloc[0] - 1) * 100, 1),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.info("👉 **Next Step**: Go to **Step 2: The Problem** to see why we can't just look at raw daily data.")
