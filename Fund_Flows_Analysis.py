"""ARK ETF Fund Flows Analysis Dashboard"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_all_etfs, add_returns, ETF_NAMES

st.set_page_config(
    page_title="ARK ETF Fund Flows Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ARK ETF Fund Flows Analysis")
st.markdown("Analyzing the relationship between ETF price performance and fund flows.")


@st.cache_data
def get_data():
    df = load_all_etfs()
    df = add_returns(df)
    return df


df = get_data()

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)

etf_df = df[df["ETF"] == selected_etf].copy()

# Summary stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Date Range", f"{etf_df['Date'].min().strftime('%Y-%m')} → {etf_df['Date'].max().strftime('%Y-%m')}")
with col2:
    st.metric("Trading Days", f"{len(etf_df):,}")
with col3:
    total_flow = etf_df["Fund_Flow"].sum()
    st.metric("Total Net Flow ($M)", f"{total_flow:,.1f}")
with col4:
    total_return = (etf_df["Close"].iloc[-1] / etf_df["Close"].iloc[0] - 1) * 100
    st.metric("Total Return", f"{total_return:+.1f}%")

st.markdown("---")

# Dual-axis chart: Price vs Cumulative Flows
st.subheader(f"{selected_etf}: Price vs Cumulative Fund Flows")

etf_plot = etf_df.copy()
etf_plot["Cumulative_Flow"] = etf_plot["Fund_Flow"].cumsum()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=etf_plot["Date"], y=etf_plot["Close"],
        name="Close Price", line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=etf_plot["Date"], y=etf_plot["Cumulative_Flow"],
        name="Cumulative Flow ($M)", line=dict(color="#ff7f0e", width=2),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.1)",
    ),
    secondary_y=True,
)

fig.update_layout(
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=60, r=60, t=40, b=40),
)
fig.update_yaxes(title_text="Close Price ($)", secondary_y=False)
fig.update_yaxes(title_text="Cumulative Flow ($M)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Daily flows bar chart
st.subheader(f"{selected_etf}: Daily Fund Flows")

flow_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in etf_plot["Fund_Flow"].fillna(0)]
fig2 = go.Figure(
    go.Bar(
        x=etf_plot["Date"], y=etf_plot["Fund_Flow"],
        marker_color=flow_colors, name="Daily Flow",
    )
)
fig2.update_layout(
    height=350,
    yaxis_title="Fund Flow ($M)",
    margin=dict(l=60, r=60, t=40, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

# Summary statistics table
st.subheader("Summary Statistics Across All ETFs")

summary_rows = []
for etf in ETF_NAMES:
    edf = df[df["ETF"] == etf]
    if len(edf) == 0:
        continue
    summary_rows.append({
        "ETF": etf,
        "Start": edf["Date"].min().strftime("%Y-%m-%d"),
        "End": edf["Date"].max().strftime("%Y-%m-%d"),
        "Days": len(edf),
        "Total Flow ($M)": round(edf["Fund_Flow"].sum(), 1),
        "Avg Daily Flow ($M)": round(edf["Fund_Flow"].mean(), 2),
        "Flow Std ($M)": round(edf["Fund_Flow"].std(), 2),
        "Total Return (%)": round((edf["Close"].iloc[-1] / edf["Close"].iloc[0] - 1) * 100, 1),
        "Avg Daily Return (%)": round(edf["Return"].mean() * 100, 4),
        "Return Std (%)": round(edf["Return"].std() * 100, 2),
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
