"""Cross-correlation at multiple lag horizons."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import get_prepared_data, ETF_NAMES
from analysis import cross_correlation_all_etfs, r_squared_by_lag

st.set_page_config(page_title="Lag Analysis", layout="wide")
st.title("Lag Analysis: Cross-Correlation")
st.markdown("""
**Positive lag k**: correlation between flow today and return k periods ago (past performance → flows)
**Negative lag k**: correlation between flow today and return k periods ahead (flows → future performance)
""")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["D", "W", "ME", "QE"],
                        format_func=lambda x: {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}[x])
    max_lag = st.slider("Max Lag", 5, 60, 20)


@st.cache_data
def load_data(freq):
    return get_prepared_data(freq=freq, zscore_type="full")


df = load_data(freq)

flow_col = "Fund_Flow_Z" if freq == "D" else "Flow_Sum_Z"
return_col = "Return_Z" if freq == "D" else "Return_Cum_Z"


@st.cache_data
def compute_cc(freq, max_lag):
    return cross_correlation_all_etfs(df, flow_col, return_col, max_lag)


cc = compute_cc(freq, max_lag)

# Single ETF correlogram
st.subheader(f"{selected_etf}: Cross-Correlogram")

etf_cc = cc[cc["ETF"] == selected_etf]
if len(etf_cc) > 0:
    colors = ["#2ca02c" if p < 0.05 else "#aec7e8" for p in etf_cc["p_value"]]
    fig = go.Figure(go.Bar(
        x=etf_cc["lag"], y=etf_cc["correlation"],
        marker_color=colors,
    ))
    fig.update_layout(
        height=400,
        xaxis_title=f"Lag ({{'D':'days','W':'weeks','ME':'months','QE':'quarters'}}['{freq}'])",
        yaxis_title="Correlation",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    # Add significance bands (approximate 95% CI)
    n = len(df[df["ETF"] == selected_etf])
    if n > 0:
        ci = 1.96 / (n ** 0.5)
        fig.add_hline(y=ci, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=-ci, line_dash="dot", line_color="red", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = p < 0.05. Red dashed lines = approximate 95% confidence interval.")

# Heatmap across all ETFs
st.subheader("Cross-Correlation Heatmap (All ETFs)")

if len(cc) > 0:
    pivot = cc.pivot_table(index="ETF", columns="lag", values="correlation")
    fig2 = px.imshow(
        pivot,
        color_continuous_scale="RdBu_r",
        zmin=-0.3, zmax=0.3,
        aspect="auto",
        labels=dict(x="Lag", y="ETF", color="Corr"),
    )
    fig2.update_layout(height=400, margin=dict(l=60, r=60, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

# R² by lag
st.subheader(f"{selected_etf}: R² by Single Lag")
st.markdown("R² from simple regression flow(t) ~ return(t-k) for each lag k.")

etf_df = df[df["ETF"] == selected_etf]

flow_col_raw = "Fund_Flow" if freq == "D" else "Flow_Sum"
return_col_raw = "Return" if freq == "D" else "Return_Cum"


@st.cache_data
def compute_r2(etf_name, freq, max_lag):
    edf = df[df["ETF"] == etf_name]
    return r_squared_by_lag(edf, flow_col_raw, return_col_raw, range(1, max_lag + 1))


r2_df = compute_r2(selected_etf, freq, max_lag)

if len(r2_df) > 0:
    sig_colors = ["#2ca02c" if p < 0.05 else "#aec7e8" for p in r2_df["p_value"]]
    fig3 = go.Figure(go.Bar(
        x=r2_df["lag"], y=r2_df["r_squared"],
        marker_color=sig_colors,
    ))
    fig3.update_layout(
        height=350,
        xaxis_title="Lag",
        yaxis_title="R²",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)
