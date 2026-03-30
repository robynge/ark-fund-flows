"""Data Overview: Price-Flow time series, summary statistics, and S&T scatter."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import get_prepared_data_with_peers, ETF_NAMES, ALL_ETF_NAMES

RESULTS_V2 = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview")

# --- Sidebar ---
ETF_FULL_NAMES = {
    "ARKK": "ARK Innovation", "ARKF": "ARK Fintech Innovation",
    "ARKG": "ARK Genomic Revolution", "ARKX": "ARK Space & Defense",
    "ARKB": "ARK 21Shares Bitcoin", "ARKQ": "ARK Autonomous Tech",
    "ARKW": "ARK Next Gen Internet", "PRNT": "3D Printing ETF",
    "IZRL": "ARK Israel Innovation",
}

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2,
    )
    selected_etf = st.selectbox(
        "Select ETF", ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_FULL_NAMES.get(x, x)}",
    )


@st.cache_data(show_spinner="Loading data...")
def load_data(freq):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark="SPY")


df = load_data(freq)
ARK_START = pd.Timestamp("2014-10-31")
df = df[df["Date"] >= ARK_START]

if freq == "D":
    fc, rc = "Fund_Flow", "Return"
else:
    fc, rc = "Flow_Sum", "Return_Cum"


# ============================================================
# 1. Price + Fund Flow Chart (Sirri-Tufano inspired visualization)
# ============================================================
st.header("1. Price and Fund Flow Time Series")

st.markdown(f"""
The chart below shows the **dual-axis relationship** between {selected_etf}'s
price (candlestick, right axis) and net fund flows (bars, left axis). This
visualization is central to the Sirri & Tufano (1998) narrative: do capital
flows follow price movements?

Green bars = net inflows; Red bars = net outflows.
""")

etf_ts = df[df["ETF"] == selected_etf].copy().sort_values("Date")

if len(etf_ts) > 0 and "Close_Last" in etf_ts.columns:
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    # Fund flow bars (primary axis)
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728"
                  for v in etf_ts[fc].fillna(0)]
    fig_dual.add_trace(
        go.Bar(x=etf_ts["Date"], y=etf_ts[fc],
               marker_color=bar_colors, name="Fund Flow", opacity=0.7),
        secondary_y=False,
    )

    # Price candlestick (secondary axis)
    if all(c in etf_ts.columns for c in ["Open_First", "High_Max", "Low_Min", "Close_Last"]):
        fig_dual.add_trace(
            go.Candlestick(
                x=etf_ts["Date"],
                open=etf_ts["Open_First"], high=etf_ts["High_Max"],
                low=etf_ts["Low_Min"], close=etf_ts["Close_Last"],
                name="Price", increasing_line_color="#1f77b4",
                decreasing_line_color="#aec7e8",
            ),
            secondary_y=True,
        )

    flow_ylabel = "Fund Flow ($M)" if fc != "Flow_Pct" else "Fund Flow (% AUM)"
    fig_dual.update_yaxes(title_text=flow_ylabel, secondary_y=False)
    fig_dual.update_yaxes(title_text="Price ($)", secondary_y=True)
    fig_dual.update_layout(
        height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=60, t=40, b=30),
        hovermode="x unified", xaxis_rangeslider_visible=True,
        xaxis_hoverformat="%b %Y",
        title=f"{selected_etf}: Fund Flow & Price",
    )

    st.plotly_chart(fig_dual, use_container_width=True)
else:
    st.warning(f"No data available for {selected_etf}")


# ============================================================
# 2. Summary Statistics (Table 1)
# ============================================================
st.header("2. Summary Statistics")

st.markdown("""
Descriptive statistics for the daily panel of 9 ARK ETFs. The **between/within
decomposition** shows how much variation is driven by differences *across* ETFs
(between) vs. changes *over time within* each ETF (within). High within-variance
justifies the use of entity fixed effects in panel regressions.
""")

summary_f = RESULTS_V2 / "table_1_summary.csv"
bw_f = RESULTS_V2 / "table_1_bw.csv"

col1, col2 = st.columns(2)

if summary_f.exists():
    summary = pd.read_csv(summary_f)
    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(
            summary.style.format({c: "{:.2f}" for c in summary.columns if c != "Variable" and c != "N"}),
            use_container_width=True, hide_index=True,
        )

if bw_f.exists():
    bw = pd.read_csv(bw_f)
    with col2:
        st.subheader("Between/Within Decomposition")
        st.dataframe(
            bw.style.format({
                "Overall_SD": "{:.2f}", "Between_SD": "{:.2f}",
                "Within_SD": "{:.2f}", "Within_Pct": "{:.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )


# ============================================================
# 3. S&T Scatter: Performance Rank vs Flow
# ============================================================
st.header("3. Performance Rank vs. Fund Flow")

st.markdown(r"""
Following **Sirri & Tufano (1998, Figure 1)**, we plot the **fractional performance
rank** against fund flow growth. The heavy red line shows 20-bin averages.

A convex (upward-curving) relationship indicates that top performers attract
disproportionately more capital — the hallmark of performance chasing.

$$
\text{Rank}_{i,t} = \frac{\text{rank of } R_{i,t} \text{ within cross-section at time } t}{N_t}
$$
""")

scatter_f = RESULTS_V2 / "figure_st1_scatter.csv"
if scatter_f.exists():
    scatter = pd.read_csv(scatter_f)

    # 20-bin averages
    scatter["rank_bin"] = pd.cut(scatter["RANK"], bins=20, labels=False) + 1
    bin_means = scatter.groupby("rank_bin").agg(
        rank_mid=("RANK", "mean"),
        flow_mean=("Flow_Pct", "mean"),
        flow_se=("Flow_Pct", "sem"),
    ).dropna()

    fig_scatter = go.Figure()

    # Raw scatter
    fig_scatter.add_trace(go.Scattergl(
        x=scatter["RANK"], y=scatter["Flow_Pct"],
        mode="markers", marker=dict(size=3, opacity=0.1, color="#1f77b4"),
        name="Individual observations", showlegend=True,
    ))

    # Bin averages
    fig_scatter.add_trace(go.Scatter(
        x=bin_means["rank_mid"], y=bin_means["flow_mean"],
        mode="lines+markers",
        line=dict(color="#d62728", width=3),
        marker=dict(size=8, color="#d62728"),
        name="20-bin average",
    ))

    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.update_layout(
        height=450,
        xaxis_title="Performance Rank (fractional, 0 = worst, 1 = best)",
        yaxis_title="Fund Flow (% of AUM)",
        title="Performance Rank vs. Fund Flow Growth (Monthly)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Run `python -m experiments.new_runner` to generate scatter data.")


# ============================================================
# 4. Flow comparison: ARK vs Peers
# ============================================================
st.header("4. ARK vs. Peer Fund Flows")

ark_flows = df[df["ETF"].isin(ETF_NAMES)].groupby("Date")[fc].mean()
peer_etfs = [e for e in df["ETF"].unique() if e not in ETF_NAMES]
if peer_etfs:
    peer_flows = df[df["ETF"].isin(peer_etfs)].groupby("Date")[fc].mean()

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=ark_flows.index, y=ark_flows.rolling(20, min_periods=5).mean(),
        name="ARK ETFs (20-period MA)", line=dict(color="#d62728", width=2),
    ))
    fig_comp.add_trace(go.Scatter(
        x=peer_flows.index, y=peer_flows.rolling(20, min_periods=5).mean(),
        name="Peer ETFs (20-period MA)", line=dict(color="#1f77b4", width=2),
    ))
    fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_comp.update_layout(
        height=400,
        yaxis_title="Average Fund Flow ($M)",
        title="Average Fund Flow: ARK vs. Tech Peers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
