"""Step 2: Why daily data is too noisy."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_all_etfs, add_returns, ETF_NAMES

st.set_page_config(page_title="Step 2: The Problem", layout="wide")

st.title("Step 2: The Problem — Daily Data is Too Noisy")
st.markdown("""
If we try to directly compare daily fund flows with daily returns, we get **almost nothing**.

The reason is simple: daily fund flows are extremely volatile — they spike up and down randomly.
Any underlying relationship is buried in noise.
""")


@st.cache_data
def get_data():
    return add_returns(load_all_etfs())


df = get_data()

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)

etf_df = df[df["ETF"] == selected_etf].copy()

# --- Show the two series side by side ---
st.subheader(f"{selected_etf} — Daily Returns vs Daily Fund Flows")
st.markdown("Notice how both series jump around wildly. It's hard to see any pattern.")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=["Daily Return (%)", "Daily Fund Flow ($M)"])

fig.add_trace(go.Scatter(
    x=etf_df["Date"], y=etf_df["Return"] * 100,
    line=dict(color="#1f77b4", width=1), name="Return",
), row=1, col=1)

flow_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in etf_df["Fund_Flow"].fillna(0)]
fig.add_trace(go.Bar(
    x=etf_df["Date"], y=etf_df["Fund_Flow"],
    marker_color=flow_colors, name="Flow",
), row=2, col=1)

fig.update_layout(height=500, showlegend=False, margin=dict(l=60, r=40, t=40, b=40))
fig.update_yaxes(title_text="Return (%)", row=1, col=1)
fig.update_yaxes(title_text="Flow ($M)", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- Scatter plot: the non-relationship ---
st.subheader("Scatter Plot: Daily Return vs Daily Flow")
st.markdown("If there were a strong relationship, we'd see a clear trend. Instead, it's a cloud of points.")

valid = etf_df.dropna(subset=["Return", "Fund_Flow"])
corr, pval = stats.pearsonr(valid["Return"], valid["Fund_Flow"])

fig2 = go.Figure(go.Scatter(
    x=valid["Return"] * 100, y=valid["Fund_Flow"],
    mode="markers", marker=dict(size=3, opacity=0.3, color="#1f77b4"),
))
fig2.update_layout(
    height=450,
    xaxis_title="Daily Return (%)",
    yaxis_title="Daily Fund Flow ($M)",
    margin=dict(l=60, r=40, t=20, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
col1.metric("Correlation", f"{corr:.4f}")
col2.metric("p-value", f"{pval:.4f}")

if abs(corr) < 0.1:
    st.error(f"Correlation is only **{corr:.4f}** — essentially zero. Daily data shows no meaningful relationship.")
elif abs(corr) < 0.3:
    st.warning(f"Correlation is **{corr:.4f}** — very weak. Some signal exists but is mostly noise.")

# --- Show all ETFs ---
st.markdown("---")
st.subheader("Daily Correlation Across All ETFs")
st.markdown("Same story across the board — daily correlations are weak.")

rows = []
for etf in ETF_NAMES:
    edf = df[df["ETF"] == etf].dropna(subset=["Return", "Fund_Flow"])
    if len(edf) < 30:
        continue
    c, p = stats.pearsonr(edf["Return"], edf["Fund_Flow"])
    rows.append({"ETF": etf, "Correlation": round(c, 4), "p-value": round(p, 4),
                 "Significant?": "Yes" if p < 0.05 else "No", "N": len(edf)})

import pandas as pd
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- The two problems ---
st.markdown("---")
st.subheader("Why Doesn't This Work?")

st.markdown("""
Two issues make it hard to see the relationship in daily data:

**1. Scale Problem**
- Daily returns are in percentages (e.g., -2% to +3%)
- Daily fund flows are in millions of dollars (e.g., -$200M to +$500M)
- They're on completely different scales, making direct comparison meaningless

**2. Timing Problem (Lag Effect)**
- Investors don't react instantly to price changes
- It takes time for people to see the performance, make decisions, and move money
- The reaction happens over **weeks, months, or quarters** — not the same day

**Solution**: We need to (a) **normalize** the data to make them comparable, and (b) look at **lagged** relationships at longer time horizons.
""")

st.info("👉 **Next Step**: Go to **Step 3: Normalize & Aggregate** to see how we fix these issues.")
