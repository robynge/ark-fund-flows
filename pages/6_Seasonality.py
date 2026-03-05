"""Step 6: Seasonality — calendar effects in fund flows."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES
from analysis import seasonality_analysis

st.set_page_config(page_title="Step 6: Seasonality", layout="wide")

st.title("Step 6: Seasonality")
st.markdown("""
Fund flows are not random throughout the year. Investors tend to **reallocate at specific times**:
- **January / Q1**: Beginning-of-year portfolio rebalancing
- **Year-end**: Tax-loss harvesting, annual reviews

If seasonality is strong, we need to control for it in our regressions to avoid confounding effects.
""")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)


@st.cache_data
def load_data():
    return get_prepared_data(freq="D", zscore_type="full")


@st.cache_data
def load_monthly():
    return get_prepared_data(freq="ME", zscore_type="full")


df = load_data()
df_m = load_monthly()

etf_df = df[df["ETF"] == selected_etf].copy()

# ================================================================
# Monthly pattern
# ================================================================
st.subheader(f"{selected_etf}: Average Daily Flow by Calendar Month")
st.markdown("Bars show the average daily fund flow for each month, across all years in the data.")

seasonal = seasonality_analysis(etf_df, "Fund_Flow")

if len(seasonal) > 0:
    colors = ["#2ca02c" if m >= 0 else "#d62728" for m in seasonal["Mean"]]
    fig = go.Figure(go.Bar(
        x=seasonal["Month_Name"], y=seasonal["Mean"],
        marker_color=colors,
        error_y=dict(type="data", array=seasonal["Std"] / seasonal["Count"] ** 0.5, visible=True),
        text=[f"${m:.1f}M" for m in seasonal["Mean"]],
        textposition="outside",
    ))
    fig.update_layout(height=420, yaxis_title="Average Daily Flow ($M)",
                      margin=dict(l=60, r=40, t=20, b=40))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# January effect test
# ================================================================
st.markdown("---")
st.subheader(f"{selected_etf}: January Effect Test")
st.markdown("""
Is January statistically different from other months?
We use a **t-test** to compare January flows vs. the rest of the year.
""")

etf_flows = etf_df.dropna(subset=["Fund_Flow"])
jan = etf_flows[etf_flows["Date"].dt.month == 1]["Fund_Flow"]
other = etf_flows[etf_flows["Date"].dt.month != 1]["Fund_Flow"]

if len(jan) > 5 and len(other) > 5:
    t_stat, p_val = stats.ttest_ind(jan, other, equal_var=False)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("January Avg ($M/day)", f"{jan.mean():.2f}")
    col2.metric("Other Months Avg ($M/day)", f"{other.mean():.2f}")
    col3.metric("Difference", f"{jan.mean() - other.mean():+.2f}")
    col4.metric("t-test p-value", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success(f"✓ January flows are **significantly different** from other months (p = {p_val:.4f})")
    else:
        st.info(f"✗ No statistically significant January effect (p = {p_val:.4f})")

# ================================================================
# Quarterly pattern
# ================================================================
st.markdown("---")
st.subheader(f"{selected_etf}: Average Flow by Quarter")

etf_q = etf_df.copy()
etf_q["Q"] = etf_q["Date"].dt.quarter
q_stats = etf_q.groupby("Q")["Fund_Flow"].agg(["mean", "std", "count"]).reset_index()
q_labels = {1: "Q1\n(Jan-Mar)", 2: "Q2\n(Apr-Jun)", 3: "Q3\n(Jul-Sep)", 4: "Q4\n(Oct-Dec)"}

q_colors = ["#2ca02c" if m >= 0 else "#d62728" for m in q_stats["mean"]]
fig_q = go.Figure(go.Bar(
    x=[q_labels[q] for q in q_stats["Q"]], y=q_stats["mean"],
    marker_color=q_colors,
    error_y=dict(type="data", array=q_stats["std"] / q_stats["count"] ** 0.5, visible=True),
    text=[f"${m:.1f}M" for m in q_stats["mean"]],
    textposition="outside",
))
fig_q.update_layout(height=380, yaxis_title="Average Daily Flow ($M)",
                    margin=dict(l=60, r=40, t=20, b=40))
fig_q.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig_q, use_container_width=True)

# ================================================================
# Heatmap: all ETFs × all months
# ================================================================
st.markdown("---")
st.subheader("Monthly Flow Patterns — All ETFs")
st.markdown("Heatmap showing average monthly flows across all ARK ETFs. Red = outflows, Blue = inflows.")

heatmap_data = []
for etf in ETF_NAMES:
    edf = df[df["ETF"] == etf]
    s = seasonality_analysis(edf, "Fund_Flow")
    for _, row in s.iterrows():
        heatmap_data.append({"ETF": etf, "Month": row["Month_Name"],
                             "Mean_Flow": row["Mean"], "Month_Num": row["Month"]})

hm_df = pd.DataFrame(heatmap_data)
if len(hm_df) > 0:
    pivot = hm_df.pivot_table(index="ETF", columns="Month_Num", values="Mean_Flow")
    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    pivot.columns = month_names[:len(pivot.columns)]

    fig_hm = px.imshow(pivot, color_continuous_scale="RdBu_r", aspect="auto",
                       labels=dict(x="Month", y="ETF", color="Avg Flow ($M)"))
    fig_hm.update_layout(height=380, margin=dict(l=60, r=40, t=20, b=40))
    st.plotly_chart(fig_hm, use_container_width=True)

# ================================================================
# Year-over-year flow by month (time series)
# ================================================================
st.markdown("---")
st.subheader(f"{selected_etf}: Monthly Flows Over Time")
st.markdown("Each bar is one month's total fund flow. This shows how flows evolve year-by-year.")

etf_m = df_m[df_m["ETF"] == selected_etf].copy()
m_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in etf_m["Flow_Sum"].fillna(0)]

fig_mt = go.Figure(go.Bar(
    x=etf_m["Date"], y=etf_m["Flow_Sum"],
    marker_color=m_colors,
))
fig_mt.update_layout(height=380, yaxis_title="Monthly Flow ($M)",
                     margin=dict(l=60, r=40, t=20, b=40))
st.plotly_chart(fig_mt, use_container_width=True)

st.info("👉 **Next Step**: Go to **Step 7: Conclusions** for a summary of all findings.")
