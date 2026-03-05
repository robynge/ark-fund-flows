"""Seasonality analysis: monthly/quarterly flow patterns, January effect."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import get_prepared_data, ETF_NAMES
from analysis import seasonality_analysis

st.set_page_config(page_title="Seasonality", layout="wide")
st.title("Seasonality Analysis")
st.markdown("Examining monthly and quarterly patterns in fund flows — particularly beginning-of-year reallocation.")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    show_all = st.checkbox("Show all ETFs comparison", value=False)


@st.cache_data
def load_data():
    return get_prepared_data(freq="D", zscore_type="full")


df = load_data()

# Monthly seasonality for selected ETF
st.subheader(f"{selected_etf}: Average Daily Flow by Month")

etf_df = df[df["ETF"] == selected_etf]
seasonal = seasonality_analysis(etf_df, "Fund_Flow")

if len(seasonal) > 0:
    colors = ["#2ca02c" if m >= 0 else "#d62728" for m in seasonal["Mean"]]
    fig = go.Figure(go.Bar(
        x=seasonal["Month_Name"], y=seasonal["Mean"],
        marker_color=colors,
        error_y=dict(type="data", array=seasonal["Std"] / seasonal["Count"] ** 0.5, visible=True),
    ))
    fig.update_layout(
        height=400,
        yaxis_title="Average Daily Flow ($M)",
        xaxis_title="Month",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(seasonal[["Month_Name", "Mean", "Median", "Std", "Count"]].rename(
        columns={"Month_Name": "Month", "Mean": "Mean Flow ($M)", "Median": "Median Flow ($M)",
                 "Std": "Std Dev", "Count": "# Days"}
    ).style.format({"Mean Flow ($M)": "{:.2f}", "Median Flow ($M)": "{:.2f}", "Std Dev": "{:.2f}"}),
        use_container_width=True, hide_index=True)

# January vs rest
st.markdown("---")
st.subheader(f"{selected_etf}: January Effect Test")

etf_df_copy = etf_df.copy()
etf_df_copy["Is_January"] = etf_df_copy["Date"].dt.month == 1
jan_flows = etf_df_copy[etf_df_copy["Is_January"]]["Fund_Flow"].dropna()
other_flows = etf_df_copy[~etf_df_copy["Is_January"]]["Fund_Flow"].dropna()

if len(jan_flows) > 5 and len(other_flows) > 5:
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(jan_flows, other_flows, equal_var=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("January Avg Flow ($M)", f"{jan_flows.mean():.2f}")
    col2.metric("Other Months Avg Flow ($M)", f"{other_flows.mean():.2f}")
    col3.metric("t-test p-value", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("January flows are statistically different from other months (p < 0.05).")
    else:
        st.info("No statistically significant January effect detected.")

# Quarterly patterns
st.markdown("---")
st.subheader(f"{selected_etf}: Average Flow by Quarter")

etf_df_q = etf_df.copy()
etf_df_q["Quarter"] = etf_df_q["Date"].dt.quarter
q_stats = etf_df_q.groupby("Quarter")["Fund_Flow"].agg(["mean", "median", "std", "count"]).reset_index()
q_stats.columns = ["Quarter", "Mean", "Median", "Std", "Count"]
q_labels = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
q_stats["Quarter_Label"] = q_stats["Quarter"].map(q_labels)

q_colors = ["#2ca02c" if m >= 0 else "#d62728" for m in q_stats["Mean"]]
fig2 = go.Figure(go.Bar(
    x=q_stats["Quarter_Label"], y=q_stats["Mean"],
    marker_color=q_colors,
))
fig2.update_layout(
    height=350,
    yaxis_title="Average Daily Flow ($M)",
    margin=dict(l=60, r=60, t=40, b=40),
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig2, use_container_width=True)

# All ETFs comparison heatmap
if show_all:
    st.markdown("---")
    st.subheader("Monthly Flow Patterns Across All ETFs")

    heatmap_data = []
    for etf in ETF_NAMES:
        edf = df[df["ETF"] == etf]
        s = seasonality_analysis(edf, "Fund_Flow")
        for _, row in s.iterrows():
            heatmap_data.append({"ETF": etf, "Month": row["Month_Name"], "Mean_Flow": row["Mean"],
                                 "Month_Num": row["Month"]})
    hm_df = pd.DataFrame(heatmap_data)
    if len(hm_df) > 0:
        pivot = hm_df.pivot_table(index="ETF", columns="Month_Num", values="Mean_Flow")
        pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]

        fig3 = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(x="Month", y="ETF", color="Avg Flow ($M)"),
        )
        fig3.update_layout(height=400, margin=dict(l=60, r=60, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)
