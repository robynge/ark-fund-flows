"""Z-score normalized flows & returns over time."""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import get_prepared_data, ETF_NAMES

st.set_page_config(page_title="Normalized Trends", layout="wide")
st.title("Normalized Trends: Flows vs Returns")
st.markdown("Z-score normalized fund flows and returns plotted together to compare on the same scale.")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["D", "W", "ME", "QE"],
                        format_func=lambda x: {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}[x])
    zscore_type = st.radio("Normalization", ["full", "rolling"],
                           format_func=lambda x: {"full": "Full-sample Z-score", "rolling": "Rolling Z-score"}[x])
    smoothing = st.slider("Smoothing Window", 1, 60, 1,
                          help="Rolling average window applied to normalized series")


@st.cache_data
def load_data(freq, zscore_type):
    return get_prepared_data(freq=freq, zscore_type=zscore_type)


df = load_data(freq, zscore_type)
etf_df = df[df["ETF"] == selected_etf].copy().sort_values("Date")

# Determine column names
if freq == "D":
    flow_z = "Fund_Flow_Z" if zscore_type == "full" else "Fund_Flow_RZ"
    return_z = "Return_Z" if zscore_type == "full" else "Return_RZ"
else:
    flow_z = "Flow_Sum_Z" if zscore_type == "full" else "Flow_Sum_RZ"
    return_z = "Return_Cum_Z" if zscore_type == "full" else "Return_Cum_RZ"

# Apply smoothing
if smoothing > 1:
    etf_df[flow_z] = etf_df[flow_z].rolling(smoothing, min_periods=1).mean()
    etf_df[return_z] = etf_df[return_z].rolling(smoothing, min_periods=1).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=etf_df["Date"], y=etf_df[return_z],
    name="Return (Z-score)", line=dict(color="#1f77b4", width=1.5),
))
fig.add_trace(go.Scatter(
    x=etf_df["Date"], y=etf_df[flow_z],
    name="Fund Flow (Z-score)", line=dict(color="#ff7f0e", width=1.5),
))
fig.update_layout(
    height=550,
    yaxis_title="Z-score",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=60, r=60, t=40, b=40),
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
st.plotly_chart(fig, use_container_width=True)

# Show correlation
valid = etf_df[[flow_z, return_z]].dropna()
if len(valid) > 5:
    corr = valid[flow_z].corr(valid[return_z])
    st.metric("Contemporaneous Correlation (Z-scored)", f"{corr:.4f}")

# Scatter plot
st.subheader("Scatter: Normalized Flows vs Returns")
fig2 = go.Figure(go.Scatter(
    x=valid[return_z], y=valid[flow_z],
    mode="markers", marker=dict(size=4, opacity=0.4, color="#1f77b4"),
))
fig2.update_layout(
    height=400,
    xaxis_title="Return (Z-score)",
    yaxis_title="Fund Flow (Z-score)",
    margin=dict(l=60, r=60, t=40, b=40),
)
st.plotly_chart(fig2, use_container_width=True)
