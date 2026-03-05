"""Step 3: Normalize and aggregate to reveal the signal."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES

st.set_page_config(page_title="Step 3: Normalize & Aggregate", layout="wide")

st.title("Step 3: Normalize & Aggregate")
st.markdown("""
To fix the two problems from Step 2, we do two things:

1. **Z-score Normalize**: subtract the mean and divide by standard deviation.
   This puts both flows and returns on the same scale (in units of standard deviations).

2. **Aggregate to Lower Frequencies**: instead of daily, look at weekly, monthly, or quarterly totals.
   This smooths out the daily noise and lets the underlying relationship emerge.
""")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)

# ================================================================
# Part A: Effect of Normalization
# ================================================================
st.subheader("Part A: Z-score Normalization")
st.markdown("""
**What is a Z-score?** For each value, we compute: `Z = (value - mean) / std_dev`.
A Z-score of +2 means "2 standard deviations above average". This makes flows and returns directly comparable.
""")


@st.cache_data
def load_daily():
    return get_prepared_data(freq="D", zscore_type="full")


df_daily = load_daily()
etf_daily = df_daily[df_daily["ETF"] == selected_etf].copy()

# Before vs After normalization
fig_norm = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                         subplot_titles=["Before: Raw Values (Different Scales)",
                                         "After: Z-score Normalized (Same Scale)"])

# Before
fig_norm.add_trace(go.Scatter(x=etf_daily["Date"], y=etf_daily["Return"] * 100,
                              name="Return (%)", line=dict(color="#1f77b4", width=1)), row=1, col=1)
fig_norm.add_trace(go.Scatter(x=etf_daily["Date"], y=etf_daily["Fund_Flow"],
                              name="Flow ($M)", line=dict(color="#ff7f0e", width=1)), row=1, col=1)

# After — apply rolling mean to smooth for visibility
smooth = 21
etf_daily["Flow_Z_smooth"] = etf_daily["Fund_Flow_Z"].rolling(smooth, min_periods=1).mean()
etf_daily["Return_Z_smooth"] = etf_daily["Return_Z"].rolling(smooth, min_periods=1).mean()

fig_norm.add_trace(go.Scatter(x=etf_daily["Date"], y=etf_daily["Return_Z_smooth"],
                              name="Return (Z, 21d avg)", line=dict(color="#1f77b4", width=1.5)), row=2, col=1)
fig_norm.add_trace(go.Scatter(x=etf_daily["Date"], y=etf_daily["Flow_Z_smooth"],
                              name="Flow (Z, 21d avg)", line=dict(color="#ff7f0e", width=1.5)), row=2, col=1)

fig_norm.update_layout(height=600, margin=dict(l=60, r=40, t=40, b=40),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
fig_norm.update_yaxes(title_text="Raw Value", row=1, col=1)
fig_norm.update_yaxes(title_text="Z-score", row=2, col=1)
st.plotly_chart(fig_norm, use_container_width=True)

st.markdown("After normalization and smoothing, you can start to see the two series moving together.")

# ================================================================
# Part B: Effect of Aggregation
# ================================================================
st.markdown("---")
st.subheader("Part B: Aggregation to Different Frequencies")
st.markdown("""
Aggregating from daily to weekly/monthly/quarterly reduces noise dramatically.
Watch how the correlation **increases** as we move to lower frequencies.
""")

freqs = [("D", "Daily"), ("W", "Weekly"), ("ME", "Monthly"), ("QE", "Quarterly")]
corr_results = []


@st.cache_data
def load_freq(f):
    return get_prepared_data(freq=f, zscore_type="full")


fig_agg = make_subplots(rows=2, cols=2, subplot_titles=[f[1] for f in freqs],
                        vertical_spacing=0.15, horizontal_spacing=0.1)

positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (freq_code, freq_name), (row, col) in zip(freqs, positions):
    df_f = load_freq(freq_code)
    etf_f = df_f[df_f["ETF"] == selected_etf]

    if freq_code == "D":
        fc, rc = "Fund_Flow_Z", "Return_Z"
    else:
        fc, rc = "Flow_Sum_Z", "Return_Cum_Z"

    valid = etf_f[[fc, rc]].dropna()
    if len(valid) > 5:
        c, p = stats.pearsonr(valid[fc], valid[rc])
        corr_results.append({"Frequency": freq_name, "Correlation": round(c, 4),
                             "p-value": round(p, 4), "N": len(valid)})

        fig_agg.add_trace(go.Scatter(
            x=valid[rc], y=valid[fc],
            mode="markers", marker=dict(size=4, opacity=0.5, color="#1f77b4"),
            name=freq_name, showlegend=False,
        ), row=row, col=col)

        # Add annotation with correlation
        fig_agg.add_annotation(
            text=f"r = {c:.3f} (p={p:.3f})",
            xref=f"x{positions.index((row, col)) + 1}" if positions.index((row, col)) > 0 else "x",
            yref=f"y{positions.index((row, col)) + 1}" if positions.index((row, col)) > 0 else "y",
            x=0.5, y=0.95, xanchor="center", yanchor="top",
            xref=f"x{(row-1)*2+col} domain", yref=f"y{(row-1)*2+col} domain",
            showarrow=False, font=dict(size=13, color="red"),
        )

fig_agg.update_layout(height=600, margin=dict(l=60, r=40, t=40, b=40))
st.plotly_chart(fig_agg, use_container_width=True)

# Correlation comparison table
st.subheader("Correlation Increases with Aggregation")
if corr_results:
    corr_df = pd.DataFrame(corr_results)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    fig_bar = go.Figure(go.Bar(
        x=corr_df["Frequency"], y=corr_df["Correlation"],
        marker_color=["#aec7e8", "#6baed6", "#3182bd", "#08519c"],
        text=[f"{c:.3f}" for c in corr_df["Correlation"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=350,
        yaxis_title="Correlation",
        title="Contemporaneous Correlation by Frequency",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# All ETFs at monthly
st.markdown("---")
st.subheader("Monthly Correlation Across All ETFs")

df_m = load_freq("ME")
rows = []
for etf in ETF_NAMES:
    edf = df_m[df_m["ETF"] == etf][["Flow_Sum_Z", "Return_Cum_Z"]].dropna()
    if len(edf) < 10:
        continue
    c, p = stats.pearsonr(edf["Flow_Sum_Z"], edf["Return_Cum_Z"])
    rows.append({"ETF": etf, "Monthly Corr": round(c, 4), "p-value": round(p, 4),
                 "Significant?": "✓" if p < 0.05 else "✗", "N": len(edf)})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.success("""
**Key Takeaway**: Normalization + aggregation reveals a real positive correlation between
flows and returns. But this is *contemporaneous* correlation — they move together in the
same period. The next question is: **does past performance predict future flows?**
That requires looking at *lagged* relationships.
""")

st.info("👉 **Next Step**: Go to **Step 4: Lag Analysis** to test if past returns predict future flows.")
