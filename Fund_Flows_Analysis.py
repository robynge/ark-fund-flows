"""ARK ETF Fund Flows vs Stock Price Analysis"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES
from analysis import cross_correlation, lag_regression, granger_causality_test

st.set_page_config(page_title="ARK Fund Flows Analysis", layout="wide")
st.title("ARK ETF: Fund Flows vs Stock Price")

# --- Sidebar ---
with st.sidebar:
    selected_etf = st.selectbox("ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["D", "W", "ME", "QE"],
                        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
                        index=2)
    max_lag = st.slider("Max Lag", 3, 30, {"D": 20, "W": 13, "ME": 12, "QE": 8}.get(freq, 12))


@st.cache_data
def load(freq):
    return get_prepared_data(freq=freq, zscore_type="full")


df = load(freq)
etf_df = df[df["ETF"] == selected_etf].copy().sort_values("Date")

if freq == "D":
    fc, rc, fc_z, rc_z = "Fund_Flow", "Return", "Fund_Flow_Z", "Return_Z"
else:
    fc, rc, fc_z, rc_z = "Flow_Sum", "Return_Cum", "Flow_Sum_Z", "Return_Cum_Z"

# ============================================================
# 1. Price vs Cumulative Flows
# ============================================================
st.subheader(f"{selected_etf} — Price vs Cumulative Flows")

etf_df["Cum_Flow"] = etf_df[fc].cumsum()
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=etf_df["Date"], y=etf_df["Close"], name="Price",
                          line=dict(color="#1f77b4", width=2)), secondary_y=False)
fig1.add_trace(go.Scatter(x=etf_df["Date"], y=etf_df["Cum_Flow"], name="Cum Flow ($M)",
                          line=dict(color="#ff7f0e", width=2),
                          fill="tozeroy", fillcolor="rgba(255,127,14,0.1)"), secondary_y=True)
fig1.update_layout(height=400, hovermode="x unified",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02),
                   margin=dict(l=60, r=60, t=30, b=30))
fig1.update_yaxes(title_text="Price ($)", secondary_y=False)
fig1.update_yaxes(title_text="Cumulative Flow ($M)", secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# 2. Normalized overlay
# ============================================================
st.subheader("Normalized: Z-score Flows vs Returns")

smooth = st.slider("Smoothing Window", 1, 30, {"D": 21, "W": 4, "ME": 3, "QE": 1}.get(freq, 3))
plot_df = etf_df.copy()
if smooth > 1:
    plot_df[fc_z] = plot_df[fc_z].rolling(smooth, min_periods=1).mean()
    plot_df[rc_z] = plot_df[rc_z].rolling(smooth, min_periods=1).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[rc_z], name="Return (Z)", line=dict(color="#1f77b4", width=1.5)))
fig2.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[fc_z], name="Flow (Z)", line=dict(color="#ff7f0e", width=1.5)))
fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
fig2.update_layout(height=350, yaxis_title="Z-score", hovermode="x unified",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02),
                   margin=dict(l=60, r=40, t=30, b=30))
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 3. Cross-Correlogram
# ============================================================
st.subheader("Cross-Correlation by Lag")
st.caption("Positive lag = past returns → current flows | Negative lag = current flows → future returns")

etf_ts = etf_df.set_index("Date").sort_index()
cc = cross_correlation(etf_ts[fc_z], etf_ts[rc_z], max_lag)

if len(cc) > 0:
    colors = ["#2ca02c" if p < 0.05 else "#c7c7c7" for p in cc["p_value"]]
    fig3 = go.Figure(go.Bar(x=cc["lag"], y=cc["correlation"], marker_color=colors))
    n = len(etf_ts[[fc_z, rc_z]].dropna())
    if n > 0:
        ci = 1.96 / n ** 0.5
        fig3.add_hline(y=ci, line_dash="dot", line_color="red", opacity=0.5)
        fig3.add_hline(y=-ci, line_dash="dot", line_color="red", opacity=0.5)
    fig3.add_hline(y=0, line_color="gray", opacity=0.4)
    freq_label = {"D": "days", "W": "weeks", "ME": "months", "QE": "quarters"}[freq]
    fig3.update_layout(height=350, xaxis_title=f"Lag ({freq_label})", yaxis_title="Correlation",
                       margin=dict(l=60, r=40, t=20, b=30))
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# 4. Regression + Granger
# ============================================================
st.subheader("Regression & Granger Causality")

lag_opts = {"D": [1, 5, 21], "W": [1, 4, 13], "ME": [1, 3, 6], "QE": [1, 2]}
lags = lag_opts[freq]

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"**OLS: Flow(t) ~ Lagged Returns**")
    result = lag_regression(etf_df, fc, rc, lags)
    if result:
        st.metric("R²", f"{result['r_squared']:.4f}")
        coef = result["coefficients"]
        lag_coef = coef[coef["Variable"].str.contains("Return_lag")].copy()
        lag_coef["Sig"] = lag_coef["p_value"].apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))
        st.dataframe(lag_coef[["Variable", "Coefficient", "p_value", "Sig"]].style.format(
            {"Coefficient": "{:.2f}", "p_value": "{:.4f}"}), hide_index=True, use_container_width=True)

with col_right:
    st.markdown("**Granger Causality**")
    gc = granger_causality_test(etf_df, fc, rc, min(max_lag, 6))
    if len(gc) > 0:
        gc["Sig"] = gc["p_value"].apply(lambda p: "✓" if p < 0.05 else "")
        st.dataframe(gc[["lag", "direction", "F_statistic", "p_value", "Sig"]].style.format(
            {"F_statistic": "{:.2f}", "p_value": "{:.4f}"}), hide_index=True, use_container_width=True)

# ============================================================
# 5. All ETFs comparison
# ============================================================
st.markdown("---")
st.subheader("All ETFs Comparison")

rows = []
for etf in ETF_NAMES:
    edf = df[df["ETF"] == etf]
    r = lag_regression(edf, fc, rc, lags)
    if r is None:
        continue
    gc_e = granger_causality_test(edf, fc, rc, min(max_lag, 6))
    rf_sig = "✓" if len(gc_e) > 0 and (gc_e[gc_e["direction"] == "Returns → Flows"]["p_value"] < 0.05).any() else "✗"
    fr_sig = "✓" if len(gc_e) > 0 and (gc_e[gc_e["direction"] == "Flows → Returns"]["p_value"] < 0.05).any() else "✗"
    rows.append({"ETF": etf, "R²": r["r_squared"], "F_p": r["f_pvalue"],
                 "Ret→Flow": rf_sig, "Flow→Ret": fr_sig, "N": r["n_obs"]})

summary = pd.DataFrame(rows)
st.dataframe(summary.style.format({"R²": "{:.4f}", "F_p": "{:.4f}"}),
             hide_index=True, use_container_width=True)
