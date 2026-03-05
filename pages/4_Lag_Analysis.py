"""Step 4: Lag analysis — does past performance predict future flows?"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES
from analysis import cross_correlation_all_etfs, r_squared_by_lag

st.set_page_config(page_title="Step 4: Lag Analysis", layout="wide")

st.title("Step 4: Lag Analysis")
st.markdown("""
The key insight from the literature: investors don't react immediately to performance.
There is a **lag** — they see the returns, think about it, then move their money.

We test this by computing the correlation between **today's fund flow** and **returns from k periods ago**.

- **Positive lag k**: does the return from k periods ago predict today's flow? *(Performance chasing)*
- **Negative lag k**: does today's flow predict the return k periods ahead? *(Flow impact on price)*
""")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["W", "ME", "QE"],
                        format_func={"W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
                        index=1)
    max_lag = st.slider("Max Lag", 3, 30,
                        {"W": 20, "ME": 12, "QE": 8}.get(freq, 12))

freq_label = {"W": "weeks", "ME": "months", "QE": "quarters"}[freq]


@st.cache_data
def load_data(freq):
    return get_prepared_data(freq=freq, zscore_type="full")


df = load_data(freq)

flow_z = "Flow_Sum_Z"
return_z = "Return_Cum_Z"
flow_raw = "Flow_Sum"
return_raw = "Return_Cum"

# ================================================================
# Cross-correlogram for selected ETF
# ================================================================
st.subheader(f"{selected_etf}: Cross-Correlogram ({freq_label})")
st.markdown(f"""
Each bar shows the correlation between flow(t) and return(t−k).
- Bars on the **right** (positive lag): past returns → current flows
- Bars on the **left** (negative lag): current flows → future returns
- **Green** bars are statistically significant (p < 0.05)
""")


@st.cache_data
def compute_cc(freq, max_lag):
    return cross_correlation_all_etfs(df, flow_z, return_z, max_lag)


cc = compute_cc(freq, max_lag)
etf_cc = cc[cc["ETF"] == selected_etf]

if len(etf_cc) > 0:
    colors = ["#2ca02c" if p < 0.05 else "#c7c7c7" for p in etf_cc["p_value"]]

    fig = go.Figure(go.Bar(x=etf_cc["lag"], y=etf_cc["correlation"], marker_color=colors))

    # Confidence interval
    n = len(df[df["ETF"] == selected_etf].dropna(subset=[flow_z, return_z]))
    if n > 0:
        ci = 1.96 / (n ** 0.5)
        fig.add_hline(y=ci, line_dash="dot", line_color="red", opacity=0.6,
                      annotation_text="95% CI", annotation_position="top right")
        fig.add_hline(y=-ci, line_dash="dot", line_color="red", opacity=0.6)

    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    # Annotations
    fig.add_annotation(x=max_lag * 0.7, y=etf_cc["correlation"].max() * 0.9,
                       text="← Past returns predict flows", showarrow=False,
                       font=dict(size=11, color="gray"))
    fig.add_annotation(x=-max_lag * 0.7, y=etf_cc["correlation"].max() * 0.9,
                       text="Flows predict future returns →", showarrow=False,
                       font=dict(size=11, color="gray"))

    fig.update_layout(
        height=420,
        xaxis_title=f"Lag ({freq_label})",
        yaxis_title="Correlation",
        margin=dict(l=60, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight strongest lag
    pos_lags = etf_cc[etf_cc["lag"] > 0]
    if len(pos_lags) > 0:
        best = pos_lags.loc[pos_lags["correlation"].abs().idxmax()]
        st.markdown(f"""
        **Strongest positive-lag correlation**: lag = **{int(best['lag'])} {freq_label}**,
        r = **{best['correlation']:.4f}**, p = {best['p_value']:.4f}
        {"✓ Significant" if best['p_value'] < 0.05 else "✗ Not significant"}
        """)

# ================================================================
# Scatter plots at key lags
# ================================================================
st.markdown("---")
st.subheader(f"Scatter Plots at Key Lags")
st.markdown("Visualize the relationship at specific lag horizons.")

etf_df = df[df["ETF"] == selected_etf].set_index("Date").sort_index()

lag_choices = {"W": [1, 4, 13], "ME": [1, 3, 6], "QE": [1, 2, 4]}
key_lags = lag_choices.get(freq, [1, 3, 6])

cols = st.columns(len(key_lags))

for i, lag in enumerate(key_lags):
    with cols[i]:
        shifted = etf_df[return_raw].shift(lag)
        valid = pd.DataFrame({"flow": etf_df[flow_raw], "ret_lagged": shifted}).dropna()
        if len(valid) < 10:
            st.write(f"Lag {lag}: insufficient data")
            continue
        c, p = stats.pearsonr(valid["ret_lagged"], valid["flow"])

        fig_s = go.Figure(go.Scatter(
            x=valid["ret_lagged"] * 100, y=valid["flow"],
            mode="markers", marker=dict(size=5, opacity=0.5, color="#1f77b4"),
        ))
        fig_s.update_layout(
            height=300,
            title=f"Lag = {lag} {freq_label}<br><sup>r={c:.3f}, p={p:.3f}</sup>",
            xaxis_title=f"Return {lag} {freq_label} ago (%)",
            yaxis_title="Flow ($M)",
            margin=dict(l=50, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ================================================================
# R² by lag — finding the optimal horizon
# ================================================================
st.markdown("---")
st.subheader(f"{selected_etf}: R² by Lag — Finding the Optimal Horizon")
st.markdown("""
R² measures how much of the variation in flows is explained by lagged returns.
Higher R² = stronger predictive power at that lag.
""")


@st.cache_data
def compute_r2(etf, freq, max_lag):
    edf = df[df["ETF"] == etf]
    return r_squared_by_lag(edf, flow_raw, return_raw, range(1, max_lag + 1))


r2_df = compute_r2(selected_etf, freq, max_lag)

if len(r2_df) > 0:
    sig_colors = ["#2ca02c" if p < 0.05 else "#c7c7c7" for p in r2_df["p_value"]]
    fig_r2 = go.Figure(go.Bar(
        x=r2_df["lag"], y=r2_df["r_squared"],
        marker_color=sig_colors,
        text=[f"{r:.3f}" for r in r2_df["r_squared"]],
        textposition="outside",
    ))
    fig_r2.update_layout(
        height=380,
        xaxis_title=f"Lag ({freq_label})",
        yaxis_title="R²",
        margin=dict(l=60, r=40, t=20, b=40),
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    best_r2 = r2_df.loc[r2_df["r_squared"].idxmax()]
    st.markdown(f"**Best single lag**: {int(best_r2['lag'])} {freq_label} → R² = {best_r2['r_squared']:.4f}")

# ================================================================
# Heatmap: all ETFs
# ================================================================
st.markdown("---")
st.subheader("Cross-Correlation Heatmap: All ETFs")
st.markdown("Compare the lag structure across all ARK ETFs.")

if len(cc) > 0:
    # Filter to positive lags only (past performance → flows)
    cc_pos = cc[cc["lag"] >= 0]
    pivot = cc_pos.pivot_table(index="ETF", columns="lag", values="correlation")

    fig_hm = px.imshow(
        pivot, color_continuous_scale="RdBu_r", zmin=-0.4, zmax=0.4,
        aspect="auto", labels=dict(x=f"Lag ({freq_label})", y="ETF", color="Correlation"),
    )
    fig_hm.update_layout(height=380, margin=dict(l=60, r=40, t=20, b=40))
    st.plotly_chart(fig_hm, use_container_width=True)

st.success("""
**Key Takeaway**: At monthly and quarterly horizons, we can see that past returns are
positively correlated with future flows — confirming the **performance-chasing** hypothesis.
The effect is strongest at 1-3 month / 1-2 quarter lags.
""")

st.info("👉 **Next Step**: Go to **Step 5: Regression & Causality** for formal statistical tests.")
