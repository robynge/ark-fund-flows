"""Page 3: The Dynamics — LP impulse response, asymmetry, regimes, drawdowns."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from _shared import (sidebar_freq, sidebar_etf, load_data, get_cols,
                      get_lp_horizon, ETF_NAMES, FREQ_LABELS)

st.set_page_config(page_title="The Dynamics", layout="wide")
st.title("The Dynamics: How Do Investors Chase?")

freq = sidebar_freq(key="dynamics_freq")
fc, rc = get_cols(freq)
period = FREQ_LABELS[freq]
horizon = get_lp_horizon(freq)

df = load_data(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]

from local_projection import (local_projection, local_projection_asymmetric,
                               local_projection_subsample)
from analysis import compute_etf_drawdowns, drawdown_flow_analysis, drawdown_flow_regression

# ============================================================
# 1. Impulse Response
# ============================================================
st.header("1. Impulse Response: How the Effect Unfolds")
st.markdown(rf"""
At each horizon $h = 0, 1, \ldots, {horizon}$ {period}, we estimate:

$$
\text{{Flow}}_{{i,t+h}} = \alpha_i^{{(h)}} + \beta_h \cdot \text{{Return}}_{{i,t}} + \varepsilon
$$

The sequence $\{{\hat\beta_h\}}$ shows how a return shock propagates through fund flows.
""")

@st.cache_data(show_spinner="Computing impulse response...")
def compute_lp(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    return local_projection(ark, _fc, _rc, max_horizon=get_lp_horizon(freq))

lp = compute_lp(freq)
if not lp.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["ci_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["ci_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.2)", name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["beta"], mode="lines+markers",
                             line=dict(color="#1f77b4", width=2), marker=dict(size=4),
                             name="Point estimate",
                             hovertemplate=f"h=%{{x}} {period}<br>β=%{{y:.2f}}<extra></extra>"))
    sig = lp[lp["p_value"] < 0.05]
    if not sig.empty:
        fig.add_trace(go.Scatter(x=sig["horizon"], y=sig["beta"], mode="markers",
                                 marker=dict(color="#d62728", size=6), name="p < 0.05",
                                 hovertemplate=f"h=%{{x}} {period}<br>β=%{{y:.2f}} (sig.)<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title=f"Horizon ({period})",
                      yaxis_title="Response of Fund Flow ($M)",
                      title=f"Impulse Response: Return Shock → Fund Flow ({period})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 2. Asymmetric Response
# ============================================================
st.header("2. Do Investors React Differently to Gains vs. Losses?")
st.markdown(rf"""
We split each {period[:-1]}'s return into a positive part and a negative part,
then track how fund flows respond to each over {horizon} {period}:

$$
\text{{Flow}}_{{i,t+h}} = \alpha_i^{{(h)}} + \beta_h^+ \cdot \max(R_{{i,t}}, 0)
+ \beta_h^- \cdot \min(R_{{i,t}}, 0) + \varepsilon
$$

- **Blue**: flow response after a **+1% gain**
- **Red**: flow response after a **-1% loss**
""")

@st.cache_data(show_spinner="Computing asymmetric LP...")
def compute_asym(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    return local_projection_asymmetric(ark, _fc, _rc, max_horizon=get_lp_horizon(freq))

asym = compute_asym(freq)
if not asym.empty:
    asym["beta_neg_plot"] = -asym["beta_neg"]
    asym["ci_lower_neg_plot"] = -asym["ci_upper_neg"]
    asym["ci_upper_neg_plot"] = -asym["ci_lower_neg"]

    fig = go.Figure()
    # Positive
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_upper_pos"],
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_lower_pos"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["beta_pos"],
                             mode="lines+markers", line=dict(color="#1f77b4", width=2),
                             marker=dict(size=4), name="+1% gain → flow",
                             hovertemplate=f"h=%{{x}} {period}<br>Flow: %{{y:.1f}}M<extra></extra>"))
    # Negative (negated)
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_upper_neg_plot"],
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_lower_neg_plot"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(214,39,40,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["beta_neg_plot"],
                             mode="lines+markers", line=dict(color="#d62728", width=2),
                             marker=dict(size=4), name="-1% loss → flow",
                             hovertemplate=f"h=%{{x}} {period}<br>Flow: %{{y:.1f}}M<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title=f"Horizon ({period})",
                      yaxis_title="Response of Fund Flow ($M)",
                      title="Asymmetric Response: Gains vs. Losses",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 3. Bull vs Bear
# ============================================================
st.header("3. Market Regime: Bull vs. Bear")

@st.cache_data(show_spinner="Computing subsample LP...")
def compute_subsample(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    h = get_lp_horizon(freq)
    periods = {"bull": ("2020-01-01", "2021-12-31"), "bear": ("2022-01-01", "2024-12-31")}
    result = {}
    for name, (start, end) in periods.items():
        sub = ark[(ark["Date"] >= start) & (ark["Date"] <= end)]
        if len(sub) > 50:
            result[name] = local_projection(sub, _fc, _rc, max_horizon=min(h, 30))
    return result

subs = compute_subsample(freq)
if subs:
    fig = go.Figure()
    colors = {"bull": "#1f77b4", "bear": "#d62728"}
    labels = {"bull": "Bull (2020-2021)", "bear": "Bear (2022-2024)"}
    for name, data in subs.items():
        if data.empty:
            continue
        c = colors[name]
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["ci_upper"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["ci_lower"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=f"rgba({','.join(str(int(c.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["beta"],
                                 mode="lines+markers", line=dict(color=c, width=2),
                                 marker=dict(size=4), name=labels[name],
                                 hovertemplate=f"h=%{{x}}<br>β=%{{y:.2f}}<extra>{labels[name]}</extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title=f"Horizon ({period})",
                      yaxis_title="Response of Fund Flow ($M)",
                      title="Performance Chasing by Market Regime",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 4. Drawdown Event Study
# ============================================================
st.header("4. What Happens After Crashes?")

dd_etf = sidebar_etf(key="dd_etf")

@st.cache_data(show_spinner="Computing drawdowns...")
def compute_dd(freq):
    _df = load_data(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    dd = compute_etf_drawdowns(_df, _rc, min_depth_pct=10.0)
    dd_flow = drawdown_flow_analysis(_df, dd, _fc) if len(dd) > 0 else pd.DataFrame()
    return _df, dd, dd_flow

df_dd, dd_all, dd_flow = compute_dd(freq)

if len(dd_all) > 0:
    etf_dd = dd_all[dd_all["ETF"] == dd_etf]
    etf_prices = df_dd[df_dd["ETF"] == dd_etf].sort_values("Date")
    if len(etf_prices) > 10:
        price_idx = (1 + etf_prices.set_index("Date")[rc].dropna()).cumprod() * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=price_idx.index, y=price_idx.values,
                                    mode="lines", name="Price Index",
                                    line=dict(color="#1f77b4", width=1.5)))
        for _, row in etf_dd.iterrows():
            fig_dd.add_vrect(x0=row["peak_date"], x1=row["trough_date"],
                             fillcolor="red", opacity=0.15, line_width=0,
                             annotation_text=f"{row['depth_pct']:.0f}%",
                             annotation_position="top left", annotation_font_size=9)
        fig_dd.update_layout(height=400, yaxis_title="Price Index (base=100)",
                             title=f"{dd_etf}: Drawdown Episodes (>=10%)")
        st.plotly_chart(fig_dd, width="stretch")

    if len(dd_flow) > 0:
        dd_reg = drawdown_flow_regression(dd_flow)
        if len(dd_reg) > 0:
            st.subheader("Post-Drawdown Flow Regression")
            st.dataframe(dd_reg.style.format({
                "β_Depth": "{:.4f}", "β_Depth_p": "{:.4f}",
                "β_Duration": "{:.4f}", "β_Duration_p": "{:.4f}", "R²": "{:.4f}"}),
                hide_index=True, width="stretch")

st.info("**Next** → *Robustness*: Are these findings reliable?")
