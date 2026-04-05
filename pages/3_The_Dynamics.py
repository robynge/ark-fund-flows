"""Page 3: The Dynamics — LP impulse response, asymmetry, regimes, drawdowns."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="The Dynamics", layout="wide")
st.title("The Dynamics: How Do Investors Chase?")
st.markdown("""
Sirri & Tufano (1998) established *that* investors chase performance using annual data.
Our **daily** data lets us answer a richer set of questions: **How quickly** does the
effect build? **How long** does it last? Is **chasing gains** stronger than **fleeing losses**?
Does the pattern change in **bull vs. bear** markets? What happens after **crashes**?
""")

# ============================================================
# 1. Impulse Response
# ============================================================
st.header("1. Impulse Response: How the Effect Unfolds")
st.markdown(r"""
Using **Local Projection** (Jordà, 2005), we estimate the response of fund flows
to a return shock at each horizon $h = 0, 1, \ldots, 40$ trading days:

$$
\text{Flow}_{i,t+h} = \alpha_i^{(h)} + \beta_h \cdot \text{Return}_{i,t}
+ \gamma_h' X_{i,t} + \varepsilon_{i,t+h}
$$

The sequence $\{\hat\beta_h\}$ traces the **impulse response function** — how a
one-unit return shock propagates through fund flows over 40 trading days.
""")

lp_f = RESULTS / "figure_1_lp.csv"
if lp_f.exists():
    lp = pd.read_csv(lp_f)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["ci_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["ci_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.2)", name="95% CI",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lp["horizon"], y=lp["beta"], mode="lines+markers",
                             line=dict(color="#1f77b4", width=2),
                             marker=dict(size=4), name="Point estimate",
                             hovertemplate="h=%{x}<br>β=%{y:.2f}<extra></extra>"))
    sig = lp[lp["p_value"] < 0.05]
    if not sig.empty:
        fig.add_trace(go.Scatter(x=sig["horizon"], y=sig["beta"], mode="markers",
                                 marker=dict(color="#d62728", size=6), name="p < 0.05",
                                 hovertemplate="h=%{x}<br>β=%{y:.2f} (sig.)<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title="Horizon (trading days)",
                      yaxis_title="Response of Fund Flow ($M)",
                      title="Impulse Response: Return Shock → Fund Flow",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 2. Asymmetric Response
# ============================================================
st.header("2. Chasing vs. Fleeing: Do Gains and Losses Have Different Effects?")
st.markdown(r"""
Sirri & Tufano (1998) showed that in the **cross-section**, top-ranked funds
attract disproportionate inflows while bottom-ranked funds do not lose
proportionate capital (Table 2 above replicates this). Here we ask a
**different but related question** using our daily data: does a positive
return shock propagate differently through fund flows than a negative one?

This is our **extension beyond S&T** — their annual data could not track
dynamic responses. We decompose the return into positive and negative
components and estimate separate impulse responses for each:

$$
\text{Flow}_{i,t+h} = \alpha_i^{(h)}
+ \beta_h^{+} \cdot \max(R_{i,t}, 0)
+ \beta_h^{-} \cdot \min(R_{i,t}, 0)
+ \varepsilon_{i,t+h}
$$

- **Blue line**: flow response over 40 days following a **+1% return** (gain chasing)
- **Red line**: flow response over 40 days following a **-1% return** (loss fleeing)

If the two lines have different shapes or magnitudes, investors react
**asymmetrically** to gains vs. losses over time.
""")

asym_f = RESULTS / "figure_2_asymmetric_lp.csv"
if asym_f.exists():
    asym = pd.read_csv(asym_f)

    # beta_neg is the coefficient on min(Return, 0) which is NEGATIVE.
    # To show the marginal effect of a 1% LOSS, we negate beta_neg and its CIs.
    # This way: positive red line = outflows after losses, which is intuitive.
    asym["beta_neg_plot"] = -asym["beta_neg"]
    asym["ci_lower_neg_plot"] = -asym["ci_upper_neg"]  # flip bounds when negating
    asym["ci_upper_neg_plot"] = -asym["ci_lower_neg"]

    fig = go.Figure()
    # Positive shock (chasing): effect of +1% gain
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_upper_pos"],
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_lower_pos"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["beta_pos"],
                             mode="lines+markers", line=dict(color="#1f77b4", width=2),
                             marker=dict(size=4), name="+1% gain → flow response",
                             hovertemplate="h=%{x}<br>Flow response: %{y:.1f}M<extra></extra>"))
    # Negative shock (fleeing): marginal effect of -1% loss (negated for intuitive display)
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_upper_neg_plot"],
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["ci_lower_neg_plot"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(214,39,40,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=asym["horizon"], y=asym["beta_neg_plot"],
                             mode="lines+markers", line=dict(color="#d62728", width=2),
                             marker=dict(size=4), name="-1% loss → flow response",
                             hovertemplate="h=%{x}<br>Flow response: %{y:.1f}M<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title="Horizon (trading days)",
                      yaxis_title="Response of Fund Flow ($M)",
                      title="Asymmetric Impulse Response",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 3. Bull vs Bear
# ============================================================
st.header("3. Market Regime: Bull vs. Bear")
st.markdown("""
Does performance chasing intensify or weaken during market stress? We split the
sample into **bull** (2020-2021, pandemic rally) and **bear** (2022-2024, rate hikes
and tech selloff) regimes and re-estimate the impulse response.
""")

bull_f, bear_f = RESULTS / "table_4_lp_bull.csv", RESULTS / "table_4_lp_bear.csv"
if bull_f.exists() and bear_f.exists():
    bull, bear = pd.read_csv(bull_f), pd.read_csv(bear_f)
    fig = go.Figure()
    for data, name, color in [(bull, "Bull (2020-2021)", "#1f77b4"),
                               (bear, "Bear (2022-2024)", "#d62728")]:
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["ci_upper"],
                                 mode="lines", line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["ci_lower"],
                                 mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=data["horizon"], y=data["beta"],
                                 mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=4), name=name,
                                 hovertemplate="h=%{x}<br>β=%{y:.2f}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title="Horizon (trading days)",
                      yaxis_title="Response of Fund Flow ($M)",
                      title="Performance Chasing by Market Regime",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

sub_f = RESULTS / "table_4_subsample.csv"
if sub_f.exists():
    st.dataframe(pd.read_csv(sub_f), width="stretch", hide_index=True)

# ============================================================
# 4. Drawdown Event Study
# ============================================================
st.header("4. Extreme Events: What Happens After Crashes?")
st.markdown(r"""
The S&T asymmetry prediction has a sharp implication for extreme events: if
investors don't flee poor performers, do they **stay** even after large price
crashes? We identify **drawdown episodes** ($\geq 10\%$ peak-to-trough) and
measure cumulative fund flows in the months that follow:

$$
\text{CumFlow}_{i,[t, t+h]} = \alpha + \beta_1 \cdot \text{DrawdownDepth}_i
+ \beta_2 \cdot \text{Duration}_i + \varepsilon
$$

- $\beta_1 < 0$: deeper drawdowns → larger outflows (panic selling)
- $\beta_1 \approx 0$: investors don't respond to drawdown severity
- $\beta_1 > 0$: contrarian buying after steep declines
""")

# Load data for drawdown computation
from data_loader import get_prepared_data_with_peers, ETF_NAMES as ETF_LIST
from analysis import compute_etf_drawdowns, drawdown_flow_analysis, drawdown_flow_regression

with st.sidebar:
    dd_etf = st.selectbox("Drawdown ETF", ETF_LIST, index=0, key="dd_etf")


@st.cache_data(show_spinner="Computing drawdowns...")
def get_drawdown_data():
    df = get_prepared_data_with_peers(freq="ME", zscore_type="full", benchmark="SPY")
    df = df[df["Date"] >= pd.Timestamp("2014-10-31")]
    dd = compute_etf_drawdowns(df, "Return_Cum", min_depth_pct=10.0)
    dd_flow = drawdown_flow_analysis(df, dd, "Flow_Sum") if len(dd) > 0 else pd.DataFrame()
    return df, dd, dd_flow


df_dd, dd_all, dd_flow = get_drawdown_data()

if len(dd_all) > 0:
    # Price index with drawdown shading
    etf_dd = dd_all[dd_all["ETF"] == dd_etf]
    etf_prices = df_dd[df_dd["ETF"] == dd_etf].sort_values("Date")
    if len(etf_prices) > 10:
        price_idx = (1 + etf_prices.set_index("Date")["Return_Cum"].dropna()).cumprod() * 100
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
                             title=f"{dd_etf}: Drawdown Episodes (≥10%)")
        st.plotly_chart(fig_dd, width="stretch")

    # Scatter plots
    if len(dd_flow) > 0:
        col1, col2 = st.columns(2)
        for col, horizon, label in [(col1, "CumFlow_1m", "1-Month"),
                                     (col2, "CumFlow_3m", "3-Month")]:
            if horizon in dd_flow.columns:
                with col:
                    fig_s = px.scatter(dd_flow, x="depth_pct", y=horizon,
                                       color="ETF", title=f"Depth vs {label} Flow",
                                       hover_data=["trough_date"])
                    fig_s.update_layout(height=350, showlegend=False,
                                        xaxis_title="Drawdown Depth (%)",
                                        yaxis_title=f"Cumulative Flow {label} ($M)")
                    st.plotly_chart(fig_s, width="stretch")

        # Regression table
        dd_reg = drawdown_flow_regression(dd_flow)
        if len(dd_reg) > 0:
            st.subheader("Post-Drawdown Flow Regression")
            st.dataframe(dd_reg.style.format({
                "β_Depth": "{:.4f}", "β_Depth_p": "{:.4f}",
                "β_Duration": "{:.4f}", "β_Duration_p": "{:.4f}", "R²": "{:.4f}",
            }), hide_index=True, width="stretch")

st.info("**Next →** *Robustness*: Are these findings reliable? We test with 7 different validation methods.")
