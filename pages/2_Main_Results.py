"""Main Results: S&T regression, panel specifications, and impulse response figures."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"
FIGURES = RESULTS / "figures"

st.set_page_config(page_title="Main Results", layout="wide")
st.title("Main Results")


# ============================================================
# Table 2: Sirri-Tufano Piecewise Linear
# ============================================================
st.header("1. Sirri-Tufano Piecewise Linear Regression")

st.markdown(r"""
Following Sirri & Tufano (1998), we estimate a **piecewise linear model** where
fund flows are regressed on fractional performance rank broken into three segments:

$$
\text{Flow}_{i,t} = \alpha_i + \beta_{\text{LOW}} \cdot \text{LOWPERF}_{i,t-1}
+ \beta_{\text{MID}} \cdot \text{MIDPERF}_{i,t-1}
+ \beta_{\text{HIGH}} \cdot \text{HIGHPERF}_{i,t-1}
+ \gamma' Z_{i,t} + \varepsilon_{i,t}
$$

where LOWPERF = bottom 20%, MIDPERF = middle 60%, HIGHPERF = top 20% of the
cross-sectional performance distribution each month.

**Key prediction**: $\beta_{\text{HIGH}} \gg \beta_{\text{MID}}$ (convexity).
""")

t2_f = RESULTS / "table_2_sirri_tufano.csv"
if t2_f.exists():
    t2 = pd.read_csv(t2_f)
    st.dataframe(t2, use_container_width=True, hide_index=True)

    st.markdown("""
    **Interpretation**: HIGHPERF coefficient (18.2***) is 10x larger than MIDPERF (1.76),
    confirming the **convex flow-performance relationship** predicted by Sirri & Tufano.
    Investors strongly chase top-quintile performers but are largely indifferent to
    mid-range performance. Adding peer aggregate flow control (column 4) increases
    R-squared within from 2.4% to 9.0%.
    """)


# ============================================================
# Table 3: Main Panel Specification
# ============================================================
st.header("2. Main Panel Specification (Daily)")

st.markdown(r"""
Our primary specification uses **non-overlapping cumulative return windows** with
progressively added controls. Entity-demeaned OLS with standard errors clustered
by ETF:

$$
\text{Flow}_{i,t} = \alpha_i
+ \beta_1 \cdot \underbrace{\sum_{k=1}^{5} R_{i,t-k}}_{\text{CumRet}_{1\text{-}5}}
+ \beta_2 \cdot \underbrace{\sum_{k=6}^{20} R_{i,t-k}}_{\text{CumRet}_{6\text{-}20}}
+ \beta_3 \cdot \underbrace{\sum_{k=21}^{60} R_{i,t-k}}_{\text{CumRet}_{21\text{-}60}}
+ \gamma' Z_{i,t} + \varepsilon_{i,t}
$$

Specifications add controls incrementally: (1) Base, (2) +VIX, (3) +Calendar,
(4) +Peer Flow, (5) +Event dummies.
""")

t3_f = RESULTS / "table_3_main_panel.csv"
if t3_f.exists():
    t3 = pd.read_csv(t3_f)

    # Pivot for display
    specs = t3["spec"].unique()
    vars_display = t3[~t3["Variable"].isin(["R²", "N"])]
    stats_display = t3[t3["Variable"].isin(["R²", "N"])]

    # Show coefficient table
    pivot = vars_display.pivot(index="Variable", columns="spec", values="Coefficient")
    pvals = vars_display.pivot(index="Variable", columns="spec", values="p_value")

    # Format with significance stars
    def fmt_cell(coef, p):
        if pd.isna(coef):
            return ""
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{coef:.2f}{stars}"

    display_df = pivot.copy()
    for col in display_df.columns:
        for idx in display_df.index:
            c = pivot.loc[idx, col] if idx in pivot.index else np.nan
            p = pvals.loc[idx, col] if idx in pvals.index else np.nan
            display_df.loc[idx, col] = fmt_cell(c, p)

    # Add stats rows
    for stat_var in ["R²", "N"]:
        stat_row = stats_display[stats_display["Variable"] == stat_var]
        for spec in specs:
            row = stat_row[stat_row["spec"] == spec]
            if not row.empty:
                val = row.iloc[0]["Coefficient"]
                display_df.loc[stat_var, spec] = f"{val:.4f}" if stat_var == "R²" else f"{int(val):,}"

    # Reorder rows
    order = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60",
             "VIX_Close", "month_end", "quarter_end", "january",
             "Peer_Agg_Flow", "event_covid", "event_ukraine",
             "event_fed_hikes", "event_banking_crisis", "event_arkb_approval",
             "R²", "N"]
    order = [v for v in order if v in display_df.index]
    display_df = display_df.loc[order]

    st.dataframe(display_df, use_container_width=True)

    st.markdown("""
    **Key finding**: CumRet_6_20 (6-20 day cumulative return) and CumRet_21_60
    (21-60 day) are consistently significant across all specifications (p < 0.01).
    The 1-5 day window is not significant, suggesting investors react to *sustained*
    trends rather than very short-term movements.
    """)


# ============================================================
# Figure 1: Impulse Response
# ============================================================
st.header("3. Local Projection Impulse Response")

st.markdown(r"""
Following **Jorda (2005)**, we estimate the impulse response of fund flows to a
return shock at each horizon $h = 0, 1, \ldots, 40$ trading days:

$$
\text{Flow}_{i,t+h} = \alpha_i^{(h)} + \beta_h \cdot \text{Return}_{i,t}
+ \gamma_h' X_{i,t} + \varepsilon_{i,t+h}
$$

The sequence $\{\hat\beta_h\}$ traces the dynamic propagation of performance
chasing over time. Shaded bands show 95% confidence intervals.
""")

lp_f = RESULTS / "figure_1_lp.csv"
if lp_f.exists():
    lp = pd.read_csv(lp_f)

    fig_lp = go.Figure()
    fig_lp.add_trace(go.Scatter(
        x=lp["horizon"], y=lp["ci_upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_lp.add_trace(go.Scatter(
        x=lp["horizon"], y=lp["ci_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.2)", name="95% CI",
    ))
    fig_lp.add_trace(go.Scatter(
        x=lp["horizon"], y=lp["beta"],
        mode="lines+markers", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4), name="Point estimate",
    ))

    # Mark significant horizons
    sig = lp[lp["p_value"] < 0.05]
    if not sig.empty:
        fig_lp.add_trace(go.Scatter(
            x=sig["horizon"], y=sig["beta"],
            mode="markers", marker=dict(color="#d62728", size=6),
            name="p < 0.05",
        ))

    fig_lp.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_lp.update_layout(
        height=450,
        xaxis_title="Horizon (trading days)",
        yaxis_title="Response of Fund Flow ($M)",
        title="Impulse Response: Return Shock -> Fund Flow",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_lp, use_container_width=True)


# ============================================================
# Figure 2: Asymmetric Response
# ============================================================
st.header("4. Asymmetric Response: Chasing vs. Fleeing")

st.markdown(r"""
We decompose the return shock into positive (gain) and negative (loss) components
to test whether investors respond asymmetrically:

$$
\text{Flow}_{i,t+h} = \alpha_i^{(h)}
+ \beta_h^{+} \cdot \max(R_{i,t}, 0)
+ \beta_h^{-} \cdot \min(R_{i,t}, 0)
+ \varepsilon_{i,t+h}
$$

**S&T prediction**: $|\beta_h^{+}| > |\beta_h^{-}|$ — investors chase gains more
aggressively than they flee losses.
""")

asym_f = RESULTS / "figure_2_asymmetric_lp.csv"
if asym_f.exists():
    asym = pd.read_csv(asym_f)

    fig_asym = go.Figure()

    # Positive shock
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["ci_upper_pos"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["ci_lower_pos"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.15)", showlegend=False,
    ))
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["beta_pos"],
        mode="lines+markers", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4), name="Positive shock (chasing)",
    ))

    # Negative shock
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["ci_upper_neg"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["ci_lower_neg"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(214,39,40,0.15)", showlegend=False,
    ))
    fig_asym.add_trace(go.Scatter(
        x=asym["horizon"], y=asym["beta_neg"],
        mode="lines+markers", line=dict(color="#d62728", width=2),
        marker=dict(size=4), name="Negative shock (fleeing)",
    ))

    fig_asym.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_asym.update_layout(
        height=450,
        xaxis_title="Horizon (trading days)",
        yaxis_title="Response of Fund Flow ($M)",
        title="Asymmetric Impulse Response: Chasing vs. Fleeing",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_asym, use_container_width=True)


# ============================================================
# Figure 3: Bull vs Bear
# ============================================================
st.header("5. Bull vs. Bear Market Comparison")

st.markdown("""
We split the sample into **bull** (2020-2021) and **bear** (2022-2024) regimes
and re-estimate the LP impulse response for each. This tests whether performance
chasing behavior intensifies or weakens during market stress.
""")

bull_f = RESULTS / "table_4_lp_bull.csv"
bear_f = RESULTS / "table_4_lp_bear.csv"

if bull_f.exists() and bear_f.exists():
    bull = pd.read_csv(bull_f)
    bear = pd.read_csv(bear_f)

    fig_bb = go.Figure()

    # Bull
    fig_bb.add_trace(go.Scatter(
        x=bull["horizon"], y=bull["ci_upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_bb.add_trace(go.Scatter(
        x=bull["horizon"], y=bull["ci_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.15)", showlegend=False,
    ))
    fig_bb.add_trace(go.Scatter(
        x=bull["horizon"], y=bull["beta"],
        mode="lines+markers", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4), name="Bull (2020-2021)",
    ))

    # Bear
    fig_bb.add_trace(go.Scatter(
        x=bear["horizon"], y=bear["ci_upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_bb.add_trace(go.Scatter(
        x=bear["horizon"], y=bear["ci_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(214,39,40,0.15)", showlegend=False,
    ))
    fig_bb.add_trace(go.Scatter(
        x=bear["horizon"], y=bear["beta"],
        mode="lines+markers", line=dict(color="#d62728", width=2),
        marker=dict(size=4), name="Bear (2022-2024)",
    ))

    fig_bb.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_bb.update_layout(
        height=450,
        xaxis_title="Horizon (trading days)",
        yaxis_title="Response of Fund Flow ($M)",
        title="Performance Chasing: Bull vs. Bear Market",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bb, use_container_width=True)


# ============================================================
# Economic Significance
# ============================================================
st.header("6. Economic Significance")

econ_f = RESULTS / "economic_significance.csv"
if econ_f.exists():
    econ = pd.read_csv(econ_f)
    if not econ.empty:
        row = econ.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("CumRet 1-5d: 1-SD effect", f"${row.get('CumRet_1_5_1sd_effect', 0):.2f}M")
        col2.metric("CumRet 6-20d: 1-SD effect", f"${row.get('CumRet_6_20_1sd_effect', 0):.2f}M")
        col3.metric("CumRet 21-60d: 1-SD effect", f"${row.get('CumRet_21_60_1sd_effect', 0):.2f}M")

        st.markdown(f"""
        A one-standard-deviation increase in cumulative returns over the 6-20 day window
        is associated with **${row.get('CumRet_6_20_1sd_effect', 0):.1f}M** in additional daily
        fund flows. With mean AUM of **${row.get('mean_aum_millions', 0):,.0f}M**, this
        represents a meaningful economic effect.
        """)
