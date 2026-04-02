"""Page 2: The Evidence — S&T scatter, piecewise regression, panel specification."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="The Evidence", layout="wide")
st.title("The Evidence: Do Investors Chase?")
st.markdown("""
We present evidence at three levels: **(1)** visual evidence from the S&T scatter,
**(2)** the classic S&T piecewise regression, and **(3)** our daily panel specification.
All three point the same way: **yes, ETF investors chase past performance.**
""")


def _stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ============================================================
# 1. S&T Scatter (moved from old Data Overview)
# ============================================================
st.header("1. Do Top Performers Attract More Money?")
st.markdown(r"""
Each month, we rank all ETFs by return (0 = worst, 1 = best) and ask:
**do higher-ranked ETFs actually attract more capital?**

- **X-axis**: the ETF's return rank among all ETFs that month (0 to 1)
- **Y-axis**: the ETF's net fund flow that month (as % of AUM)
- **Red line**: average flow within each of 20 equal-width rank bins

If the red line **curves upward** (steep rise at the right end), it means
top performers attract **disproportionately more capital** — the hallmark of
performance chasing documented by Sirri & Tufano (1998).

$$
\text{Rank}_{i,t} = \frac{\text{position of } R_{i,t} \text{ among all ETFs at time } t}{N_t}
\in [0, 1]
$$
""")

scatter_f = RESULTS / "figure_st1_scatter.csv"
if scatter_f.exists():
    scatter = pd.read_csv(scatter_f)
    scatter["rank_bin"] = pd.cut(scatter["RANK"], bins=20, labels=False) + 1
    bin_means = scatter.groupby("rank_bin").agg(
        rank_mid=("RANK", "mean"), flow_mean=("Flow_Pct", "mean")).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=scatter["RANK"], y=scatter["Flow_Pct"],
        mode="markers", marker=dict(size=3, opacity=0.08, color="#1f77b4"),
        name="Individual obs.", showlegend=True))
    fig.add_trace(go.Scatter(
        x=bin_means["rank_mid"], y=bin_means["flow_mean"],
        mode="lines+markers", line=dict(color="#d62728", width=3),
        marker=dict(size=8), name="20-bin average"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=450,
        xaxis_title="Performance Rank (0 = worst, 1 = best)",
        yaxis_title="Fund Flow (% of AUM)",
        title="Sirri-Tufano Replication: Performance Rank vs. Flow Growth (Monthly)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 2. S&T Piecewise Linear Regression (Table 2)
# ============================================================
st.header("2. S&T Piecewise Linear Regression")
st.markdown(r"""
We formalize the scatter pattern with the **Sirri & Tufano piecewise linear model**,
decomposing performance rank into three segments:

$$
\text{Flow}_{i,t} = \alpha_i
+ \beta_L \cdot \text{LOWPERF}_{i,t-1}
+ \beta_M \cdot \text{MIDPERF}_{i,t-1}
+ \beta_H \cdot \text{HIGHPERF}_{i,t-1}
+ \gamma' Z + \varepsilon_{i,t}
$$

- **LOWPERF** = min(Rank, 0.2) — bottom 20%
- **MIDPERF** = min(Rank − LOWPERF, 0.6) — middle 60%
- **HIGHPERF** = Rank − LOWPERF − MIDPERF — top 20%

Columns add controls incrementally: (1) Base, (2) +VIX, (3) +Calendar, (4) +Peer Flow.
""")

t2_f = RESULTS / "table_2_sirri_tufano.csv"
if t2_f.exists():
    st.dataframe(pd.read_csv(t2_f), width="stretch", hide_index=True)
    st.success(r"""
    **Result**: HIGHPERF = 18.2*** is **10× larger** than MIDPERF = 1.76 (ns).
    This confirms the **convex flow-performance relationship** in ETFs — the S&T
    prediction holds. Adding peer flow control (col 4) raises R² from 2.4% to 9.0%.
    """)

# ============================================================
# 3. Daily Panel Specification (Table 3)
# ============================================================
st.header("3. Our Panel Specification (Daily)")
st.markdown(r"""
S&T used annual data with one "last year return" variable. Our **daily data** lets
us decompose returns into **non-overlapping cumulative windows** to identify which
time horizon drives performance chasing:

$$
\text{Flow}_{i,t} = \alpha_i
+ \beta_1 \cdot \underbrace{\sum_{k=1}^{5} R_{i,t-k}}_{\text{last week}}
+ \beta_2 \cdot \underbrace{\sum_{k=6}^{20} R_{i,t-k}}_{\text{last month}}
+ \beta_3 \cdot \underbrace{\sum_{k=21}^{60} R_{i,t-k}}_{\text{last quarter}}
+ \gamma' Z_{i,t} + \varepsilon_{i,t}
$$

Entity-demeaned OLS, standard errors clustered by ETF.
""")

t3_f = RESULTS / "table_3_main_panel.csv"
if t3_f.exists():
    t3 = pd.read_csv(t3_f)
    specs = t3["spec"].unique()
    var_rows = t3[~t3["Variable"].isin(["R²", "N"])]
    stat_rows = t3[t3["Variable"].isin(["R²", "N"])]

    pivot = var_rows.pivot(index="Variable", columns="spec", values="Coefficient")
    pvals = var_rows.pivot(index="Variable", columns="spec", values="p_value")

    display = pivot.copy()
    for col in display.columns:
        for idx in display.index:
            c = pivot.loc[idx, col] if idx in pivot.index else np.nan
            p = pvals.loc[idx, col] if idx in pvals.index else np.nan
            display.loc[idx, col] = f"{c:.2f}{_stars(p)}" if not pd.isna(c) else ""

    for sv in ["R²", "N"]:
        for spec in specs:
            row = stat_rows[(stat_rows["spec"] == spec) & (stat_rows["Variable"] == sv)]
            if not row.empty:
                val = row.iloc[0]["Coefficient"]
                display.loc[sv, spec] = f"{val:.4f}" if sv == "R²" else f"{int(val):,}"

    order = [v for v in ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60",
             "VIX_Close", "month_end", "quarter_end", "january",
             "Peer_Agg_Flow", "event_covid", "event_ukraine",
             "event_fed_hikes", "event_banking_crisis", "event_arkb_approval",
             "R²", "N"] if v in display.index]
    st.dataframe(display.loc[order], width="stretch")

    st.success("""
    **Key finding**: The 1-5 day window (last week) is **not significant** — investors
    don't react to daily noise. The 6-20 day and 21-60 day windows are **highly
    significant** (p < 0.01) across all specifications. Performance chasing operates
    on a **2-week to 3-month** horizon.
    """)

# ============================================================
# 4. Economic Significance
# ============================================================
st.header("4. How Big Is the Effect?")

econ_f = RESULTS / "economic_significance.csv"
if econ_f.exists():
    econ = pd.read_csv(econ_f)
    if not econ.empty:
        row = econ.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("CumRet 1-5d", f"${row.get('CumRet_1_5_1sd_effect', 0):.2f}M per 1-SD")
        col2.metric("CumRet 6-20d", f"${row.get('CumRet_6_20_1sd_effect', 0):.2f}M per 1-SD")
        col3.metric("CumRet 21-60d", f"${row.get('CumRet_21_60_1sd_effect', 0):.2f}M per 1-SD")

        st.markdown(f"""
        A one-standard-deviation increase in 6-20 day cumulative returns is associated
        with **${row.get('CumRet_6_20_1sd_effect', 0):.1f}M** in additional daily fund flows.
        With mean AUM of **${row.get('mean_aum_millions', 0):,.0f}M**, this represents a
        meaningful capital reallocation signal.
        """)

st.info("**Next →** *The Dynamics*: S&T showed *that* investors chase. Our daily data shows *how*.")
