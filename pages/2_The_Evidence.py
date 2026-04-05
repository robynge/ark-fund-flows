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

> **Reading the tables**: Stars next to coefficients indicate statistical significance.
> **\*\*\*** = p < 0.01 (very strong), **\*\*** = p < 0.05 (strong), **\*** = p < 0.10 (moderate),
> no stars = not statistically significant. In plain terms: more stars = more confident
> the result is real, not due to random chance.
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
Each month, we rank all ETFs by return and convert the rank to a **fractional
percentile** between 0 and 1 (following Sirri & Tufano 1998). For example,
a rank of **0.21** means this ETF's return was at the 21st percentile that month
(better than 21% of ETFs). A rank of **0.95** means it was near the top.

- **X-axis**: fractional performance rank (0.0 = worst return, 1.0 = best return)
- **Y-axis**: net fund flow that month (as % of AUM)
- **Red line**: average flow within each of 20 equal-width rank bins

If the red line **curves upward** at the right end, top performers attract
**disproportionately more capital** — the hallmark of performance chasing.

$$
\text{Rank}_{i,t} = \frac{\text{position of ETF } i \text{'s return among all ETFs at time } t}{N_t}
\in [0, 1]
$$
""")

scatter_f = RESULTS / "figure_st1_scatter.csv"
if scatter_f.exists():
    scatter = pd.read_csv(scatter_f)

    # Filter extreme outliers (ETF inception months where Flow/AUM > 100%)
    p1, p99 = scatter["Flow_Pct"].quantile(0.01), scatter["Flow_Pct"].quantile(0.99)
    scatter_clean = scatter[(scatter["Flow_Pct"] >= p1) & (scatter["Flow_Pct"] <= p99)]

    scatter_clean["rank_bin"] = pd.cut(scatter_clean["RANK"], bins=20, labels=False) + 1
    bin_means = scatter_clean.groupby("rank_bin").agg(
        rank_mid=("RANK", "mean"), flow_mean=("Flow_Pct", "mean")).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=scatter_clean["RANK"], y=scatter_clean["Flow_Pct"],
        mode="markers", marker=dict(size=4, opacity=0.35, color="#1f77b4"),
        name="Individual obs.", showlegend=True,
        hovertemplate="Rank: %{x:.2f}<br>Flow: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=bin_means["rank_mid"], y=bin_means["flow_mean"],
        mode="lines+markers", line=dict(color="#d62728", width=3),
        marker=dict(size=8), name="20-bin average",
        hovertemplate="Rank: %{x:.2f}<br>Avg flow: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=450,
        xaxis_title="Performance Rank (0 = worst, 1 = best)",
        yaxis_title="Fund Flow (% of AUM)",
        title="Sirri-Tufano Replication: Performance Rank vs. Flow Growth (Monthly)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

    # Line-only version (professor requested for presentations)
    with st.expander("Show line-only version (no individual points)"):
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=bin_means["rank_mid"], y=bin_means["flow_mean"],
            mode="lines+markers", line=dict(color="#d62728", width=3),
            marker=dict(size=8), name="20-bin average",
            hovertemplate="Rank: %{x:.2f}<br>Avg flow: %{y:.2f}%<extra></extra>"))
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_line.update_layout(
            height=400,
            xaxis_title="Performance Rank (0 = worst, 1 = best)",
            yaxis_title="Fund Flow (% of AUM)",
            title="Performance Rank vs. Fund Flow (Bin Averages Only)")
        st.plotly_chart(fig_line, width="stretch")

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
    **Result**: HIGHPERF is significantly larger than MIDPERF, confirming the **convex
    flow-performance relationship** in ETFs — the S&T prediction holds. Adding peer
    flow control raises R² substantially.
    """)

    st.warning("""
    **Note on LOWPERF**: The positive LOWPERF coefficient is puzzling — it means
    even poorly performing ETFs receive some inflows. A possible explanation is
    that ETFs with strong marketing (like ARK) continue attracting capital
    despite poor relative performance. This is consistent with the Sirri & Tufano
    "costly search" theory: marketing reduces search costs and drives flows
    independently of performance.
    """)

# Quadratic specification
quad_f = RESULTS / "table_2_quadratic.csv"
if quad_f.exists():
    with st.expander("Alternative: Quadratic Performance Specification"):
        st.markdown(r"""
        Instead of splitting performance into LOW/MID/HIGH bins, we can test
        for non-linearity using a **quadratic** specification:

        $$
        \text{Flow}_{i,t} = \alpha_i + \beta_1 \cdot \text{RANK}_{i,t}
        + \beta_2 \cdot \text{RANK}_{i,t}^2 + \varepsilon_{i,t}
        $$

        If $\beta_2 > 0$, the relationship is convex (accelerating at the top).
        """)
        st.dataframe(pd.read_csv(quad_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)

# VIF multicollinearity check
vif_f = RESULTS / "vif_table2.csv"
if vif_f.exists():
    with st.expander("Multicollinearity Check (VIF)"):
        st.markdown("""
        Variance Inflation Factors for the Table 2 regressors. VIF > 10 indicates
        problematic multicollinearity.
        """)
        st.dataframe(pd.read_csv(vif_f).style.format({"VIF": "{:.2f}"}),
                      width="stretch", hide_index=True)

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

    # Build string DataFrame directly (pandas 3.x forbids writing str into float cols)
    rows_display = {}
    for idx in pivot.index:
        row_vals = {}
        for col in pivot.columns:
            c = pivot.loc[idx, col] if idx in pivot.index else np.nan
            p = pvals.loc[idx, col] if idx in pvals.index else np.nan
            row_vals[col] = f"{c:.2f}{_stars(p)}" if not pd.isna(c) else ""
        rows_display[idx] = row_vals

    for sv in ["R²", "N"]:
        for spec in specs:
            row = stat_rows[(stat_rows["spec"] == spec) & (stat_rows["Variable"] == sv)]
            if not row.empty:
                val = row.iloc[0]["Coefficient"]
                rows_display.setdefault(sv, {})[spec] = f"{val:.4f}" if sv == "R²" else f"{int(val):,}"

    display = pd.DataFrame.from_dict(rows_display, orient="index")
    display.index.name = "Variable"

    order = [v for v in ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60",
             "VIX_Change", "VIX_Lag_Change", "VIX_Close",
             "month_end", "quarter_end", "january",
             "Peer_Agg_Flow", "event_covid", "event_ukraine",
             "event_fed_hikes", "event_banking_crisis", "event_arkb_approval",
             "R²", "N"] if v in display.index]
    st.dataframe(display.loc[order], width="stretch")

    st.success("""
    **Key finding**: The 1-5 day window (last week) is **not significant** — investors
    don't react to daily noise. The 6-20 day and 21-60 day windows are **highly
    significant** across all specifications. Performance chasing operates
    on a **2-week to 3-month** horizon.
    """)

    # Highlight peer flow result (Task 10)
    if "Peer_Agg_Flow" in display.index:
        st.info("""
        **Peer Aggregate Flow**: When money flows into the ETF's peer group,
        the ETF itself also receives inflows — even after controlling for its own
        performance. This suggests that investors allocate at the **category level**
        (e.g., "tech ETFs" or "innovation ETFs"), not just the individual fund level.
        This is a key control variable because it captures sector-wide momentum
        that is not specific to any single ETF's performance.
        """)

# Per-ETF individual regressions (Task 5)
t9_f = RESULTS / "table_9_per_etf.csv"
if t9_f.exists():
    with st.expander("Per-ETF Individual Regressions"):
        st.markdown("""
        The panel regression above pools all ETFs. Here we run the **same specification
        separately for each ETF** to see which funds exhibit the strongest performance
        chasing. HC1 robust standard errors (no clustering since N=1 per regression).
        """)
        t9 = pd.read_csv(t9_f)
        # Pivot to show one row per ETF
        for etf in t9["ETF"].unique():
            etf_df = t9[t9["ETF"] == etf]
            coefs = etf_df[~etf_df["Variable"].isin(["R²", "N"])]
            r2_row = etf_df[etf_df["Variable"] == "R²"]
            n_row = etf_df[etf_df["Variable"] == "N"]
            r2_val = r2_row.iloc[0]["Coefficient"] if not r2_row.empty else None
            n_val = int(n_row.iloc[0]["Coefficient"]) if not n_row.empty else None
            label = f"**{etf}** (R²={r2_val:.4f}, N={n_val:,})" if r2_val else f"**{etf}**"
            st.markdown(label)
            st.dataframe(coefs[["Variable", "Coefficient", "Std_Error", "t_stat", "p_value"]].style.format({
                "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
                "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
                width="stretch", hide_index=True)

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
