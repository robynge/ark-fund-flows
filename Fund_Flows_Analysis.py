"""Do ETF Investors Chase Past Performance?

Home page: research question, S&T framework, our extensions, key findings.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "experiments" / "results_v2"

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")

st.markdown("""
Sirri & Tufano (1998) established that mutual fund investors **chase top performers**
while failing to flee poor performers — a pattern driven by costly search.
We ask: **does this extend to ETFs?** With daily data and modern econometric
methods, we not only replicate the S&T finding but reveal the **dynamics** of
how performance chasing unfolds over time.
""")

# ============================================================
# Theoretical Framework
# ============================================================
st.header("Theoretical Framework: Sirri & Tufano (1998)")

st.markdown(r"""
**Core hypothesis**: Investors use past performance as a low-cost search signal.
Because gathering information is costly, top performers attract disproportionate
inflows, while poor performers do *not* experience proportionate outflows.

The S&T piecewise linear specification:

$$
\text{Flow}_{i,t} = \alpha + \beta_{\text{LOW}} \cdot \text{LOWPERF}_{i,t-1}
+ \beta_{\text{MID}} \cdot \text{MIDPERF}_{i,t-1}
+ \beta_{\text{HIGH}} \cdot \text{HIGHPERF}_{i,t-1}
+ \gamma' X_{i,t-1} + \varepsilon_{i,t}
$$

**Key prediction**: $\beta_{\text{HIGH}} \gg \beta_{\text{MID}} \approx \beta_{\text{LOW}}$
— a **convex** flow-performance relationship.
""")

# ============================================================
# Our Extension
# ============================================================
st.header("What We Add")

st.markdown(r"""
S&T used **annual** mutual fund data — they showed *that* investors chase, but
couldn't show *how* the effect unfolds over time. Our **daily** ETF data enables
three extensions:

| Extension | Method | What It Reveals |
|-----------|--------|-----------------|
| **Cumulative return windows** | Panel FE with non-overlapping 1-5, 6-20, 21-60 day windows | Which time horizon drives chasing? |
| **Local Projection impulse response** | Jordà (2005), horizon h = 0..40 days | How does the flow response build and decay? |
| **Asymmetric & regime analysis** | Positive/negative shock decomposition; bull/bear split | Do investors chase gains more than they flee losses? Does it change in crises? |

Our main panel specification:

$$
\text{Flow}_{i,t} = \alpha_i
+ \beta_1 \cdot \text{CumRet}_{i,[t-5,t-1]}
+ \beta_2 \cdot \text{CumRet}_{i,[t-20,t-6]}
+ \beta_3 \cdot \text{CumRet}_{i,[t-60,t-21]}
+ \gamma' Z_{i,t} + \varepsilon_{i,t}
$$

where $\alpha_i$ = ETF fixed effects, $Z$ = VIX, calendar dummies, peer flow, event dummies.
""")

# ============================================================
# Key Findings Preview
# ============================================================
st.header("Key Findings")

t3_f = RESULTS / "table_3_main_panel.csv"
econ_f = RESULTS / "economic_significance.csv"

col1, col2, col3 = st.columns(3)

if t3_f.exists():
    t3 = pd.read_csv(t3_f)
    r = t3[(t3["spec"] == "(1) Base") & (t3["Variable"] == "CumRet_6_20")]
    if not r.empty:
        col1.metric("6-20 day return → flow", f"β = {r.iloc[0]['Coefficient']:.1f}***",
                     f"p = {r.iloc[0]['p_value']:.4f}")
    r2 = t3[(t3["spec"] == "(1) Base") & (t3["Variable"] == "CumRet_21_60")]
    if not r2.empty:
        col2.metric("21-60 day return → flow", f"β = {r2.iloc[0]['Coefficient']:.1f}***",
                     f"p = {r2.iloc[0]['p_value']:.4f}")

if econ_f.exists():
    econ = pd.read_csv(econ_f)
    if not econ.empty:
        col3.metric("1-SD shock (6-20d)", f"${econ.iloc[0].get('CumRet_6_20_1sd_effect', 0):.1f}M flow")

st.markdown("""
**Summary**: ETF investors chase past performance. The effect is concentrated in
the **6-60 day** return horizon, robust to Driscoll-Kraay SE, placebo tests,
and leave-one-ETF-out analysis. The 1-5 day window is *not* significant —
investors react to sustained trends, not daily noise.
""")

# ============================================================
# Dashboard Navigation
# ============================================================
st.divider()
st.markdown("""
### Dashboard Guide

| Page | Question | Content |
|------|----------|---------|
| **The Data** | *What does the data look like?* | Price + fund flow chart, cumulative flows, summary statistics, ARK vs peers |
| **The Evidence** | *Do investors chase?* | S&T scatter, piecewise + quadratic regression, panel specification, per-ETF results |
| **The Dynamics** | *How do they chase?* | Impulse response, asymmetric response, bull/bear regimes, drawdown events |
| **Robustness** | *Should we believe this?* | SE comparison, Granger causality, ARKK heterogeneity, rolling coefficients |
| **Explorer** | *Let me see for myself* | Per-ETF interactive analysis: distributions, lag structure, correlations |
| **Marketing Premium** | *Is marketing driving flows?* | Predicted vs actual flows per ETF, residual analysis |
""")
