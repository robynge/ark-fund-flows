"""ETF Performance Chasing — Research Overview

Home page: research question, methodology, theoretical framework, and key formulas.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

RESULTS_V2 = Path(__file__).parent / "experiments" / "results_v2"
GRID_DIR = RESULTS_V2 / "grid_search"

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")

st.markdown("""
This study examines whether ETF investors exhibit **performance chasing behavior**
--- the tendency to allocate capital toward funds with strong recent returns.
We follow the theoretical framework of **Sirri & Tufano (1998)** and extend it
to the ETF market with high-frequency data and modern econometric methods.
""")

# ============================================================
# Theoretical Framework
# ============================================================
st.header("Theoretical Framework")

st.markdown(r"""
### Sirri & Tufano (1998): Costly Search and Mutual Fund Flows

The seminal Sirri & Tufano (1998) paper establishes that **investors disproportionately
chase top-performing funds**, while poorly performing funds do not experience
proportionate outflows. This asymmetry arises from costly search: investors use
past performance as a low-cost signal to identify funds.

Their core empirical model estimates the **piecewise linear relationship** between
performance rank and fund flows:

$$
\text{Flow}_{i,t} = \alpha + \beta_{\text{LOW}} \cdot \text{LOWPERF}_{i,t-1}
+ \beta_{\text{MID}} \cdot \text{MIDPERF}_{i,t-1}
+ \beta_{\text{HIGH}} \cdot \text{HIGHPERF}_{i,t-1}
+ \gamma \cdot X_{i,t-1} + \varepsilon_{i,t}
$$

where:
- $\text{LOWPERF}$, $\text{MIDPERF}$, $\text{HIGHPERF}$ = fractional performance
  rank in bottom 20%, middle 60%, top 20%
- $X_{i,t-1}$ = controls (fund size, category flows, risk, fees)
- **Key prediction**: $\beta_{\text{HIGH}} \gg \beta_{\text{MID}} \approx \beta_{\text{LOW}}$
  (convex flow-performance relationship)
""")

# ============================================================
# Our Approach
# ============================================================
st.header("Our Approach")

st.markdown(r"""
We adapt the Sirri & Tufano framework to ETFs with three key extensions:

### 1. Main Panel Specification (Daily)

Our primary specification uses **non-overlapping cumulative return windows** as regressors,
estimated with entity fixed effects and clustered standard errors:

$$
\text{Flow}_{i,t} = \alpha_i + \beta_1 \cdot \text{CumRet}_{i,[t-5,t-1]}
+ \beta_2 \cdot \text{CumRet}_{i,[t-20,t-6]}
+ \beta_3 \cdot \text{CumRet}_{i,[t-60,t-21]}
+ \gamma' Z_{i,t} + \varepsilon_{i,t}
$$

where:
- $\alpha_i$ = ETF fixed effect (captures fund-specific average flow)
- $\text{CumRet}_{[a,b]}$ = cumulative return from day $t-b$ to $t-a$
- $Z_{i,t}$ = controls: VIX, calendar dummies, peer aggregate flow, event dummies
- Standard errors clustered by ETF

### 2. Local Projection Impulse Response (Jorda, 2005)

To trace the **dynamic response** of flows to a return shock:

$$
\text{Flow}_{i,t+h} = \alpha_i^h + \beta_h \cdot \text{Shock}_{i,t}
+ \gamma_h' X_{i,t} + \varepsilon_{i,t+h}
\quad \text{for } h = 0, 1, \ldots, 40
$$

The sequence $\{\hat\beta_h\}$ traces the impulse response function, showing
how fund flows evolve over 40 trading days following a return shock.

### 3. Systematic Noise Decomposition

We hypothesize that confounders obscure the true signal. Five noise factors
are toggled on/off in all $2^5 - 1 = 31$ combinations:

| Factor | What it Controls | Method |
|--------|-----------------|--------|
| **A** | Macro events (COVID, Ukraine, Fed hikes) | Exclude event windows |
| **B** | Market-wide flow trends | Subtract cross-sectional mean flow |
| **C** | Volatility regime | Add VIX as control variable |
| **D** | Calendar seasonality | Add month-end, quarter-end, January dummies |
| **E** | Peer aggregate flow | Add leave-one-out peer flow mean |
""")

# ============================================================
# Data
# ============================================================
st.header("Data")

st.markdown("""
| Item | Source | Coverage |
|------|--------|----------|
| 9 ARK ETFs (daily flows + OHLCV) | Bloomberg | 2014-10 to 2026-03 |
| 29 Tech peer ETFs (daily flows + OHLCV) | Bloomberg | 2017+ (varies) |
| Monthly AUM | Bloomberg | All ETFs |
| SPY, QQQ benchmarks | Yahoo Finance | Full sample |
| VIX index | CBOE via Yahoo | Full sample |

**Fund Flow Definition** (Sirri & Tufano, 1998):

$$
\\text{Flow}_{i,t} = \\frac{\\text{TNA}_{i,t} - \\text{TNA}_{i,t-1} \\times (1 + R_{i,t})}{\\text{TNA}_{i,t-1}}
$$

For ETFs, we use Bloomberg's net creation/redemption flow (in millions USD)
as a direct measure of investor-driven capital allocation.
""")

# ============================================================
# Key Results Summary
# ============================================================
st.header("Key Results")

col1, col2, col3 = st.columns(3)

# Load key results
t3_f = RESULTS_V2 / "table_3_main_panel.csv"
econ_f = RESULTS_V2 / "economic_significance.csv"

if t3_f.exists():
    t3 = pd.read_csv(t3_f)
    base_cumret620 = t3[(t3["spec"] == "(1) Base") & (t3["Variable"] == "CumRet_6_20")]
    if not base_cumret620.empty:
        beta = base_cumret620.iloc[0]["Coefficient"]
        p = base_cumret620.iloc[0]["p_value"]
        col1.metric(
            "CumRet_6_20 coefficient",
            f"${beta:.1f}M",
            f"p = {p:.4f}",
        )

    base_cumret2160 = t3[(t3["spec"] == "(1) Base") & (t3["Variable"] == "CumRet_21_60")]
    if not base_cumret2160.empty:
        beta2 = base_cumret2160.iloc[0]["Coefficient"]
        p2 = base_cumret2160.iloc[0]["p_value"]
        col2.metric(
            "CumRet_21_60 coefficient",
            f"${beta2:.1f}M",
            f"p = {p2:.4f}",
        )

if econ_f.exists():
    econ = pd.read_csv(econ_f)
    if not econ.empty:
        sd_effect = econ.iloc[0].get("CumRet_6_20_1sd_effect", None)
        if sd_effect:
            col3.metric(
                "1-SD Return Shock (6-20d)",
                f"${sd_effect:.1f}M flow response",
            )

st.markdown("""
**Summary**: Past returns significantly predict future fund flows. A 1-standard-deviation
increase in 6-20 day cumulative returns is associated with approximately $2.6M in
additional daily fund flows. The effect is robust to Driscoll-Kraay standard errors,
alternative dependent variables (% of AUM), and leave-one-ETF-out tests. Placebo
tests using future (lead) returns show no significant relationship, confirming
the direction of causality.
""")

# ============================================================
# Navigation
# ============================================================
st.divider()
st.subheader("Dashboard Pages")
st.markdown("""
| Page | Content |
|------|---------|
| **Data Overview** | Price & fund flow time series, summary statistics, S&T scatter plot |
| **Main Results** | Sirri-Tufano regression, panel specifications, impulse response figures |
| **Robustness** | Placebo tests, leave-one-out, Fama-MacBeth, Driscoll-Kraay SE, diagnostics |
| **Interactive Analysis** | Deep-dive: per-ETF flow distributions, cross-correlations, asymmetry, seasonality |
| **Drawdown Analysis** | Event study: do large price drops trigger capital flight? |
""")
