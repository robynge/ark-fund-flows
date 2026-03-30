"""ETF Performance Chasing — Research Overview

Home page: research question, methodology summary, and navigation guide.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "experiments" / "results_v2" / "grid_search"

st.set_page_config(page_title="ETF Performance Chasing", layout="wide")
st.title("Do ETF Investors Chase Past Performance?")

st.markdown("""
## Research Question

Do ETF investors **chase past performance** — i.e., do prior returns predict
subsequent fund flows? We study **38 tech ETFs** (9 ARK + 29 peers) using
Bloomberg daily net fund flow data, daily OHLCV prices, and monthly AUM
covering **2014-10-31 to 2026**.

## Approach

1. **Baseline analysis** — 14 statistical models (univariate R-squared by lag, cross-correlation,
   multi-lag OLS, panel regressions with various fixed effects, asymmetry tests,
   Granger causality, seasonality, drawdown event study).
   The baseline signal is **weak at the per-ETF level** but significant in panel models.

2. **Noise removal** — We hypothesize that confounders obscure the true signal.
   We systematically remove 5 noise factors in all 31 non-empty combinations:

   | Factor | What it removes |
   |--------|----------------|
   | **A** — Macro events | Periods around FOMC, CPI, NFP releases |
   | **B** — Market-wide flow | Cross-sectional mean flow on each date |
   | **C** — VIX volatility | High-VIX regimes (> 80th percentile) |
   | **D** — Calendar dummies | Month-of-year seasonal patterns |
   | **E** — Peer aggregate flow | Peer-group average flow |

3. **Gate test** — For each of the 32 experiments x 14 models = **448 tests**,
   we check: is the return-flow coefficient positive and statistically significant
   ($\\beta_1 > 0$ and $p < 0.05$)?

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| **Robustness Results** | Specification curve, forest plot, coefficient path, factor contribution |
| **Model Comparison** | Deep dive into one model across all 32 experiments |
| **Interactive Analysis** | Explore a single ETF: flow distribution, time series, correlations, panel regressions |
| **Drawdown Analysis** | Drawdown event study: post-drawdown flow behavior |
""")

st.divider()

# Quick summary stats
@st.cache_data
def load_master():
    path = RESULTS_DIR / "master_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


master = load_master()

if not master.empty:
    st.subheader("Quick Summary")

    n_total = len(master)
    n_pass = int(master["gate_pass"].sum())
    pass_rate = n_pass / n_total * 100

    # Baseline vs best noise removal
    bl = master[master["experiment_id"] == "baseline"]
    bl_pass = int(bl["gate_pass"].sum())
    bl_total = len(bl)

    noise = master[master["experiment_id"] != "baseline"]
    noise_pass = int(noise["gate_pass"].sum())
    noise_total = len(noise)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline gate pass", f"{bl_pass}/{bl_total} ({bl_pass/bl_total*100:.0f}%)")
    col2.metric("Noise-removed gate pass", f"{noise_pass}/{noise_total} ({noise_pass/noise_total*100:.1f}%)")
    col3.metric("Overall gate pass", f"{n_pass}/{n_total} ({pass_rate:.1f}%)")

    st.caption(
        "Gate pass = $\\beta_1 > 0$ and $p < 0.05$. "
        "A higher pass rate after noise removal suggests confounders were masking the signal."
    )
else:
    st.info("No experiment results found. Run `python -m experiments.runner` first.")
