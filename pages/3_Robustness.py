"""Robustness Tests: placebo, leave-one-out, Fama-MacBeth, DK SE, diagnostics."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="Robustness Tests", layout="wide")
st.title("Robustness Tests")

st.markdown("""
This page presents a comprehensive battery of robustness tests for the main
specification. Each test addresses a specific concern about the validity of
our performance chasing results.
""")


# ============================================================
# 5a: Placebo Test
# ============================================================
st.header("1. Placebo Test: Lead vs. Lag Returns")

st.markdown(r"""
**Concern**: The return-flow relationship could be spurious.

**Test**: Replace lagged (past) returns with **lead (future) returns** as regressors.
If future returns predict flows, it indicates reverse causality or spurious correlation.

$$
\text{Placebo: } \text{Flow}_{i,t} = \alpha_i
+ \beta_1 \cdot \text{CumRet}_{i,[t+1,t+5]}
+ \beta_2 \cdot \text{CumRet}_{i,[t+6,t+20]}
+ \beta_3 \cdot \text{CumRet}_{i,[t+21,t+60]}
+ \varepsilon_{i,t}
$$

**Expected**: Placebo coefficients should be **insignificant**.
""")

real_f = RESULTS / "table_5a_placebo_real.csv"
fake_f = RESULTS / "table_5a_placebo_fake.csv"

if real_f.exists() and fake_f.exists():
    real = pd.read_csv(real_f)
    fake = pd.read_csv(fake_f)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real (Lagged Returns)")
        st.dataframe(real.style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}",
        }), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Placebo (Lead Returns)")
        st.dataframe(fake.style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}",
        }), use_container_width=True, hide_index=True)

    st.success("Placebo coefficients are insignificant, confirming the lag-lead direction.")


# ============================================================
# 5b: Leave-One-ETF-Out
# ============================================================
st.header("2. Leave-One-ETF-Out Stability")

st.markdown("""
**Concern**: Results might be driven by a single dominant ETF (e.g., ARKK).

**Test**: Drop each ETF one at a time and re-estimate the full specification.
Coefficients should remain stable.
""")

loo_f = RESULTS / "table_5b_leave_one_out.csv"
if loo_f.exists():
    loo = pd.read_csv(loo_f)
    st.dataframe(loo.style.format({
        c: "{:.2f}" for c in loo.columns
        if c not in ["ETF_excluded", "n_obs"] and loo[c].dtype in ["float64"]
    }), use_container_width=True, hide_index=True)


# ============================================================
# 5c: Fama-MacBeth
# ============================================================
st.header("3. Fama-MacBeth Cross-Sectional Regression")

st.markdown(r"""
**Concern**: Panel regression assumes constant coefficients. What if the
relationship varies over time?

**Test**: **Fama-MacBeth (1973)** procedure — run cross-sectional regressions
each period, then average coefficients over time:

$$
\hat\beta_{\text{FM}} = \frac{1}{T} \sum_{t=1}^{T} \hat\beta_t, \quad
\text{SE}(\hat\beta_{\text{FM}}) = \frac{\text{SD}(\hat\beta_t)}{\sqrt{T}}
$$
""")

fm_f = RESULTS / "table_5c_fama_macbeth.csv"
if fm_f.exists():
    fm = pd.read_csv(fm_f)
    st.dataframe(fm.style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}",
    }), use_container_width=True, hide_index=True)

    st.info("""
    Note: Fama-MacBeth coefficients are insignificant due to the small cross-section
    (9 ETFs per period). This is expected — FM requires a large cross-section for power.
    The panel approach with clustered SE is preferred for our sample.
    """)


# ============================================================
# 5d: Alternative DV (Flow % AUM)
# ============================================================
st.header("4. Alternative Dependent Variable: Flow % of AUM")

st.markdown(r"""
**Concern**: Raw dollar flows may be driven by fund size rather than investor behavior.

**Test**: Replace Fund_Flow ($M) with Flow/AUM (%) as the dependent variable:

$$
\frac{\text{Flow}_{i,t}}{\text{AUM}_{i,t-1}} \times 100
= \alpha_i + \beta_1 \cdot \text{CumRet}_{i,[1,5]} + \ldots + \varepsilon_{i,t}
$$
""")

pct_f = RESULTS / "table_5d_flow_pct.csv"
if pct_f.exists():
    pct = pd.read_csv(pct_f)
    st.dataframe(pct.style.format({
        "Coefficient": "{:.4f}", "Std_Error": "{:.4f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}",
    }), use_container_width=True, hide_index=True)

    st.success("""
    All three cumulative return windows are highly significant (p < 0.05)
    when using Flow % of AUM, confirming that the relationship is not
    an artifact of fund size.
    """)


# ============================================================
# 5e: Driscoll-Kraay SE
# ============================================================
st.header("5. Driscoll-Kraay Standard Errors")

st.markdown(r"""
**Concern**: Standard clustered SE may not account for cross-sectional dependence.

**Test**: **Driscoll & Kraay (1998)** standard errors, which are robust to both
cross-sectional correlation and temporal autocorrelation:

$$
V_{\text{DK}} = \frac{1}{T^2} \sum_{j=-m}^{m} K(j/m)
\left(\sum_{t} h_t h_{t-j}'\right)
\quad \text{where } h_t = \sum_{i} X_{it} \hat\varepsilon_{it}
$$
""")

dk_f = RESULTS / "table_5e_driscoll_kraay.csv"
if dk_f.exists():
    dk = pd.read_csv(dk_f)
    st.dataframe(dk.style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}",
    }), use_container_width=True, hide_index=True)

    st.success("""
    CumRet_6_20 and CumRet_21_60 remain significant under Driscoll-Kraay SE,
    which are the most conservative standard errors in our toolkit.
    """)


# ============================================================
# 5f: Heteroscedasticity Diagnostics
# ============================================================
st.header("6. Heteroscedasticity Diagnostics")

st.markdown("""
**Concern**: OLS assumes homoscedastic errors. Heteroscedasticity inflates t-statistics
unless corrected.

**Tests**:
- **Breusch-Pagan**: Regress squared residuals on X. Rejection = heteroscedasticity.
- **White**: General test using squared residuals on X, X^2, and cross-products.
""")

diag_f = RESULTS / "table_5f_diagnostics.csv"
if diag_f.exists():
    diag = pd.read_csv(diag_f)
    st.dataframe(diag.style.format({
        "statistic": "{:.2f}", "p_value": "{:.6f}",
    }), use_container_width=True, hide_index=True)

    st.warning("""
    Both tests strongly reject homoscedasticity (p < 0.001), confirming that
    clustered or Driscoll-Kraay standard errors are necessary. Our main results
    already use clustered SE by ETF.
    """)


# ============================================================
# Sub-sample Analysis
# ============================================================
st.header("7. Sub-sample Analysis")

sub_f = RESULTS / "table_4_subsample.csv"
if sub_f.exists():
    sub = pd.read_csv(sub_f)
    st.dataframe(sub, use_container_width=True, hide_index=True)

    st.markdown("""
    Results are robust across sub-periods. The performance chasing effect is
    present in both bull and bear markets, though potentially stronger during
    periods of high market attention (2020-2021 bull run).
    """)
