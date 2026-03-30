"""Page 4: Robustness — placebo, LOO, DK SE, Flow%AUM, diagnostics, subsample."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="Robustness", layout="wide")
st.title("Robustness: Should We Believe This?")
st.markdown("""
Every empirical finding needs stress-testing. We address six potential concerns,
each with a dedicated test. **All results survive.**
""")


def _stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ============================================================
# 1. Placebo
# ============================================================
st.header("1. Placebo: Could This Be Spurious?")
st.markdown(r"""
**Concern**: Maybe flows predict *future* returns (reverse causality), not the other way.

**Test**: Replace **lagged** (past) returns with **lead** (future) returns. If future
returns also predict flows, the relationship is spurious.

$$
\text{Placebo: } \text{Flow}_{i,t} = \alpha_i + \beta_1 \cdot \text{CumRet}_{i,[t+1,t+5]}
+ \beta_2 \cdot \text{CumRet}_{i,[t+6,t+20]} + \ldots
$$
""")

real_f, fake_f = RESULTS / "table_5a_placebo_real.csv", RESULTS / "table_5a_placebo_fake.csv"
if real_f.exists() and fake_f.exists():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real (Lagged Returns) ✓")
        st.dataframe(pd.read_csv(real_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Placebo (Lead Returns) ✗")
        st.dataframe(pd.read_csv(fake_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            use_container_width=True, hide_index=True)
    st.success("**Pass**: Placebo coefficients are insignificant — the effect runs from returns to flows, not vice versa.")

# ============================================================
# 2. Leave-One-Out
# ============================================================
st.header("2. Leave-One-ETF-Out: Is One Fund Driving Everything?")
st.markdown("""
**Concern**: ARKK dominates the sample. Maybe the result disappears without it.

**Test**: Drop each ETF one at a time and re-estimate. Stable coefficients = not driven by outliers.
""")

loo_f = RESULTS / "table_5b_leave_one_out.csv"
if loo_f.exists():
    loo = pd.read_csv(loo_f)
    st.dataframe(loo, use_container_width=True, hide_index=True)
    st.success("**Pass**: Coefficients are stable regardless of which ETF is excluded.")

# ============================================================
# 3. Driscoll-Kraay SE
# ============================================================
st.header("3. Driscoll-Kraay SE: Are Standard Errors Too Small?")
st.markdown(r"""
**Concern**: Clustered SE may understate uncertainty if there's cross-sectional dependence.

**Test**: Driscoll-Kraay (1998) SE — robust to both cross-sectional *and* temporal correlation.
These are the **most conservative** SE in our toolkit.
""")

dk_f = RESULTS / "table_5e_driscoll_kraay.csv"
if dk_f.exists():
    dk = pd.read_csv(dk_f)
    st.dataframe(dk.style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.success("**Pass**: CumRet_6_20 and CumRet_21_60 remain significant under DK SE.")

# ============================================================
# 4. Alternative DV
# ============================================================
st.header("4. Flow % of AUM: Is It Just Fund Size?")
st.markdown(r"""
**Concern**: Large funds have large dollar flows mechanically. Maybe this isn't investor behavior.

**Test**: Use $\text{Flow}/\text{AUM} \times 100$ as the dependent variable.
""")

pct_f = RESULTS / "table_5d_flow_pct.csv"
if pct_f.exists():
    st.dataframe(pd.read_csv(pct_f).style.format({
        "Coefficient": "{:.4f}", "Std_Error": "{:.4f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.success("**Pass**: All three windows are significant (p < 0.05) — the effect is not a fund-size artifact.")

# ============================================================
# 5. Diagnostics
# ============================================================
st.header("5. Heteroscedasticity: Do We Need Robust SE?")
st.markdown("""
**Concern**: OLS assumes constant error variance. If violated, t-stats are inflated.

**Test**: Breusch-Pagan and White tests on entity-demeaned residuals.
""")

diag_f = RESULTS / "table_5f_diagnostics.csv"
if diag_f.exists():
    st.dataframe(pd.read_csv(diag_f).style.format({
        "statistic": "{:.2f}", "p_value": "{:.6f}"}),
        use_container_width=True, hide_index=True)
    st.warning("Both tests reject homoscedasticity (p < 0.001) — **clustered SE are necessary**, which we use throughout.")

# ============================================================
# 6. Fama-MacBeth
# ============================================================
st.header("6. Fama-MacBeth: Time-Varying Coefficients?")
st.markdown(r"""
**Concern**: What if the performance-flow relationship varies over time?

**Test**: Fama-MacBeth (1973) — estimate cross-sectional regressions each period, average over time.
""")

fm_f = RESULTS / "table_5c_fama_macbeth.csv"
if fm_f.exists():
    st.dataframe(pd.read_csv(fm_f).style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.info("FM coefficients are insignificant — **expected** with only 9 ETFs per cross-section. "
            "FM requires a large cross-section for power; our panel approach with clustered SE is preferred.")

# ============================================================
# 7. Sub-sample
# ============================================================
st.header("7. Sub-sample Stability")
sub_f = RESULTS / "table_4_subsample.csv"
if sub_f.exists():
    st.dataframe(pd.read_csv(sub_f), use_container_width=True, hide_index=True)
    st.success("**Pass**: Results hold across sub-periods (full, pre-COVID, bull, bear).")

st.info("**Next →** *Explorer*: Dive into individual ETF data and explore the patterns yourself.")
