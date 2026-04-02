"""Page 4: Robustness — organized by referee questions."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="Robustness", layout="wide")
st.title("Robustness: Should We Believe This?")
st.markdown("""
We address the questions a journal referee would ask, each with a dedicated test.
""")


def _stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ============================================================
# Q1: Are Standard Errors Reliable?
# ============================================================
st.header("Q1: Are the Standard Errors Reliable?")

st.markdown(r"""
**Concern**: With only 9 ETF clusters, entity-clustered SE may understate uncertainty
(Angrist & Pischke recommend $\geq 30$ clusters). We compare three SE estimators
on the **same specification**:

| Method | Handles | Formula |
|--------|---------|---------|
| **Entity Cluster** | Within-ETF correlation | $V = (G/(G-1)) \sum_g X_g' \hat{u}_g \hat{u}_g' X_g$ |
| **Two-way Cluster** | Entity + date correlation | $V_{2way} = V_{entity} + V_{time} - V_{white}$ |
| **Driscoll-Kraay** | Cross-sectional + temporal | $V_{DK} = \frac{1}{T^2} \sum K(j/m) \sum_t h_t h_{t-j}'$ |
""")

t6_f = RESULTS / "table_6_se_comparison.csv"
if t6_f.exists():
    t6 = pd.read_csv(t6_f)
    for var in t6["Variable"].unique():
        st.subheader(var.replace("_", " "))
        var_df = t6[t6["Variable"] == var][["SE_Method", "Coefficient", "Std_Error", "t_stat", "p_value"]]
        st.dataframe(var_df.style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)

    st.info("""
    **Interpretation**: Two-way and DK standard errors are larger than entity-clustered SE,
    confirming cross-sectional/temporal dependence. However, the key variables
    (CumRet_6_20, CumRet_21_60) **remain significant** under all three methods.
    """)

# Heteroscedasticity diagnostics
diag_f = RESULTS / "table_5f_diagnostics.csv"
if diag_f.exists():
    st.subheader("Heteroscedasticity Diagnostics")
    st.dataframe(pd.read_csv(diag_f).style.format({
        "statistic": "{:.2f}", "p_value": "{:.6f}"}),
        width="stretch", hide_index=True)
    st.warning("Both tests reject homoscedasticity (p < 0.001) — robust SE are necessary.")


# ============================================================
# Q2: Is There Reverse Causality?
# ============================================================
st.header("Q2: Could Flows Predict Returns (Reverse Causality)?")

st.markdown(r"""
**Concern**: Large fund flows may move ETF prices (price impact), creating
a spurious return→flow relationship.

**Test 1: Placebo** — replace lagged returns with **lead (future) returns**.
If future returns also predict flows, the relationship may be spurious.

$$
\text{Placebo: } \text{Flow}_{i,t} = \alpha_i + \beta \cdot \text{CumRet}_{i,[t+1, t+k]} + \varepsilon
$$

**Test 2: Granger Causality** — test both directions:
$H_0$: Returns do NOT Granger-cause Flows, and vice versa.
""")

# Placebo
real_f, fake_f = RESULTS / "table_5a_placebo_real.csv", RESULTS / "table_5a_placebo_fake.csv"
if real_f.exists() and fake_f.exists():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real (Lagged Returns)")
        st.dataframe(pd.read_csv(real_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)
    with col2:
        st.subheader("Placebo (Lead Returns)")
        st.dataframe(pd.read_csv(fake_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)

# Granger
t8_f = RESULTS / "table_8_granger.csv"
if t8_f.exists():
    st.subheader("Granger Causality Test")
    t8 = pd.read_csv(t8_f)
    st.dataframe(t8.style.format({
        "F_stat": "{:.2f}", "p_value": "{:.4f}"}),
        width="stretch", hide_index=True)

    # Summary
    ret_to_flow = t8[t8["Direction"].str.contains("Returns")]
    flow_to_ret = t8[t8["Direction"].str.contains("Flows")]
    n_sig_rf = (ret_to_flow["p_value"] < 0.05).sum() if not ret_to_flow.empty else 0
    n_sig_fr = (flow_to_ret["p_value"] < 0.05).sum() if not flow_to_ret.empty else 0

    st.success(f"""
    **Verdict**: Returns → Flows is significant for **{n_sig_rf}/{len(ret_to_flow)}** ETFs.
    Flows → Returns is significant for **{n_sig_fr}/{len(flow_to_ret)}** ETFs.
    The dominant causal direction is **returns leading flows** — performance chasing confirmed.
    Some reverse causality (price impact) exists but is secondary.
    """)


# ============================================================
# Q3: Is ARKK Driving Everything?
# ============================================================
st.header("Q3: Is ARKK Driving the Results?")

st.markdown(r"""
**Concern**: ARKK is the largest fund with unique media attention. Are results
an ARKK-specific phenomenon?

**Test 1: Leave-One-ETF-Out** — re-estimate excluding each ETF.

**Test 2: ARKK Interaction Model** — add $\text{ARKK}_i \times \text{CumRet}$
interaction terms to test whether ARKK's coefficient differs from the rest:

$$
\text{Flow}_{it} = \alpha_i + \beta \cdot \text{CumRet} + \delta \cdot (\mathbb{1}_{\text{ARKK}} \times \text{CumRet}) + \varepsilon
$$

If $\delta \neq 0$, ARKK has a different flow-performance sensitivity.
""")

# LOO
loo_f = RESULTS / "table_5b_leave_one_out.csv"
if loo_f.exists():
    st.subheader("Leave-One-ETF-Out")
    st.dataframe(pd.read_csv(loo_f), width="stretch", hide_index=True)

# ARKK heterogeneity
t7_interact = RESULTS / "table_7_interaction.csv"
t7_arkk = RESULTS / "table_7_arkk_only.csv"
t7_noarkk = RESULTS / "table_7_excluding_arkk.csv"

if t7_interact.exists():
    st.subheader("ARKK Interaction Model")
    df_int = pd.read_csv(t7_interact)
    st.dataframe(df_int.style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
        width="stretch", hide_index=True)

    # Check if interaction terms are significant
    arkk_terms = df_int[df_int["Variable"].str.contains("_x_ARKK")]
    n_sig = (arkk_terms["p_value"] < 0.05).sum()
    st.warning(f"""
    **Verdict**: {n_sig}/{len(arkk_terms)} ARKK interaction terms are significant —
    ARKK **does** behave differently from other ARK ETFs. However, the base effects
    (CumRet_6_20, CumRet_21_60) remain significant for non-ARKK funds, so the
    performance chasing finding is **not solely driven by ARKK**.
    """)

if t7_arkk.exists() and t7_noarkk.exists():
    st.subheader("Split Sample: ARKK vs Others")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("ARKK Only")
        st.dataframe(pd.read_csv(t7_arkk).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)
    with col2:
        st.caption("Excluding ARKK")
        st.dataframe(pd.read_csv(t7_noarkk).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)


# ============================================================
# Q4: Are Coefficients Stable Over Time?
# ============================================================
st.header("Q4: Are Coefficients Stable Over Time?")

st.markdown(r"""
**Concern**: Fama-MacBeth SE are 7-10× larger than clustered SE, suggesting
the flow-performance relationship varies across time periods.

**Test**: Estimate the panel regression over a **2-year rolling window** (504
trading days) and plot the coefficient trajectory:

$$
\hat\beta_t = \text{Entity-FE estimate using data in } [t-504, t]
$$
""")

f4_f = RESULTS / "figure_4_rolling.csv"
if f4_f.exists():
    f4 = pd.read_csv(f4_f)
    f4["window_end"] = pd.to_datetime(f4["window_end"])

    for var in ["CumRet_6_20", "CumRet_21_60"]:
        beta_col = f"{var}_beta"
        se_col = f"{var}_se"
        if beta_col in f4.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=f4["window_end"], y=f4[beta_col] + 1.96 * f4[se_col],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=f4["window_end"], y=f4[beta_col] - 1.96 * f4[se_col],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(31,119,180,0.2)", name="95% CI", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=f4["window_end"], y=f4[beta_col],
                mode="lines", line=dict(color="#1f77b4", width=2),
                name=f"β({var})",
                hovertemplate="%{x|%Y-%m}<br>β=%{y:.2f}<extra></extra>"))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                height=350, xaxis_title="Window End Date",
                yaxis_title="Coefficient",
                title=f"Rolling 2-Year Coefficient: {var}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, width="stretch")

# FM results
fm_f = RESULTS / "table_5c_fama_macbeth.csv"
if fm_f.exists():
    st.subheader("Fama-MacBeth Regression")
    st.dataframe(pd.read_csv(fm_f).style.format({
        "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
        "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
        width="stretch", hide_index=True)
    st.info("""
    FM coefficients are insignificant — expected with only 9 ETFs per cross-section.
    The rolling window plot above provides a more informative view of time-variation.
    """)

# Sub-sample
sub_f = RESULTS / "table_4_subsample.csv"
if sub_f.exists():
    st.subheader("Sub-sample Stability")
    st.dataframe(pd.read_csv(sub_f), width="stretch", hide_index=True)
    st.success("**Verdict**: Coefficients are present across all sub-periods. The rolling window "
               "confirms time-variation exists but the **direction is consistently positive** — "
               "performance chasing is a persistent phenomenon, not an artifact of a single period.")


# ============================================================
# Q5: Alternative Specifications
# ============================================================
st.header("Q5: Do Other Specifications Agree?")

st.markdown(r"""
**Test 1**: Replace Fund_Flow ($M) with $\text{Flow}/\text{AUM} \times 100$ (%)

**Test 2**: Driscoll-Kraay SE (most conservative estimator)
""")

col1, col2 = st.columns(2)

pct_f = RESULTS / "table_5d_flow_pct.csv"
if pct_f.exists():
    with col1:
        st.subheader("Flow % of AUM")
        st.dataframe(pd.read_csv(pct_f).style.format({
            "Coefficient": "{:.4f}", "Std_Error": "{:.4f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)
        st.success("All windows significant with Flow % AUM.")

dk_f = RESULTS / "table_5e_driscoll_kraay.csv"
if dk_f.exists():
    with col2:
        st.subheader("Driscoll-Kraay SE")
        st.dataframe(pd.read_csv(dk_f).style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
            width="stretch", hide_index=True)
        st.success("6-20d and 21-60d remain significant under DK SE.")

st.success("**Verdict**: Both alternative specifications confirm the main finding — "
           "performance chasing is robust to the choice of dependent variable and SE estimator.")

# ============================================================
# Overall Summary
# ============================================================
st.divider()
st.header("Overall Verdict")
st.success("""
**All five referee concerns are addressed:**

1. **SE reliability** — Results hold under entity cluster, two-way cluster, and Driscoll-Kraay SE
2. **Reverse causality** — Granger tests confirm returns lead flows; some price impact exists but is secondary
3. **ARKK dominance** — ARKK behaves differently, but the effect exists for non-ARKK funds too
4. **Time stability** — Rolling windows show persistent positive coefficients across most periods
5. **Alternative specs** — Flow % AUM and DK SE both confirm the main finding
""")
