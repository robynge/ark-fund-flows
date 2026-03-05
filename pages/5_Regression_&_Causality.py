"""Step 5: Formal statistical tests — OLS regression and Granger causality."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES
from analysis import lag_regression, lag_regression_all_etfs, granger_causality_test

st.set_page_config(page_title="Step 5: Regression & Causality", layout="wide")

st.title("Step 5: Regression & Causality")
st.markdown("""
Now we run formal statistical tests to quantify the relationship.

**Two questions**:
1. **OLS Regression**: Can lagged returns explain fund flows? (How much? Which lags matter?)
2. **Granger Causality**: Does past performance *statistically cause* future flows — and not vice versa?
""")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["ME", "QE"],
                        format_func={"ME": "Monthly", "QE": "Quarterly"}.get,
                        index=0)
    add_months = st.checkbox("Control for Seasonality (Month Dummies)", value=False)

freq_label = {"ME": "months", "QE": "quarters"}[freq]


@st.cache_data
def load_data(freq):
    return get_prepared_data(freq=freq, zscore_type="full")


df = load_data(freq)

flow_col = "Flow_Sum"
return_col = "Return_Cum"

lag_options = {"ME": [1, 3, 6, 12], "QE": [1, 2, 4]}
lags = lag_options[freq]

# ================================================================
# Part A: Single ETF Regression
# ================================================================
st.subheader(f"Part A: {selected_etf} — OLS Regression")
st.markdown(f"""
**Model**: Flow(t) = β₀ + β₁·Return(t−{lags[0]}) + β₂·Return(t−{lags[1]}) + β₃·Return(t−{lags[2]}) + ε
{' + month dummies' if add_months else ''}

We are asking: **do returns from {lags[0]}, {lags[1]}, {lags[2]} {freq_label} ago predict this period's fund flow?**
""")

etf_df = df[df["ETF"] == selected_etf]
result = lag_regression(etf_df, flow_col, return_col, lags, add_months)

if result is not None:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{result['r_squared']:.4f}",
                help="Fraction of flow variation explained by the model")
    col2.metric("Adj R²", f"{result['adj_r_squared']:.4f}",
                help="R² adjusted for number of predictors")
    col3.metric("F-test p-value", f"{result['f_pvalue']:.4f}",
                help="p < 0.05 means the model is significant overall")
    col4.metric("Observations", f"{result['n_obs']}")

    if result['f_pvalue'] < 0.05:
        st.success(f"The model is **statistically significant** (F-test p = {result['f_pvalue']:.4f} < 0.05).")
    else:
        st.warning(f"The model is **not significant** (F-test p = {result['f_pvalue']:.4f} > 0.05).")

    # Coefficients table — only show lag variables
    st.markdown("**Coefficient Table** (lag variables only):")
    coef = result["coefficients"].copy()
    lag_coef = coef[coef["Variable"].str.contains("Return_lag")].copy()
    lag_coef["Lag"] = lag_coef["Variable"].str.extract(r"(\d+)").astype(int)
    lag_coef["Significant"] = lag_coef["p_value"].apply(
        lambda p: "*** (p<0.01)" if p < 0.01 else ("** (p<0.05)" if p < 0.05 else ("* (p<0.1)" if p < 0.1 else "No")))

    st.dataframe(
        lag_coef[["Lag", "Coefficient", "Std_Error", "t_stat", "p_value", "Significant"]].style.format({
            "Coefficient": "{:.2f}", "Std_Error": "{:.2f}", "t_stat": "{:.3f}", "p_value": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("""
    **How to read this**:
    - A **positive coefficient** means higher past returns → more inflows (performance chasing)
    - **p-value < 0.05** means the relationship is statistically significant
    - The coefficient tells you: for every 1% higher return X months ago, flows change by $β M
    """)

    # Bar chart of coefficients
    fig_coef = go.Figure()
    for _, row in lag_coef.iterrows():
        color = "#2ca02c" if row["p_value"] < 0.05 else "#c7c7c7"
        fig_coef.add_trace(go.Bar(
            x=[f"Lag {int(row['Lag'])}"], y=[row["Coefficient"]],
            marker_color=color, showlegend=False,
            error_y=dict(type="data", array=[1.96 * row["Std_Error"]], visible=True),
        ))
    fig_coef.update_layout(
        height=350, yaxis_title="Coefficient ($ per 1 unit return)",
        title="Regression Coefficients by Lag (green = significant)",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig_coef.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_coef, use_container_width=True)

else:
    st.warning("Not enough data for regression.")

# ================================================================
# Part B: All ETFs Summary
# ================================================================
st.markdown("---")
st.subheader("Part B: Regression Summary — All ETFs")


@st.cache_data
def run_all(freq, lags_tuple, add_months):
    return lag_regression_all_etfs(df, flow_col, return_col, list(lags_tuple), add_months)


summary = run_all(freq, tuple(lags), add_months)

if len(summary) > 0:
    # Simplify display
    display_cols = ["ETF", "R²", "Adj_R²", "F_p_value", "N"]
    for lag in lags:
        if f"Return_lag{lag}" in summary.columns:
            display_cols.extend([f"Return_lag{lag}", f"Return_lag{lag}_pval"])

    st.dataframe(
        summary[display_cols].style.format({
            "R²": "{:.4f}", "Adj_R²": "{:.4f}", "F_p_value": "{:.4f}",
            **{f"Return_lag{l}": "{:.1f}" for l in lags if f"Return_lag{l}" in summary.columns},
            **{f"Return_lag{l}_pval": "{:.4f}" for l in lags if f"Return_lag{l}_pval" in summary.columns},
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # R² bar chart
    fig_r2 = go.Figure(go.Bar(
        x=summary["ETF"], y=summary["R²"],
        marker_color="#3182bd",
        text=[f"{r:.3f}" for r in summary["R²"]],
        textposition="outside",
    ))
    fig_r2.update_layout(height=350, yaxis_title="R²",
                         title="Model Explanatory Power (R²) by ETF",
                         margin=dict(l=60, r=40, t=60, b=40))
    st.plotly_chart(fig_r2, use_container_width=True)

# ================================================================
# Part C: Granger Causality
# ================================================================
st.markdown("---")
st.subheader(f"Part C: Granger Causality — {selected_etf}")
st.markdown("""
**Granger causality** tests whether knowing past values of X improves our prediction of Y
beyond just using past values of Y alone.

We test **both directions**:
- **Returns → Flows**: Does past performance help predict future flows? *(Expected: Yes)*
- **Flows → Returns**: Do past flows help predict future returns? *(Expected: No)*

If only the first direction is significant, this confirms the **performance-chasing** story:
investors react to past performance, but their flows don't move future prices.
""")

gc_max = st.slider("Max lag for Granger test", 1, 8, {"ME": 6, "QE": 4}.get(freq, 4))


@st.cache_data
def run_granger(etf, freq, gc_max):
    edf = df[df["ETF"] == etf]
    return granger_causality_test(edf, flow_col, return_col, gc_max)


gc = run_granger(selected_etf, freq, gc_max)

if len(gc) > 0:
    # Split into two tables
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Returns → Flows** (performance chasing)")
        gc_rf = gc[gc["direction"] == "Returns → Flows"].copy()
        gc_rf["Significant"] = gc_rf["p_value"].apply(
            lambda p: "✓ Yes" if p < 0.05 else "✗ No")
        st.dataframe(gc_rf[["lag", "F_statistic", "p_value", "Significant"]].style.format({
            "F_statistic": "{:.2f}", "p_value": "{:.4f}",
        }), use_container_width=True, hide_index=True)

        any_sig_rf = (gc_rf["p_value"] < 0.05).any()
        if any_sig_rf:
            st.success("✓ Past returns DO Granger-cause flows")
        else:
            st.error("✗ No significant Granger causality")

    with col_right:
        st.markdown("**Flows → Returns** (flow impact)")
        gc_fr = gc[gc["direction"] == "Flows → Returns"].copy()
        gc_fr["Significant"] = gc_fr["p_value"].apply(
            lambda p: "✓ Yes" if p < 0.05 else "✗ No")
        st.dataframe(gc_fr[["lag", "F_statistic", "p_value", "Significant"]].style.format({
            "F_statistic": "{:.2f}", "p_value": "{:.4f}",
        }), use_container_width=True, hide_index=True)

        any_sig_fr = (gc_fr["p_value"] < 0.05).any()
        if not any_sig_fr:
            st.success("✓ Flows do NOT Granger-cause returns (as expected)")
        else:
            st.warning("⚠ Flows also Granger-cause returns (bidirectional)")

    # Summary across all ETFs
    st.markdown("---")
    st.markdown("**Granger Causality Summary — All ETFs**")

    gc_summary = []
    for etf in ETF_NAMES:
        gc_etf = run_granger(etf, freq, gc_max)
        if len(gc_etf) == 0:
            continue
        rf = gc_etf[gc_etf["direction"] == "Returns → Flows"]
        fr = gc_etf[gc_etf["direction"] == "Flows → Returns"]
        rf_sig = "✓" if (rf["p_value"] < 0.05).any() else "✗"
        fr_sig = "✓" if (fr["p_value"] < 0.05).any() else "✗"
        rf_best_p = rf["p_value"].min() if len(rf) > 0 else None
        fr_best_p = fr["p_value"].min() if len(fr) > 0 else None
        gc_summary.append({
            "ETF": etf,
            "Returns→Flows": rf_sig,
            "Best p-value": round(rf_best_p, 4) if rf_best_p else None,
            "Flows→Returns": fr_sig,
            "Best p-value ": round(fr_best_p, 4) if fr_best_p else None,
        })

    gc_sum_df = pd.DataFrame(gc_summary)
    st.dataframe(gc_sum_df, use_container_width=True, hide_index=True)

else:
    st.info("Not enough data for Granger causality test at this frequency.")

st.info("👉 **Next Step**: Go to **Step 6: Seasonality** to check for calendar effects.")
