"""Time series regression results."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import get_prepared_data, ETF_NAMES
from analysis import lag_regression, lag_regression_all_etfs, granger_causality_test

st.set_page_config(page_title="Regression Results", layout="wide")
st.title("Regression Results")

with st.sidebar:
    selected_etf = st.selectbox("Select ETF", ETF_NAMES, index=0)
    freq = st.selectbox("Frequency", ["D", "W", "ME", "QE"],
                        format_func=lambda x: {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}[x])
    add_month_dummies = st.checkbox("Include Month Dummies (Seasonality Control)", value=False)

    # Lag selection depends on frequency
    lag_options = {
        "D": [1, 5, 21, 63, 126, 252],
        "W": [1, 4, 13, 26, 52],
        "ME": [1, 3, 6, 12],
        "QE": [1, 2, 4],
    }
    available_lags = lag_options[freq]
    selected_lags = st.multiselect("Lags to Include", available_lags, default=available_lags[:3])


@st.cache_data
def load_data(freq):
    return get_prepared_data(freq=freq, zscore_type="full")


df = load_data(freq)

flow_col = "Fund_Flow" if freq == "D" else "Flow_Sum"
return_col = "Return" if freq == "D" else "Return_Cum"

if not selected_lags:
    st.warning("Select at least one lag.")
    st.stop()

# Single ETF detailed regression
st.subheader(f"{selected_etf}: OLS Regression — Flow(t) ~ Lagged Returns")
st.markdown(f"**Lags**: {selected_lags} | **Month dummies**: {'Yes' if add_month_dummies else 'No'}")

etf_df = df[df["ETF"] == selected_etf]
result = lag_regression(etf_df, flow_col, return_col, selected_lags, add_month_dummies)

if result is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{result['r_squared']:.4f}")
    col2.metric("Adj R²", f"{result['adj_r_squared']:.4f}")
    col3.metric("F-test p-value", f"{result['f_pvalue']:.4e}")
    col4.metric("N obs", f"{result['n_obs']:,}")

    # Coefficients table
    coef_df = result["coefficients"].copy()
    coef_df["Significant"] = coef_df["p_value"].apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))
    st.dataframe(coef_df.style.format({
        "Coefficient": "{:.6f}",
        "Std_Error": "{:.6f}",
        "t_stat": "{:.3f}",
        "p_value": "{:.4f}",
    }), use_container_width=True, hide_index=True)
else:
    st.warning("Not enough data for regression.")

# Summary across all ETFs
st.markdown("---")
st.subheader("Regression Summary Across All ETFs")


@st.cache_data
def run_all_regressions(freq, lags_tuple, add_month_dummies):
    return lag_regression_all_etfs(df, flow_col, return_col, list(lags_tuple), add_month_dummies)


summary = run_all_regressions(freq, tuple(selected_lags), add_month_dummies)

if len(summary) > 0:
    st.dataframe(summary.style.format({
        "R²": "{:.4f}",
        "Adj_R²": "{:.4f}",
        "F_p_value": "{:.4e}",
    }, na_rep="—"), use_container_width=True, hide_index=True)

    # Bar chart of R² across ETFs
    fig = go.Figure(go.Bar(
        x=summary["ETF"], y=summary["R²"],
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        height=350,
        yaxis_title="R²",
        title="Model R² by ETF",
        margin=dict(l=60, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# Granger Causality
st.markdown("---")
st.subheader(f"{selected_etf}: Granger Causality Tests")
st.markdown("Tests whether past values of one series help predict the other beyond its own history.")

gc_max_lag = st.slider("Max Lag for Granger Test", 1, 10, 5)


@st.cache_data
def run_granger(etf_name, freq, gc_max_lag):
    edf = df[df["ETF"] == etf_name]
    return granger_causality_test(edf, flow_col, return_col, gc_max_lag)


gc_results = run_granger(selected_etf, freq, gc_max_lag)

if len(gc_results) > 0:
    gc_results["Significant"] = gc_results["p_value"].apply(
        lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    )
    st.dataframe(gc_results.style.format({
        "F_statistic": "{:.3f}",
        "p_value": "{:.4f}",
    }), use_container_width=True, hide_index=True)
else:
    st.info("Not enough data for Granger causality test.")
