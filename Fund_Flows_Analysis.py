"""ARK ETF Fund Flows Analysis Dashboard — Step-by-Step"""
import streamlit as st

st.set_page_config(
    page_title="ARK ETF Fund Flows Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ARK ETF Fund Flows vs Stock Price")
st.markdown("### Do investors chase performance? A step-by-step analysis.")

st.markdown("---")

st.markdown("""
## Research Question

**Is there a relationship between ARK ETF stock prices and fund flows?**

The academic literature suggests that fund flows *follow* past performance — investors
buy after prices go up and sell after prices go down. But they don't do it immediately.
There is a **lag effect**: it takes weeks, months, or even quarters for investors to react.

This dashboard walks through the analysis step by step:

| Step | Page | What We Do |
|------|------|-----------|
| 1 | **Raw Data** | Look at the raw price and flow data |
| 2 | **The Problem** | See why daily data is too noisy to find a relationship |
| 3 | **Normalize & Aggregate** | Clean the data: z-score normalization + aggregate to weekly/monthly/quarterly |
| 4 | **Lag Analysis** | Test correlations at different lag horizons to find where the relationship shows up |
| 5 | **Regression & Causality** | Formal statistical tests: OLS regressions and Granger causality |
| 6 | **Seasonality** | Check for calendar effects (e.g., January reallocation) |
| 7 | **Conclusions** | Summary of findings |

👈 **Use the sidebar to navigate through each step.**
""")
