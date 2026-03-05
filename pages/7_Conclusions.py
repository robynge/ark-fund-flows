"""Step 7: Conclusions — summary of all findings."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import get_prepared_data, ETF_NAMES
from analysis import lag_regression, granger_causality_test

st.set_page_config(page_title="Step 7: Conclusions", layout="wide")

st.title("Step 7: Conclusions")
st.markdown("---")


@st.cache_data
def load_all_freqs():
    return {
        "D": get_prepared_data(freq="D", zscore_type="full"),
        "ME": get_prepared_data(freq="ME", zscore_type="full"),
        "QE": get_prepared_data(freq="QE", zscore_type="full"),
    }


data = load_all_freqs()

# ================================================================
# Finding 1: The relationship exists but depends on time horizon
# ================================================================
st.subheader("Finding 1: The Relationship Exists — But Only at Longer Horizons")
st.markdown("""
Daily data is too noisy to see any relationship. But when we aggregate to monthly
or quarterly frequency, a clear pattern emerges.
""")

# Compute R² at each frequency for each ETF
r2_summary = []
configs = [
    ("D", "Fund_Flow", "Return", [1, 5, 21], "Daily"),
    ("ME", "Flow_Sum", "Return_Cum", [1, 3, 6], "Monthly"),
    ("QE", "Flow_Sum", "Return_Cum", [1, 2], "Quarterly"),
]
for freq, fc, rc, lags, label in configs:
    df_f = data[freq]
    for etf in ETF_NAMES:
        edf = df_f[df_f["ETF"] == etf]
        r = lag_regression(edf, fc, rc, lags)
        if r is None:
            continue
        r2_summary.append({"ETF": etf, "Frequency": label, "R²": r["r_squared"],
                           "Significant": "✓" if r["f_pvalue"] < 0.05 else "✗"})

r2_df = pd.DataFrame(r2_summary)

if len(r2_df) > 0:
    # Pivot for display
    pivot = r2_df.pivot_table(index="ETF", columns="Frequency", values="R²")
    pivot = pivot[["Daily", "Monthly", "Quarterly"]]

    fig1 = go.Figure()
    colors = {"Daily": "#aec7e8", "Monthly": "#3182bd", "Quarterly": "#08519c"}
    for freq_name in ["Daily", "Monthly", "Quarterly"]:
        if freq_name in pivot.columns:
            fig1.add_trace(go.Bar(
                x=pivot.index, y=pivot[freq_name],
                name=freq_name, marker_color=colors[freq_name],
            ))
    fig1.update_layout(
        barmode="group", height=400,
        yaxis_title="R² (Explanatory Power)",
        title="How Much Do Lagged Returns Explain Fund Flows?",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)

    st.markdown("""
    **Interpretation**:
    - **Daily R² ≈ 0.5%**: lagged returns explain almost nothing at daily frequency
    - **Monthly R² ≈ 5-17%**: signal starts to emerge
    - **Quarterly R² ≈ 19-53%**: strong explanatory power for some ETFs

    This confirms the professor's guidance: **you need to aggregate the data** to see the relationship.
    """)

# ================================================================
# Finding 2: Causality runs one way
# ================================================================
st.markdown("---")
st.subheader("Finding 2: Performance Drives Flows, Not the Other Way Around")
st.markdown("""
Granger causality tests confirm the academic literature:
- **Returns → Flows**: significant for most ETFs ✓
- **Flows → Returns**: not significant for most ETFs ✗

This means investors **chase past performance**, but their buying/selling does not predict future returns.
""")

gc_summary = []
for etf in ETF_NAMES:
    edf = data["ME"][data["ME"]["ETF"] == etf]
    gc = granger_causality_test(edf, "Flow_Sum", "Return_Cum", max_lag=6)
    if len(gc) == 0:
        continue
    rf = gc[gc["direction"] == "Returns → Flows"]
    fr = gc[gc["direction"] == "Flows → Returns"]
    gc_summary.append({
        "ETF": etf,
        "Returns → Flows": "✓ Significant" if (rf["p_value"] < 0.05).any() else "✗ Not significant",
        "p-value (best)": rf["p_value"].min() if len(rf) > 0 else None,
        "Flows → Returns": "✓ Significant" if (fr["p_value"] < 0.05).any() else "✗ Not significant",
        "p-value (best) ": fr["p_value"].min() if len(fr) > 0 else None,
    })

gc_df = pd.DataFrame(gc_summary)
if len(gc_df) > 0:
    st.dataframe(gc_df.style.format({
        "p-value (best)": "{:.4f}",
        "p-value (best) ": "{:.4f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)

# ================================================================
# Finding 3: Which lag matters most?
# ================================================================
st.markdown("---")
st.subheader("Finding 3: The Lag is 1-2 Quarters")
st.markdown("""
The strongest predictive power comes from returns **1-2 quarters ago** (3-6 months).
This aligns with typical investor behavior: they see end-of-quarter performance reports,
evaluate their portfolios, and make allocation changes in the following quarter.
""")

# Show quarterly regression details for key ETFs
key_etfs = ["ARKK", "ARKG", "ARKQ"]
for etf in key_etfs:
    edf = data["QE"][data["QE"]["ETF"] == etf]
    r = lag_regression(edf, "Flow_Sum", "Return_Cum", [1, 2, 4])
    if r is None:
        continue
    coef = r["coefficients"]
    lag_coef = coef[coef["Variable"].str.contains("Return_lag")]

    st.markdown(f"**{etf}** (R² = {r['r_squared']:.4f}):")
    for _, row in lag_coef.iterrows():
        lag_num = int(row["Variable"].replace("Return_lag", ""))
        sig = "✓" if row["p_value"] < 0.05 else ""
        st.markdown(f"- Lag {lag_num} quarter: coef = {row['Coefficient']:+.1f}, p = {row['p_value']:.4f} {sig}")

# ================================================================
# Finding 4: Not all ETFs are equal
# ================================================================
st.markdown("---")
st.subheader("Finding 4: The Effect Varies by ETF")
st.markdown("""
The performance-chasing effect is **strongest** for the more popular, thematic ETFs
and **weakest** for niche ETFs:

| Strong Effect | Weak/No Effect |
|--------------|----------------|
| ARKG (Genomics) — R² up to 43% | PRNT (3D Printing) — R² near 0% |
| ARKQ (Autonomous Tech) — R² up to 53% | ARKW (Next Gen Internet) — Mixed |
| ARKK (Innovation) — R² up to 19% | ARKX (Space) — Limited data |

This makes sense: popular thematic ETFs attract more retail investors who are more
prone to performance chasing.
""")

# ================================================================
# Summary
# ================================================================
st.markdown("---")
st.subheader("Summary")

st.markdown("""
| Question | Answer |
|----------|--------|
| Is there a relationship between fund flows and stock price? | **Yes**, but only visible at monthly/quarterly frequency |
| Which direction is the causality? | **Returns → Flows** (performance chasing), not the reverse |
| How strong is the effect? | R² ranges from 5% (monthly) to 53% (quarterly) depending on ETF |
| What is the lag? | **1-2 quarters** (3-6 months) |
| Is there seasonality? | Some evidence of Q1 inflows (January reallocation), but weak |
| Are all ETFs the same? | No — popular thematic ETFs (ARKG, ARKQ) show stronger effects |

### In Plain English

> Investors look at how ARK ETFs performed last quarter. If the performance was good,
> they put more money in. If it was bad, they pull money out — but more slowly.
> This happens with a 1-2 quarter delay. However, this flow of money does **not**
> predict future returns. The crowd arrives late.
""")
