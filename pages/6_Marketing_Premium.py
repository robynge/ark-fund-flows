"""Page 6: Marketing Premium — Predicted vs Actual fund flows."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"

st.set_page_config(page_title="Marketing Premium", layout="wide")
st.title("Marketing Premium: Actual vs. Predicted Flows")

st.markdown(r"""
If fund flows are solely driven by performance, then our regression model should
fully explain them. Any **excess flow** (actual > predicted) suggests that
something beyond performance — such as **marketing, media attention, or brand
recognition** — is driving capital into the fund.

We compute predicted flows from the panel FE model:

$$
\hat{F}_{i,t} = \hat\alpha_i + \hat\beta_1 \cdot \text{CumRet}_{1\text{-}5}
+ \hat\beta_2 \cdot \text{CumRet}_{6\text{-}20}
+ \hat\beta_3 \cdot \text{CumRet}_{21\text{-}60}
+ \hat\gamma' Z_{i,t}
$$

Then the **marketing premium** (or residual) for each ETF is:

$$
\text{Residual}_i = \frac{1}{T_i} \sum_t (F_{i,t} - \hat{F}_{i,t})
$$

A positive residual means the ETF receives **more flow than its performance
alone would predict**.

> **Reading the chart**: Bars above zero = ETF attracts more capital than
> performance predicts (marketing premium). Bars below zero = underperformance
> in attracting capital relative to what the model expects.
""")

pva_f = RESULTS / "predicted_vs_actual.csv"
if pva_f.exists():
    pva = pd.read_csv(pva_f)

    # Sort by residual
    pva = pva.sort_values("mean_residual", ascending=False)

    colors = ["#2ca02c" if r > 0 else "#d62728" for r in pva["mean_residual"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pva["ETF"], y=pva["mean_residual"],
        marker_color=colors,
        hovertemplate="%{x}<br>Avg residual: %{y:.2f}M/day<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=450,
        xaxis_title="ETF",
        yaxis_title="Average Residual Flow ($M/day)",
        title="Marketing Premium: Average Excess Flow Beyond Performance")
    st.plotly_chart(fig, width="stretch")

    # Detailed table
    st.subheader("Detail by ETF")
    st.dataframe(pva.style.format({
        "mean_actual": "{:.2f}", "mean_predicted": "{:.2f}",
        "mean_residual": "{:.2f}", "std_residual": "{:.2f}",
    }), width="stretch", hide_index=True)

    # Interpretation
    top = pva.iloc[0]
    bottom = pva.iloc[-1]
    st.success(f"""
    **Result**: **{top['ETF']}** has the highest marketing premium
    ({top['mean_residual']:.2f}M/day above predicted), while **{bottom['ETF']}**
    has the lowest ({bottom['mean_residual']:.2f}M/day).
    ETFs with positive residuals attract more capital than their performance
    alone would explain — consistent with effective marketing, media visibility,
    or strong brand recognition.
    """)
else:
    st.info("Run `python -m experiments.new_runner --all` to generate predicted vs actual data.")
