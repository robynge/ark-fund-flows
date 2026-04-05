"""Page 6: Marketing Premium — Predicted vs Actual fund flows."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (sidebar_freq, load_data_with_controls, get_cols,
                      get_cumret_windows, cumret_cols, build_cumret,
                      ETF_NAMES, FREQ_LABELS)

st.set_page_config(page_title="Marketing Premium", layout="wide")
st.title("Marketing Premium: Actual vs. Predicted Flows")

st.markdown(r"""
We compute predicted flows from the panel FE model, then compare to actual flows.
**Excess flow** (actual > predicted) suggests something beyond performance —
such as marketing or brand recognition — is driving capital into the fund.

$$
\text{Residual}_i = \frac{1}{T_i} \sum_t (F_{i,t} - \hat{F}_{i,t})
$$

Positive residual = ETF attracts more capital than its performance alone predicts.
""")

freq = sidebar_freq(key="mktg_freq")
fc, rc = get_cols(freq)
windows = get_cumret_windows(freq)
period = FREQ_LABELS[freq]

from placebo import predicted_vs_actual

@st.cache_data(show_spinner="Computing predicted vs actual...")
def compute_pva(freq):
    df = load_data_with_controls(freq)
    df = df[df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    _windows = get_cumret_windows(freq)
    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    ark = build_cumret(ark, _rc, _windows)
    x_cols = cumret_cols(_windows)
    for c in ["VIX_Change", "Peer_Agg_Flow"]:
        if c in ark.columns:
            x_cols.append(c)
    return predicted_vs_actual(ark, _fc, x_cols)

pva = compute_pva(freq)
if not pva.empty:
    pva = pva.sort_values("mean_residual", ascending=False)
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in pva["mean_residual"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pva["ETF"], y=pva["mean_residual"], marker_color=colors,
                         hovertemplate="%{x}<br>Avg residual: %{y:.2f}M<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title="ETF",
                      yaxis_title=f"Average Residual Flow ($M/{period[:-1]})",
                      title="Marketing Premium: Excess Flow Beyond Performance")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(pva.style.format({
        "mean_actual": "{:.2f}", "mean_predicted": "{:.2f}",
        "mean_residual": "{:.2f}", "std_residual": "{:.2f}"}),
        width="stretch", hide_index=True)

    top = pva.iloc[0]
    st.success(f"**{top['ETF']}** has the highest marketing premium: "
               f"{top['mean_residual']:.2f}M/{period[:-1]} above what performance predicts.")
else:
    st.info("Not enough data to compute predicted vs actual flows.")
