"""Drawdown Analysis — Separate page for drawdown episodes and post-drawdown flow behavior."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES, PEER_ETF_NAMES, ALL_ETF_NAMES,
)
from analysis import (
    compute_etf_drawdowns, drawdown_flow_analysis, drawdown_flow_regression,
)

st.set_page_config(page_title="Drawdown Analysis", layout="wide")
st.title("Drawdown Analysis")

# --- Sidebar (mirrors main page) ---
ETF_FULL_NAMES = {
    "ARKK": "ARK Innovation", "ARKF": "ARK Fintech Innovation",
    "ARKG": "ARK Genomic Revolution", "ARKX": "ARK Space & Defense Innovation",
    "ARKB": "ARK 21Shares Bitcoin", "ARKQ": "ARK Autonomous Tech & Robotics",
    "ARKW": "ARK Next Generation Internet", "PRNT": "The 3D Printing ETF",
    "IZRL": "ARK Israel Innovative Technology", "FTXL": "First Trust Nasdaq Semiconductor",
    "PSI": "Invesco Semiconductors", "SMH": "VanEck Semiconductor",
    "SOXX": "iShares Semiconductor", "PTF": "Invesco DW Tech Momentum",
    "XSD": "SPDR S&P Semiconductor", "PSCT": "Invesco S&P SmallCap IT",
    "IGPT": "Invesco AI & Next Gen Software", "KNCT": "Invesco Next Gen Connectivity",
    "IXN": "iShares Global Tech", "IGM": "iShares Expanded Tech Sector",
    "IYW": "iShares U.S. Technology", "XLK": "Technology Select Sector SPDR",
    "FTEC": "Fidelity MSCI IT Index", "VGT": "Vanguard Information Technology",
    "TDIV": "First Trust NASDAQ Tech Dividend", "QTEC": "First Trust NASDAQ-100 Tech",
    "FID": "First Trust S&P Intl Dividend", "FXL": "First Trust Tech AlphaDEX",
    "ERTH": "Invesco MSCI Sustainable Future", "XT": "iShares Exponential Technologies",
    "GAMR": "Amplify Video Game Tech", "CQQQ": "Invesco China Technology",
    "FDN": "First Trust DJ Internet Index", "HACK": "Amplify Cybersecurity",
    "PNQI": "Invesco NASDAQ Internet", "SKYY": "First Trust Cloud Computing",
    "CIBR": "First Trust NASDAQ Cybersecurity", "SOCL": "Global X Social Media",
}

FLOW_UNIT_OPTIONS = {"Raw $ (millions)": "raw", "% of AUM": "pct"}
TIME_RANGE_OPTIONS = ["From 2014-10-31", "Last 5 Years", "Since Pandemic (2020-03)", "Custom"]

with st.sidebar:
    freq = st.selectbox(
        "Frequency", ["D", "W", "ME", "QE"],
        format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly"}.get,
        index=2, key="dd_freq",
    )
    flow_unit_label = st.selectbox("Flow Unit", list(FLOW_UNIT_OPTIONS.keys()), key="dd_flow_unit")
    flow_unit = FLOW_UNIT_OPTIONS[flow_unit_label]
    selected_etf = st.selectbox(
        "Per-ETF View", ALL_ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_FULL_NAMES.get(x, x)}",
        key="dd_etf",
    )

    st.markdown("---")
    time_range = st.selectbox("Time Range", TIME_RANGE_OPTIONS, index=0, key="dd_time")
    if time_range == "Custom":
        date_start = st.date_input("Start date", value=pd.Timestamp("2014-01-01"), key="dd_start")
        date_end = st.date_input("End date", value=pd.Timestamp("2026-12-31"), key="dd_end")
    elif time_range == "Last 5 Years":
        date_end = pd.Timestamp("today")
        date_start = date_end - pd.DateOffset(years=5)
    elif time_range == "Since Pandemic (2020-03)":
        date_start = pd.Timestamp("2020-03-01")
        date_end = pd.Timestamp("today")
    else:
        date_start = None
        date_end = None

    st.markdown("---")
    st.caption("38 ETFs: 9 ARK + 29 tech peers")


@st.cache_data(show_spinner="Loading 38 ETFs...")
def load_peer_data(freq, benchmark):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark=benchmark)


df = load_peer_data(freq, "peer_avg")

# Global floor: earliest ARK ETF inception date (ARKK/ARKG: 2014-10-31)
ARK_START = pd.Timestamp("2014-10-31")
df = df[df["Date"] >= ARK_START]

if date_start is not None and date_end is not None:
    _start = max(pd.Timestamp(date_start), ARK_START)
    _end = pd.Timestamp(date_end)
    df = df[(df["Date"] >= _start) & (df["Date"] <= _end)]

if freq == "D":
    fc_raw, rc = "Fund_Flow", "Return"
else:
    fc_raw, rc = "Flow_Sum", "Return_Cum"

if flow_unit == "pct" and "Flow_Pct" in df.columns and df["Flow_Pct"].notna().any():
    fc = "Flow_Pct"
    flow_ylabel = "Fund Flow (% of AUM)"
else:
    fc = fc_raw
    flow_ylabel = "Fund Flow ($M)"

etfs_with_flows = df.groupby("ETF")[fc].apply(lambda x: x.notna().sum())
valid_etfs = etfs_with_flows[etfs_with_flows > 20].index.tolist()
df_valid = df[df["ETF"].isin(valid_etfs)].copy()

# ============================================================
# Drawdown Analysis
# ============================================================
st.markdown("""
We identify **non-overlapping drawdown episodes** (peak-to-trough declines ≥ 10%)
for each ETF using an iterative deepest-first algorithm on a cumulative-return price index.
For each episode, we measure cumulative net flows over the 1, 2, 3, and 6 months following
the trough, then regress those post-drawdown flows on the drawdown's depth and duration:
""")
st.latex(r"CumFlow_{i,[t,t+h]} = \alpha + \beta_1 \cdot DrawdownDepth_i + \beta_2 \cdot Duration_i + \varepsilon")
st.markdown("""
- **CumFlow**: sum of net flows from trough date *t* to *t + h* months.
- **DrawdownDepth**: peak-to-trough decline in %, negative (e.g., −30% means a 30% drop).
- **Duration**: number of trading days from peak to trough.
- **Interpretation**: β₁ < 0 would mean deeper drawdowns lead to *larger outflows* (panic selling);
  β₁ > 0 would suggest contrarian buying after steep declines.
""")

_time_key = f"{date_start}_{date_end}" if date_start else "all"


@st.cache_data(show_spinner="Computing drawdowns...")
def compute_drawdowns(_df, return_col, _time_key=None):
    return compute_etf_drawdowns(_df, return_col, min_depth_pct=10.0)


dd_all = compute_drawdowns(df_valid, rc, _time_key=_time_key)

if len(dd_all) > 0:
    # Chart 1: price index with drawdown shading for selected ETF
    etf_dd = dd_all[dd_all["ETF"] == selected_etf]
    etf_prices = df_valid[df_valid["ETF"] == selected_etf].copy().sort_values("Date")

    if len(etf_prices) > 10:
        price_idx = (1 + etf_prices.set_index("Date")[rc].dropna()).cumprod() * 100

        fig_dd_price = go.Figure()
        fig_dd_price.add_trace(go.Scatter(
            x=price_idx.index, y=price_idx.values,
            mode="lines", name="Price Index", line=dict(color="#1f77b4", width=1.5),
        ))
        for _, dd_row in etf_dd.iterrows():
            fig_dd_price.add_vrect(
                x0=dd_row["peak_date"], x1=dd_row["trough_date"],
                fillcolor="red", opacity=0.15, line_width=0,
                annotation_text=f"{dd_row['depth_pct']:.0f}%",
                annotation_position="top left",
                annotation_font_size=9,
            )
        fig_dd_price.update_layout(
            height=400, yaxis_title="Price Index (base=100)",
            title=f"{selected_etf}: Price Index with Drawdown Periods",
            margin=dict(l=60, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_dd_price, width="stretch")
        st.caption(
            "Price index constructed from cumulative returns (base = 100). Drawdown episodes identified "
            "using an iterative deepest-first algorithm: find the largest peak-to-trough decline (≥ 10%), "
            "remove that segment, repeat on remaining periods. Labels show depth (%)."
        )

    # Compute flow analysis
    dd_flow = drawdown_flow_analysis(df_valid, dd_all, fc)

    if len(dd_flow) > 0:
        # Chart 2: scatter — drawdown depth vs cumulative flow (1m)
        col_scat1, col_scat2 = st.columns(2)
        with col_scat1:
            if "CumFlow_1m" in dd_flow.columns:
                fig_scat = px.scatter(
                    dd_flow, x="depth_pct", y="CumFlow_1m",
                    color="ETF", hover_data=["trough_date", "duration_days"],
                    title="Drawdown Depth vs 1-Month Cumulative Flow",
                )
                fig_scat.update_layout(
                    height=400, xaxis_title="Drawdown Depth (%)",
                    yaxis_title=f"Cumulative Flow 1m ({flow_ylabel})",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat, width="stretch")
                st.caption(
                    "Each dot = one drawdown episode across all 38 ETFs. A negative slope would indicate deeper "
                    "drawdowns trigger larger outflows (panic selling); a positive slope suggests contrarian buying."
                )

        with col_scat2:
            if "CumFlow_3m" in dd_flow.columns:
                fig_scat3 = px.scatter(
                    dd_flow, x="depth_pct", y="CumFlow_3m",
                    color="ETF", hover_data=["trough_date", "duration_days"],
                    title="Drawdown Depth vs 3-Month Cumulative Flow",
                )
                fig_scat3.update_layout(
                    height=400, xaxis_title="Drawdown Depth (%)",
                    yaxis_title=f"Cumulative Flow 3m ({flow_ylabel})",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat3, width="stretch")
                st.caption(
                    "3-month forward window captures delayed investor reactions beyond the immediate post-trough period."
                )

        # Regression table
        dd_reg = drawdown_flow_regression(dd_flow)
        if len(dd_reg) > 0:
            st.subheader("Post-Drawdown Flow Regression")
            st.caption(
                "OLS regression of cumulative post-trough flows on drawdown characteristics, pooled across all ETFs. "
                "Each row = a different forward horizon (1m, 2m, 3m, 6m). "
                "$\\beta_{\\text{Depth}}$: change in cumulative flow per 1 pp deeper drawdown. "
                "$\\beta_{\\text{Duration}}$: change in cumulative flow per additional trading day of drawdown. "
                "Significant $p_{\\beta_{\\text{Depth}}} < 0.05$ confirms that drawdown severity predicts subsequent flow behavior."
            )
            st.dataframe(
                dd_reg.style.format({
                    "β_Depth": "{:.4f}", "β_Depth_p": "{:.4f}",
                    "β_Duration": "{:.4f}", "β_Duration_p": "{:.4f}",
                    "R²": "{:.4f}",
                }),
                hide_index=True, width="stretch",
            )
    else:
        st.info("Not enough post-drawdown flow data for analysis.")
else:
    st.info("No drawdowns ≥ 10% found in the selected time range.")
