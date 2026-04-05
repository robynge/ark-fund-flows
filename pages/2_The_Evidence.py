"""Page 2: The Evidence — S&T scatter, piecewise + quadratic regression, panel spec."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (sidebar_freq, load_data_with_controls, get_cols,
                      get_cumret_windows, cumret_cols, build_cumret,
                      stars, ETF_NAMES, FREQ_LABELS)

st.set_page_config(page_title="The Evidence", layout="wide")
st.title("The Evidence: Do Investors Chase?")
st.markdown("""
We present evidence at three levels: **(1)** visual, **(2)** the classic S&T
piecewise regression, and **(3)** our panel specification with cumulative return windows.

> **Reading the tables**: **\\*\\*\\*** = p < 0.01 (very strong), **\\*\\*** = p < 0.05,
> **\\*** = p < 0.10, no stars = not significant. More stars = more confident the result is real.
""")

freq = sidebar_freq(key="evidence_freq")
fc, rc = get_cols(freq)
windows = get_cumret_windows(freq)
period = FREQ_LABELS[freq]

df = load_data_with_controls(freq)
df = df[df["Date"] >= pd.Timestamp("2014-10-31")]


# ============================================================
# 1. S&T Scatter
# ============================================================
st.header("1. Do Top Performers Attract More Money?")
st.markdown(f"""
Each {period[:-1]}, we rank all ETFs by return and plot rank vs. fund flow.
Rank 0 = worst, 1 = best. The red line shows 20-bin averages.
An **upward curve** at the right = top performers attract disproportionate capital.
""")

from sirri_tufano import compute_fractional_rank

@st.cache_data(show_spinner="Computing ranks...")
def compute_scatter(freq):
    _df = load_data_with_controls(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    ranked = compute_fractional_rank(_df, return_col=_rc)
    if "Flow_Pct" not in ranked.columns:
        ranked["Flow_Pct"] = ranked[_fc]
    out = ranked[["ETF", "Date", "RANK", "Flow_Pct"]].dropna()
    p1, p99 = out["Flow_Pct"].quantile(0.01), out["Flow_Pct"].quantile(0.99)
    return out[(out["Flow_Pct"] >= p1) & (out["Flow_Pct"] <= p99)].copy()

scatter = compute_scatter(freq)
if not scatter.empty:
    scatter["rank_bin"] = pd.cut(scatter["RANK"], bins=20, labels=False) + 1
    bin_means = scatter.groupby("rank_bin").agg(
        rank_mid=("RANK", "mean"), flow_mean=("Flow_Pct", "mean")).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=scatter["RANK"], y=scatter["Flow_Pct"],
        mode="markers", marker=dict(size=4, opacity=0.35, color="#1f77b4"),
        name="Individual obs.",
        hovertemplate="Rank: %{x:.2f}<br>Flow: %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=bin_means["rank_mid"], y=bin_means["flow_mean"],
        mode="lines+markers", line=dict(color="#d62728", width=3),
        marker=dict(size=8), name="20-bin average",
        hovertemplate="Rank: %{x:.2f}<br>Avg flow: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=450,
        xaxis_title="Performance Rank (0 = worst, 1 = best)",
        yaxis_title="Fund Flow (% of AUM)",
        title=f"Performance Rank vs. Flow Growth ({FREQ_LABELS[freq].title()})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Line-only version (for presentations)"):
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=bin_means["rank_mid"], y=bin_means["flow_mean"],
            mode="lines+markers", line=dict(color="#d62728", width=3),
            marker=dict(size=8),
            hovertemplate="Rank: %{x:.2f}<br>Avg flow: %{y:.2f}%<extra></extra>"))
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_line.update_layout(height=400,
            xaxis_title="Performance Rank (0 = worst, 1 = best)",
            yaxis_title="Fund Flow (% of AUM)")
        st.plotly_chart(fig_line, width="stretch")


# ============================================================
# 2. S&T Piecewise Linear Regression
# ============================================================
st.header("2. S&T Piecewise Linear Regression")
st.markdown(r"""
$$
\text{Flow}_{i,t} = \alpha_i + \beta_L \cdot \text{LOWPERF} + \beta_M \cdot \text{MIDPERF}
+ \beta_H \cdot \text{HIGHPERF} + \gamma' Z + \varepsilon
$$

LOWPERF = bottom 20%, MIDPERF = middle 60%, HIGHPERF = top 20% of performance rank.
""")

from sirri_tufano import sirri_tufano_table

@st.cache_data(show_spinner="Running S&T regression...")
def compute_st_table(freq):
    _df = load_data_with_controls(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    ranked = compute_fractional_rank(_df, return_col=_rc)
    flow_col = "Flow_Pct" if "Flow_Pct" in ranked.columns else _fc
    controls_seq = [
        ("(1) Base", []),
        ("(2) + VIX", [c for c in ["VIX_Change", "VIX_Lag_Change"] if c in ranked.columns]),
        ("(3) + Peer", [c for c in ["VIX_Change", "VIX_Lag_Change", "Peer_Agg_Flow"]
                        if c in ranked.columns]),
    ]
    return sirri_tufano_table(ranked, flow_col=flow_col, controls_sequence=controls_seq)

t2 = compute_st_table(freq)
if not t2.empty:
    st.dataframe(t2, width="stretch", hide_index=True)
    st.success("**Result**: HIGHPERF >> MIDPERF confirms the **convex flow-performance relationship**.")
    st.warning("""
    **Note on LOWPERF**: A positive LOWPERF coefficient means even poorly performing
    ETFs receive some inflows — possibly due to marketing offsetting bad performance.
    """)


# ============================================================
# 3. Panel Specification
# ============================================================
st.header("3. Panel Specification with Cumulative Return Windows")

win_labels = " + ".join([f"CumRet_{s}_{e}" for s, e in windows])
st.markdown(rf"""
$$
\text{{Flow}}_{{i,t}} = \alpha_i + \beta_1 \cdot \text{{CumRet}}_{{{windows[0][0]}\text{{-}}{windows[0][1]}}}
+ \beta_2 \cdot \text{{CumRet}}_{{{windows[1][0]}\text{{-}}{windows[1][1]}}}
+ \beta_3 \cdot \text{{CumRet}}_{{{windows[2][0]}\text{{-}}{windows[2][1]}}}
+ \gamma' Z + \varepsilon
$$

Windows are in **{period}**. Entity-demeaned OLS, clustered SE by ETF.
""")

from placebo import _panel_ols_demeaned, panel_ols_twoway

@st.cache_data(show_spinner="Running panel regressions...")
def compute_table_3(freq):
    _df = load_data_with_controls(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    _windows = get_cumret_windows(freq)

    ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    ark = build_cumret(ark, _rc, _windows)

    x_base = cumret_cols(_windows)
    vix = [c for c in ["VIX_Change", "VIX_Lag_Change"] if c in ark.columns]
    cal = [c for c in ["month_end", "quarter_end", "january"] if c in ark.columns]
    peer = [c for c in ["Peer_Agg_Flow"] if c in ark.columns]
    events = [c for c in ark.columns if c.startswith("event_")]

    specs = [
        ("(1) Base", x_base),
        ("(2) + VIX", x_base + vix),
        ("(3) + Calendar", x_base + vix + cal),
        ("(4) + Peer Flow", x_base + vix + cal + peer),
        ("(5) + Events", x_base + vix + cal + peer + events),
    ]

    results = {}
    for name, x_cols in specs:
        valid = [c for c in x_cols if c in ark.columns]
        res = _panel_ols_demeaned(ark, _fc, valid)
        if res:
            results[name] = res
    return results

t3 = compute_table_3(freq)
if t3:
    rows_display = {}
    for spec_name, res in t3.items():
        for _, cr in res["coefficients"].iterrows():
            var = cr["Variable"]
            rows_display.setdefault(var, {})[spec_name] = (
                f"{cr['Coefficient']:.2f}{stars(cr['p_value'])}")
        r2 = res.get("r_squared", res.get("r_squared_within", np.nan))
        rows_display.setdefault("R²", {})[spec_name] = f"{r2:.4f}"
        rows_display.setdefault("N", {})[spec_name] = f"{res['n_obs']:,}"

    display = pd.DataFrame.from_dict(rows_display, orient="index")
    st.dataframe(display, width="stretch")

    st.success(f"""
    **Key finding**: The shortest window is not significant — investors don't react
    to noise. The medium and long windows are highly significant. Performance chasing
    operates on a **multi-{period[:-1]}** horizon.
    """)

    # Peer flow highlight
    if any("Peer_Agg_Flow" in (res.get("coefficients", pd.DataFrame()).get("Variable", pd.Series())).values
           for res in t3.values()):
        st.info("""
        **Peer Aggregate Flow**: When money flows into the ETF's peer group, the ETF
        itself also receives inflows — investors allocate at the **category level**.
        """)


# ============================================================
# 4. Per-ETF Individual Regressions
# ============================================================
with st.expander("Per-ETF Individual Regressions"):
    @st.cache_data(show_spinner="Running per-ETF regressions...")
    def compute_per_etf(freq):
        _df = load_data_with_controls(freq)
        _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
        _fc, _rc = get_cols(freq)
        _windows = get_cumret_windows(freq)
        ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
        ark = ark[np.isfinite(ark[_fc])]
        ark = build_cumret(ark, _rc, _windows)
        x_cols = cumret_cols(_windows)

        rows = []
        for etf in ETF_NAMES:
            etf_df = ark[ark["ETF"] == etf][[_fc] + x_cols].dropna()
            if len(etf_df) < 30:
                continue
            y = etf_df[_fc]
            X = sm.add_constant(etf_df[x_cols])
            m = sm.OLS(y, X).fit(cov_type="HC1")
            for var in x_cols:
                rows.append({"ETF": etf, "Variable": var,
                             "Coefficient": m.params[var], "Std_Error": m.bse[var],
                             "p_value": m.pvalues[var]})
            rows.append({"ETF": etf, "Variable": "R²",
                         "Coefficient": m.rsquared, "Std_Error": np.nan, "p_value": np.nan})
        return pd.DataFrame(rows)

    per_etf = compute_per_etf(freq)
    if not per_etf.empty:
        for etf in per_etf["ETF"].unique():
            edf = per_etf[per_etf["ETF"] == etf]
            r2 = edf[edf["Variable"] == "R²"]["Coefficient"].iloc[0] if "R²" in edf["Variable"].values else 0
            st.markdown(f"**{etf}** (R² = {r2:.4f})")
            st.dataframe(edf[edf["Variable"] != "R²"][["Variable", "Coefficient", "Std_Error", "p_value"]].style.format(
                {"Coefficient": "{:.2f}", "Std_Error": "{:.2f}", "p_value": "{:.4f}"}),
                width="stretch", hide_index=True)


# ============================================================
# 5. Economic Significance
# ============================================================
st.header("4. How Big Is the Effect?")

@st.cache_data(show_spinner="Computing economic significance...")
def compute_econ_sig(freq):
    _df = load_data_with_controls(freq)
    _df = _df[_df["Date"] >= pd.Timestamp("2014-10-31")]
    _fc, _rc = get_cols(freq)
    _windows = get_cumret_windows(freq)
    ark = _df[_df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark[_fc])]
    ark = build_cumret(ark, _rc, _windows)

    sd_flow = ark[_fc].std()
    x_cols = cumret_cols(_windows)
    result = {"sd_flow": sd_flow}
    for col in x_cols:
        sd = ark[col].dropna().std()
        # Get coefficient from base spec
        res = _panel_ols_demeaned(ark, _fc, x_cols)
        if res:
            coef_row = res["coefficients"][res["coefficients"]["Variable"] == col]
            if not coef_row.empty:
                beta = coef_row.iloc[0]["Coefficient"]
                result[f"{col}_1sd_effect"] = beta * sd
    return result

econ = compute_econ_sig(freq)
if econ:
    cols = st.columns(len(windows))
    for i, (s, e) in enumerate(windows):
        col_name = f"CumRet_{s}_{e}"
        effect = econ.get(f"{col_name}_1sd_effect", 0)
        cols[i].metric(f"CumRet {s}-{e} {period}", f"${effect:.2f}M per 1-SD")

st.info(f"**Next** → *The Dynamics*: How does the effect unfold over {period}?")
