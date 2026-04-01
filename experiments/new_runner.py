"""Structured paper runner — replaces the grid-search approach.

Produces Tables 1-5 and Figures 1-2 for the paper:
    Table 1: Summary Statistics
    Table 2: Sirri-Tufano piecewise linear replication (monthly)
    Table 3: Main panel specification with incremental controls (daily)
    Table 4: Sub-sample analysis (bull vs bear)
    Table 5: Robustness battery
    Figure 1: Local Projection impulse response
    Figure 2: Asymmetric LP (positive vs negative shocks)

Usage:
    python -m experiments.new_runner
    python -m experiments.new_runner --table 3
    python -m experiments.new_runner --figure 1
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES,
    aggregate_to_frequency, add_returns, load_all_etfs_with_peers,
    add_source_flag, add_market_benchmark, merge_aum,
    add_zscore_columns,
)
from noise_factors import apply_factor_C, apply_factor_D, apply_factor_E
from macro_events import add_event_dummies, get_event_ids
from summary_stats import panel_summary, summary_statistics, correlation_matrix
from sirri_tufano import compute_fractional_rank, sirri_tufano_regression, sirri_tufano_table
from local_projection import (
    local_projection, local_projection_asymmetric,
    local_projection_subsample, local_projection_cumulative,
)
from placebo import (
    placebo_test, leave_one_etf_out,
    subsample_comparison, fama_macbeth,
    breusch_pagan_test, white_test, driscoll_kraay_panel,
    panel_ols_twoway, rolling_panel_regression,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results_v2"


# ============================================================
# Helper: build non-overlapping cumulative return windows
# ============================================================

def build_non_overlapping_cumret(df: pd.DataFrame, return_col: str,
                                 windows: list[tuple[int, int]] = None
                                 ) -> pd.DataFrame:
    """Build non-overlapping cumulative return windows as regressors.

    windows: [(1,5), (6,20), (21,60)] means:
        CumRet_1_5  = sum of Return_{t-1} to Return_{t-5}
        CumRet_6_20 = sum of Return_{t-6} to Return_{t-20}
        CumRet_21_60 = sum of Return_{t-21} to Return_{t-60}
    """
    if windows is None:
        windows = [(1, 5), (6, 20), (21, 60)]

    pdf = df.copy()
    for start, end in windows:
        col_name = f"CumRet_{start}_{end}"
        pdf[col_name] = pdf.groupby("ETF")[return_col].transform(
            lambda x, s=start, e=end: sum(x.shift(k) for k in range(s, e + 1))
        )
    return pdf


# ============================================================
# Helper: prepare all control variables
# ============================================================

def prepare_controls(df: pd.DataFrame, freq: str = "D",
                     date_col: str = "Date") -> pd.DataFrame:
    """Add all control variable columns (VIX, calendar, peer flow, event dummies).

    No data is deleted — all controls are added as columns.
    """
    pdf = df.copy()

    # Factor C: VIX
    try:
        pdf = apply_factor_C(pdf, method="control", date_col=date_col)
    except Exception as e:
        logger.warning("Failed to add VIX control: %s", e)

    # Factor D: Calendar dummies
    pdf = apply_factor_D(pdf, date_col=date_col)

    # Factor E: Peer aggregate flow
    flow_col = "Fund_Flow" if freq == "D" else "Flow_Sum"
    if flow_col in pdf.columns:
        pdf = apply_factor_E(pdf, flow_col=flow_col, date_col=date_col)

    # Event dummies (NOT exclusion)
    all_events = get_event_ids()
    pdf = add_event_dummies(pdf, all_events, date_col=date_col)

    return pdf


# ============================================================
# Table 1: Summary Statistics
# ============================================================

def run_table_1(df_daily: pd.DataFrame) -> dict:
    """Generate Table 1: Summary Statistics."""
    logger.info("Running Table 1: Summary Statistics")

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()

    variables = ["Fund_Flow", "Return", "Close"]
    if "Flow_Pct" in ark.columns:
        variables.append("Flow_Pct")
    if "Excess_Return" in ark.columns:
        variables.append("Excess_Return")
    if "VIX_Close" in ark.columns:
        variables.append("VIX_Close")

    result = panel_summary(ark, "Fund_Flow", "Return",
                           extra_vars=[v for v in variables if v not in ["Fund_Flow", "Return"]])

    corr = correlation_matrix(ark, ["Fund_Flow", "Return", "Excess_Return"])

    return {"summary": result, "correlation": corr}


# ============================================================
# Table 2: Sirri-Tufano Replication
# ============================================================

def run_table_2(df_monthly: pd.DataFrame) -> dict:
    """Generate Table 2: Sirri-Tufano piecewise linear model."""
    logger.info("Running Table 2: Sirri-Tufano Replication")

    df = compute_fractional_rank(df_monthly, return_col="Return_Cum")

    controls_seq = [
        ("(1) Base", []),
        ("(2) + VIX", ["VIX_Close"]),
        ("(3) + Calendar", ["VIX_Close", "month_end", "quarter_end", "january"]),
        ("(4) + Peer", ["VIX_Close", "month_end", "quarter_end", "january",
                        "Peer_Agg_Flow"]),
    ]

    # Filter to only include controls that exist
    filtered_seq = []
    for name, ctrls in controls_seq:
        valid = [c for c in ctrls if c in df.columns]
        filtered_seq.append((name, valid))

    table = sirri_tufano_table(df, flow_col="Flow_Pct",
                                controls_sequence=filtered_seq)

    # Also get the base regression result for convexity test
    base = sirri_tufano_regression(df, flow_col="Flow_Pct")

    return {"table": table, "base_regression": base}


# ============================================================
# Table 3: Main Panel Specification
# ============================================================

def run_table_3(df_daily: pd.DataFrame) -> dict:
    """Generate Table 3: Main daily panel specification with incremental controls.

    Flow_it = α_i + β₁ CumRet_1_5 + β₂ CumRet_6_20 + β₃ CumRet_21_60
              + γ Controls + ε
    """
    logger.info("Running Table 3: Main Panel Specification")

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]

    # Build non-overlapping cumulative returns
    ark = build_non_overlapping_cumret(ark, "Return")

    base_x = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]

    # Define incremental control sets
    vix_cols = [c for c in ["VIX_Close"] if c in ark.columns]
    cal_cols = [c for c in ["month_end", "quarter_end", "january"] if c in ark.columns]
    peer_cols = [c for c in ["Peer_Agg_Flow"] if c in ark.columns]
    event_cols = [c for c in ark.columns if c.startswith("event_")]

    specs = [
        ("(1) Base", base_x),
        ("(2) + VIX", base_x + vix_cols),
        ("(3) + Calendar", base_x + vix_cols + cal_cols),
        ("(4) + Peer Flow", base_x + vix_cols + cal_cols + peer_cols),
        ("(5) + Events", base_x + vix_cols + cal_cols + peer_cols + event_cols),
    ]

    from placebo import _panel_ols_demeaned, panel_ols_twoway

    results = {}
    for name, x_cols in specs:
        valid_cols = [c for c in x_cols if c in ark.columns]
        res = _panel_ols_demeaned(ark, "Fund_Flow", valid_cols)
        if res:
            results[name] = res
            logger.info("  %s: R²=%.4f, N=%d", name, res["r_squared"], res["n_obs"])

    # (6) Entity + Time FE (two-way fixed effects)
    full_x = [c for c in base_x + vix_cols + cal_cols + peer_cols + event_cols if c in ark.columns]
    res_twfe = panel_ols_twoway(ark, "Fund_Flow", full_x,
                                 entity_effects=True, time_effects=True)
    if res_twfe:
        results["(6) Two-way FE"] = res_twfe
        logger.info("  (6) Two-way FE: R²=%.4f, N=%d",
                     res_twfe["r_squared_within"], res_twfe["n_obs"])

    # (7) Two-way clustered SE (entity + date)
    res_2cl = panel_ols_twoway(ark, "Fund_Flow", full_x,
                                cluster_entity=True, cluster_time=True)
    if res_2cl:
        results["(7) Two-way Cluster"] = res_2cl
        logger.info("  (7) Two-way Cluster: R²=%.4f, N=%d",
                     res_2cl["r_squared_within"], res_2cl["n_obs"])

    return results


# ============================================================
# Figure 1: Local Projection Impulse Response
# ============================================================

def run_figure_1(df_daily: pd.DataFrame, max_horizon: int = 40) -> pd.DataFrame:
    """Generate Figure 1 data: LP impulse response."""
    logger.info("Running Figure 1: LP Impulse Response (h=0..%d)", max_horizon)

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()

    controls = [c for c in ["VIX_Close", "month_end", "quarter_end", "january",
                             "Peer_Agg_Flow"] if c in ark.columns]

    lp = local_projection(ark, "Fund_Flow", "Return",
                          max_horizon=max_horizon, controls=controls)
    return lp


# ============================================================
# Figure 2: Asymmetric LP
# ============================================================

def run_figure_2(df_daily: pd.DataFrame, max_horizon: int = 40) -> pd.DataFrame:
    """Generate Figure 2 data: Asymmetric LP."""
    logger.info("Running Figure 2: Asymmetric LP (h=0..%d)", max_horizon)

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()

    controls = [c for c in ["VIX_Close", "month_end", "quarter_end", "january",
                             "Peer_Agg_Flow"] if c in ark.columns]

    lp_asym = local_projection_asymmetric(ark, "Fund_Flow", "Return",
                                           max_horizon=max_horizon,
                                           controls=controls)
    return lp_asym


# ============================================================
# Table 4: Sub-sample (Bull vs Bear)
# ============================================================

def run_table_4(df_daily: pd.DataFrame) -> dict:
    """Generate Table 4: Sub-sample analysis."""
    logger.info("Running Table 4: Sub-sample Analysis")

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")

    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]
    # Add available controls
    for c in ["VIX_Close", "month_end", "quarter_end", "january", "Peer_Agg_Flow"]:
        if c in ark.columns:
            x_cols.append(c)

    periods = {
        "Full Sample": ("2014-01-01", "2025-12-31"),
        "Pre-COVID": ("2014-01-01", "2020-01-31"),
        "Bull (2020-2021)": ("2020-01-01", "2021-12-31"),
        "Bear (2022-2024)": ("2022-01-01", "2024-12-31"),
    }

    sub_results = subsample_comparison(ark, "Fund_Flow", "Return", x_cols, periods)

    # Also LP sub-sample comparison
    lp_sub = local_projection_subsample(
        ark, "Fund_Flow", "Return",
        periods={"bull": ("2020-01-01", "2021-12-31"),
                 "bear": ("2022-01-01", "2024-12-31")},
        max_horizon=30,
    )

    return {"regression": sub_results, "lp_subsample": lp_sub}


# ============================================================
# Table 5: Robustness Battery
# ============================================================

def run_table_5(df_daily: pd.DataFrame, df_weekly: pd.DataFrame = None,
                df_monthly: pd.DataFrame = None) -> dict:
    """Generate Table 5: Robustness checks."""
    logger.info("Running Table 5: Robustness Battery")

    ark = df_daily[df_daily["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")

    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]
    results = {}

    # (a) Placebo test
    logger.info("  5a: Placebo test")
    results["placebo"] = placebo_test(ark, "Fund_Flow", "Return",
                                       lag_windows=[(1, 5), (6, 20), (21, 60)],
                                       lead_windows=[(1, 5), (6, 20), (21, 60)])

    # (b) Leave-one-ETF-out
    logger.info("  5b: Leave-one-ETF-out")
    results["leave_one_out"] = leave_one_etf_out(ark, "Fund_Flow", "Return",
                                                   x_cols)

    # (c) Fama-MacBeth
    logger.info("  5c: Fama-MacBeth")
    results["fama_macbeth"] = fama_macbeth(ark, "Fund_Flow", x_cols)

    # (d) Alternative flow measure (% AUM)
    if "Flow_Pct" in ark.columns:
        logger.info("  5d: Alternative flow measure (pct AUM)")
        ark_pct = ark[np.isfinite(ark["Flow_Pct"]) & (ark["Flow_Pct"].abs() < 100)]
        from placebo import _panel_ols_demeaned
        results["flow_pct"] = _panel_ols_demeaned(
            build_non_overlapping_cumret(ark_pct, "Return"),
            "Flow_Pct", x_cols)

    return results


# ============================================================
# Economic Significance
# ============================================================

def run_table_5e(df: pd.DataFrame) -> dict | None:
    """Table 5e: Driscoll-Kraay standard errors for main specification."""
    logger.info("Running Table 5e (Driscoll-Kraay SE)")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")

    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]
    return driscoll_kraay_panel(ark, "Fund_Flow", x_cols)


def run_table_5f(df: pd.DataFrame) -> dict:
    """Table 5f: Heteroscedasticity diagnostics for main specification."""
    logger.info("Running Table 5f (heteroscedasticity tests)")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")

    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]
    bp = breusch_pagan_test(ark, "Fund_Flow", x_cols)
    w = white_test(ark, "Fund_Flow", x_cols)

    return {
        "breusch_pagan": bp,
        "white": w,
    }


def run_table_6(df: pd.DataFrame) -> pd.DataFrame:
    """Table 6: SE comparison — same spec, different SE methods."""
    logger.info("Running Table 6 (SE comparison)")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")
    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]

    se_methods = [
        ("Entity Cluster", dict(cluster_entity=True, cluster_time=False, cov_type="clustered")),
        ("Two-way Cluster", dict(cluster_entity=True, cluster_time=True, cov_type="clustered")),
        ("Driscoll-Kraay", dict(cov_type="kernel")),
    ]

    rows = []
    for method_name, kwargs in se_methods:
        res = panel_ols_twoway(ark, "Fund_Flow", x_cols, **kwargs)
        if res is None:
            continue
        for _, cr in res["coefficients"].iterrows():
            rows.append({
                "SE_Method": method_name,
                "Variable": cr["Variable"],
                "Coefficient": cr["Coefficient"],
                "Std_Error": cr["Std_Error"],
                "t_stat": cr["t_stat"],
                "p_value": cr["p_value"],
            })
    return pd.DataFrame(rows)


def run_table_7(df: pd.DataFrame) -> dict:
    """Table 7: ARKK heterogeneity — split sample + interaction."""
    logger.info("Running Table 7 (ARKK heterogeneity)")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")
    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]

    from placebo import _panel_ols_demeaned

    # (A) ARKK only — use HC1 since only 1 entity (can't cluster)
    arkk_only = ark[ark["ETF"] == "ARKK"][["Fund_Flow"] + x_cols].dropna()
    if len(arkk_only) > 30:
        y = arkk_only["Fund_Flow"] - arkk_only["Fund_Flow"].mean()
        X = arkk_only[x_cols].copy()
        for c in x_cols:
            X[c] = X[c] - X[c].mean()
        X = sm.add_constant(X)
        m = sm.OLS(y, X).fit(cov_type="HC1")
        coef_df = pd.DataFrame({
            "Variable": m.params.index, "Coefficient": m.params.values,
            "Std_Error": m.bse.values, "t_stat": m.tvalues.values,
            "p_value": m.pvalues.values,
        })
        coef_df = coef_df[coef_df["Variable"] != "const"].reset_index(drop=True)
        res_arkk = {"coefficients": coef_df, "r_squared": m.rsquared, "n_obs": int(m.nobs), "n_entities": 1}
    else:
        res_arkk = None

    # (B) Excluding ARKK
    no_arkk = ark[ark["ETF"] != "ARKK"].dropna(subset=["Fund_Flow"] + x_cols)
    try:
        res_no_arkk = _panel_ols_demeaned(no_arkk, "Fund_Flow", x_cols)
    except Exception as e:
        logger.warning("Table 7 excluding ARKK failed: %s", e)
        res_no_arkk = None

    # (C) Interaction: ARKK_dummy × CumRet (use linearmodels to avoid demeaning issues)
    ark_int = ark[["ETF", "Date", "Fund_Flow"] + x_cols].dropna().copy()
    ark_int["ARKK_d"] = (ark_int["ETF"] == "ARKK").astype(float)
    interact_cols = list(x_cols)
    for col in x_cols:
        icol = f"{col}_x_ARKK"
        ark_int[icol] = ark_int[col] * ark_int["ARKK_d"]
        interact_cols.append(icol)
    # Use linearmodels directly for interaction (avoids demeaning issues with dummy × continuous)
    try:
        from linearmodels.panel import PanelOLS as _PanelOLS
        _sub = ark_int.set_index(["ETF", "Date"])
        _m = _PanelOLS(_sub["Fund_Flow"], _sub[interact_cols],
                        entity_effects=True, drop_absorbed=True)
        _r = _m.fit(cov_type="clustered", cluster_entity=True)
        res_interact = {
            "coefficients": pd.DataFrame({
                "Variable": _r.params.index, "Coefficient": _r.params.values,
                "Std_Error": _r.std_errors.values, "t_stat": _r.tstats.values,
                "p_value": _r.pvalues.values}).reset_index(drop=True),
            "r_squared_within": _r.rsquared_within,
            "n_obs": int(_r.nobs), "n_entities": _r.entity_info.total,
        }
    except Exception as e:
        logger.warning("Table 7 interaction failed: %s", e)
        res_interact = None

    return {
        "arkk_only": res_arkk,
        "excluding_arkk": res_no_arkk,
        "interaction": res_interact,
    }


def run_table_8(df: pd.DataFrame) -> pd.DataFrame:
    """Table 8: Granger causality test — Return → Flow vs Flow → Return."""
    logger.info("Running Table 8 (Granger causality)")

    from analysis import granger_causality_test

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    fc = "Fund_Flow" if "Fund_Flow" in ark.columns else "Flow_Sum"
    rc = "Return" if "Return" in ark.columns else "Return_Cum"

    rows = []
    for etf in ETF_NAMES:
        etf_df = ark[ark["ETF"] == etf]
        if len(etf_df) < 50:
            continue
        gc = granger_causality_test(etf_df, fc, rc, max_lag=5)
        if gc is not None and not gc.empty:
            for direction in gc["direction"].unique():
                dir_df = gc[gc["direction"] == direction]
                best = dir_df.loc[dir_df["p_value"].idxmin()]
                rows.append({
                    "ETF": etf,
                    "Direction": direction,
                    "Best_Lag": int(best["lag"]),
                    "F_stat": best["F_statistic"],
                    "p_value": best["p_value"],
                })
    return pd.DataFrame(rows)


def run_figure_4(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 4 data: Rolling-window panel regression coefficients (2-year window)."""
    logger.info("Running Figure 4 (rolling coefficients)")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")
    x_cols = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]

    return rolling_panel_regression(ark, "Fund_Flow", x_cols, window_days=504)


def run_economic_significance(df: pd.DataFrame,
                               main_results: dict) -> dict:
    """Compute economic significance of main specification coefficients."""
    logger.info("Computing economic significance")

    ark = df[df["ETF"].isin(ETF_NAMES)].copy()
    ark = ark[np.isfinite(ark["Fund_Flow"])]
    ark = build_non_overlapping_cumret(ark, "Return")

    # Standard deviations
    sd_flow = ark["Fund_Flow"].std()
    sd_ret_1_5 = ark["CumRet_1_5"].dropna().std()
    sd_ret_6_20 = ark["CumRet_6_20"].dropna().std()
    sd_ret_21_60 = ark["CumRet_21_60"].dropna().std()
    mean_aum = ark["AUM"].dropna().mean() if "AUM" in ark.columns else np.nan

    # Get coefficients from main spec (last column = full model)
    last_spec = list(main_results.values())[-1] if main_results else None
    if not last_spec:
        return {}

    econ = {"sd_flow": sd_flow, "mean_aum_millions": mean_aum}

    for _, row in last_spec["coefficients"].iterrows():
        var = row["Variable"]
        coef = row["Coefficient"]
        if var == "CumRet_1_5":
            econ["CumRet_1_5_1sd_effect"] = coef * sd_ret_1_5
            econ["CumRet_1_5_std_beta"] = coef * sd_ret_1_5 / sd_flow
        elif var == "CumRet_6_20":
            econ["CumRet_6_20_1sd_effect"] = coef * sd_ret_6_20
            econ["CumRet_6_20_std_beta"] = coef * sd_ret_6_20 / sd_flow
        elif var == "CumRet_21_60":
            econ["CumRet_21_60_1sd_effect"] = coef * sd_ret_21_60
            econ["CumRet_21_60_std_beta"] = coef * sd_ret_21_60 / sd_flow

    return econ


# ============================================================
# Figure ST1: Sirri-Tufano scatter data
# ============================================================

def run_figure_st1(df: pd.DataFrame) -> pd.DataFrame:
    """Generate data for S&T-style performance rank vs flow scatter plot.

    Returns DataFrame with columns: ETF, Date, RANK, Flow_Pct, Trailing_Vol.
    """
    logger.info("Running Figure ST1 (S&T scatter data)")

    out = df.copy()

    # Identify return and flow columns (names vary by frequency)
    ret_col = "Return_Cum" if "Return_Cum" in out.columns else "Return"
    flow_col = "Flow_Pct" if "Flow_Pct" in out.columns else "Fund_Flow"

    # Compute fractional rank within each period
    out["RANK"] = out.groupby("Date")[ret_col].rank(pct=True)

    # Compute Flow as % of AUM if not already available
    if flow_col != "Flow_Pct" and "AUM" in out.columns:
        out["Flow_Pct_ST"] = out[flow_col] / out["AUM"].replace(0, np.nan) * 100
    else:
        out["Flow_Pct_ST"] = out[flow_col]

    # Compute trailing 12-month volatility (std of monthly returns)
    out["Trailing_Vol"] = out.groupby("ETF")[ret_col].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )

    result = out[["ETF", "Date", "RANK", "Flow_Pct_ST", "Trailing_Vol"]].dropna()
    result = result.rename(columns={"Flow_Pct_ST": "Flow_Pct"})
    return result


# ============================================================
# Main orchestrator
# ============================================================

def run_all(output_dir: str | None = None) -> dict:
    """Run the complete structured analysis pipeline."""
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()

    # Load data at all frequencies
    logger.info("Loading daily data...")
    df_daily = get_prepared_data_with_peers(freq="D", benchmark="SPY")
    df_daily = prepare_controls(df_daily, freq="D")

    logger.info("Loading monthly data...")
    df_monthly = get_prepared_data_with_peers(freq="ME", benchmark="SPY")
    df_monthly = prepare_controls(df_monthly, freq="ME")

    # Table 1
    t1 = run_table_1(df_daily)
    all_results["table_1"] = t1
    t1["summary"]["overall"].to_csv(out / "table_1_summary.csv", index=False)
    t1["summary"]["between_within"].to_csv(out / "table_1_bw.csv", index=False)
    logger.info("Table 1 saved")

    # Table 2
    t2 = run_table_2(df_monthly)
    all_results["table_2"] = t2
    if not t2["table"].empty:
        t2["table"].to_csv(out / "table_2_sirri_tufano.csv", index=False)
    logger.info("Table 2 saved")

    # Table 3
    t3 = run_table_3(df_daily)
    all_results["table_3"] = t3
    # Save coefficients from each spec
    rows = []
    for spec_name, res in t3.items():
        for _, cr in res["coefficients"].iterrows():
            rows.append({"spec": spec_name, **cr.to_dict()})
        r2 = res.get("r_squared", res.get("r_squared_within", np.nan))
        rows.append({"spec": spec_name, "Variable": "R²",
                      "Coefficient": r2})
        rows.append({"spec": spec_name, "Variable": "N",
                      "Coefficient": res["n_obs"]})
    pd.DataFrame(rows).to_csv(out / "table_3_main_panel.csv", index=False)
    logger.info("Table 3 saved")

    # Figure 1
    f1 = run_figure_1(df_daily)
    all_results["figure_1"] = f1
    f1.to_csv(out / "figure_1_lp.csv", index=False)
    logger.info("Figure 1 data saved")

    # Figure 2
    f2 = run_figure_2(df_daily)
    all_results["figure_2"] = f2
    f2.to_csv(out / "figure_2_asymmetric_lp.csv", index=False)
    logger.info("Figure 2 data saved")

    # Table 4
    t4 = run_table_4(df_daily)
    all_results["table_4"] = t4
    if not t4["regression"].empty:
        t4["regression"].to_csv(out / "table_4_subsample.csv", index=False)
    for period_name, lp_df in t4.get("lp_subsample", {}).items():
        lp_df.to_csv(out / f"table_4_lp_{period_name}.csv", index=False)
    logger.info("Table 4 saved")

    # Table 5
    t5 = run_table_5(df_daily)
    all_results["table_5"] = t5
    # Save individual robustness results
    if t5.get("placebo") and t5["placebo"].get("real"):
        t5["placebo"]["real"]["coefficients"].to_csv(
            out / "table_5a_placebo_real.csv", index=False)
    if t5.get("placebo") and t5["placebo"].get("placebo"):
        t5["placebo"]["placebo"]["coefficients"].to_csv(
            out / "table_5a_placebo_fake.csv", index=False)
    if t5.get("leave_one_out") is not None and not t5["leave_one_out"].empty:
        t5["leave_one_out"].to_csv(out / "table_5b_leave_one_out.csv", index=False)
    if t5.get("fama_macbeth"):
        t5["fama_macbeth"]["coefficients"].to_csv(
            out / "table_5c_fama_macbeth.csv", index=False)
    if t5.get("flow_pct"):
        t5["flow_pct"]["coefficients"].to_csv(
            out / "table_5d_flow_pct.csv", index=False)
    logger.info("Table 5 saved")

    # Table 5e: Driscoll-Kraay SE
    t5e = run_table_5e(df_daily)
    all_results["table_5e"] = t5e
    if t5e:
        t5e["coefficients"].to_csv(out / "table_5e_driscoll_kraay.csv", index=False)
    logger.info("Table 5e saved")

    # Table 5f: Heteroscedasticity tests
    t5f = run_table_5f(df_daily)
    all_results["table_5f"] = t5f
    pd.DataFrame([
        {"Test": "Breusch-Pagan", **t5f["breusch_pagan"]},
        {"Test": "White", **t5f["white"]},
    ]).to_csv(out / "table_5f_diagnostics.csv", index=False)
    logger.info("Table 5f saved")

    # Economic significance
    econ = run_economic_significance(df_daily, t3)
    all_results["economic_significance"] = econ
    pd.DataFrame([econ]).to_csv(out / "economic_significance.csv", index=False)

    # Figure ST1: Sirri-Tufano scatter data
    fst1 = run_figure_st1(df_monthly)
    all_results["figure_st1"] = fst1
    fst1.to_csv(out / "figure_st1_scatter.csv", index=False)
    logger.info("Figure ST1 data saved")

    # Table 6: SE comparison
    t6 = run_table_6(df_daily)
    all_results["table_6"] = t6
    if not t6.empty:
        t6.to_csv(out / "table_6_se_comparison.csv", index=False)
    logger.info("Table 6 saved")

    # Table 7: ARKK heterogeneity
    t7 = run_table_7(df_daily)
    all_results["table_7"] = t7
    for key in ["arkk_only", "excluding_arkk", "interaction"]:
        if t7.get(key) and "coefficients" in t7[key]:
            t7[key]["coefficients"].to_csv(out / f"table_7_{key}.csv", index=False)
    logger.info("Table 7 saved")

    # Table 8: Granger causality
    t8 = run_table_8(df_daily)
    all_results["table_8"] = t8
    if not t8.empty:
        t8.to_csv(out / "table_8_granger.csv", index=False)
    logger.info("Table 8 saved")

    # Figure 4: Rolling coefficients
    f4 = run_figure_4(df_daily)
    all_results["figure_4"] = f4
    if not f4.empty:
        f4.to_csv(out / "figure_4_rolling.csv", index=False)
    logger.info("Figure 4 data saved")

    elapsed = time.time() - t0
    logger.info("All done in %.1fs. Results saved to %s", elapsed, out)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Structured paper analysis runner")
    parser.add_argument("--table", type=int, help="Run specific table (1-5)")
    parser.add_argument("--figure", type=int, help="Run specific figure (1-2)")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.all or (args.table is None and args.figure is None):
        run_all()
    else:
        # Load data
        logger.info("Loading data...")
        df_d = get_prepared_data_with_peers(freq="D", benchmark="SPY")
        df_d = prepare_controls(df_d, freq="D")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if args.table == 1:
            r = run_table_1(df_d)
            print(r["summary"]["overall"].to_string(index=False))
        elif args.table == 2:
            df_m = get_prepared_data_with_peers(freq="ME", benchmark="SPY")
            df_m = prepare_controls(df_m, freq="ME")
            r = run_table_2(df_m)
            print(r["table"].to_string(index=False))
        elif args.table == 3:
            r = run_table_3(df_d)
            for name, res in r.items():
                print(f"\n{name}:")
                print(res["coefficients"].to_string(index=False))
                print(f"R²={res['r_squared']:.4f}, N={res['n_obs']}")
        elif args.table == 4:
            r = run_table_4(df_d)
            print(r["regression"].to_string(index=False))
        elif args.table == 5:
            r = run_table_5(df_d)
            for k, v in r.items():
                print(f"\n--- {k} ---")
                if isinstance(v, dict) and "coefficients" in v:
                    print(v["coefficients"].to_string(index=False))
                elif isinstance(v, pd.DataFrame):
                    print(v.to_string(index=False))
        elif args.figure == 1:
            r = run_figure_1(df_d)
            print(r.to_string(index=False))
        elif args.figure == 2:
            r = run_figure_2(df_d)
            print(r.to_string(index=False))


if __name__ == "__main__":
    main()
