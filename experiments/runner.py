"""Offline experiment runner.

Usage:
    python -m experiments.runner                  # default config (monthly, raw $, SPY)
    python -m experiments.runner --full           # full grid (all freqs × units × benchmarks)
    python -m experiments.runner --baseline-only  # baseline only, no noise factors
    python -m experiments.runner --freq ME --flow raw --benchmark SPY
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    get_prepared_data_with_peers, ETF_NAMES, PEER_ETF_NAMES, ALL_ETF_NAMES,
)
from analysis import (
    auto_lags, cross_correlation_all_etfs,
    r_squared_by_lag, r_squared_by_lag_all_etfs,
    lag_regression, lag_regression_all_etfs,
    granger_causality_test,
    relative_performance_regression, relative_performance_all_etfs,
    asymmetry_regression, asymmetry_all_etfs,
    panel_regression, panel_regression_comparison,
    seasonality_analysis, seasonality_inflow_outflow,
    compute_etf_drawdowns, drawdown_flow_analysis, drawdown_flow_regression,
)
from noise_factors import apply_factors  # noqa: E402 — src/ is on sys.path

from experiments.config import (
    MODELS, FLOW_UNITS, FACTOR_COMBOS,
    DEFAULT_CONFIG, FULL_CONFIG,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _resolve_columns(freq: str, flow_unit: str):
    """Determine flow and return column names based on frequency and flow unit."""
    if freq == "D":
        return_col = "Return"
        if flow_unit == "pct_aum":
            flow_col = "Flow_Pct"
        else:
            flow_col = "Fund_Flow"
    else:
        return_col = "Return_Cum"
        if flow_unit == "pct_aum":
            flow_col = "Flow_Pct"
        else:
            flow_col = "Flow_Sum"
    return flow_col, return_col


def _safe_float(val):
    """Convert value to float, returning NaN on failure."""
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def run_model(model_name: str, df: pd.DataFrame,
              flow_col: str, return_col: str,
              freq: str) -> tuple[dict, pd.DataFrame | None]:
    """Run a single model and return (summary_dict, detail_df).

    summary_dict: model, r2, beta_lag1, beta_lag1_p, f_stat, n_obs, n_etfs,
                  gate_pass, extra_json.
    detail_df: model-specific detailed results (coefficients, per-ETF, etc.)
    """
    model_spec = MODELS[model_name]
    kwargs = dict(model_spec.get("kwargs", {}))  # copy to avoid mutation
    excess_col = "Excess_Return" if "Excess_Return" in df.columns else None

    # Detect noise-factor control columns added by factors C, D, E
    FACTOR_CONTROL_COLS = [
        "VIX_Close",       # Factor C
        "VIX_High",        # Factor C (regime_dummy mode)
        "month_end",       # Factor D
        "quarter_end",     # Factor D
        "january",         # Factor D
        "Peer_Agg_Flow",   # Factor E
    ]
    extra_controls = [c for c in FACTOR_CONTROL_COLS if c in df.columns]

    result = {
        "model": model_name,
        "r2": np.nan,
        "beta_lag1": np.nan,
        "beta_lag1_p": np.nan,
        "f_stat": np.nan,
        "n_obs": 0,
        "n_etfs": df["ETF"].nunique(),
        "gate_pass": False,
        "extra_json": "{}",
    }
    detail = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if model_name == "univariate_r2_by_lag":
                r2_df = r_squared_by_lag_all_etfs(df, flow_col, return_col,
                                                   extra_controls)
                if not r2_df.empty:
                    detail = r2_df
                    lag1 = r2_df[r2_df["lag"] == 1]
                    result["r2"] = _safe_float(lag1["r_squared"].median())
                    result["beta_lag1"] = _safe_float(lag1["coefficient"].median())
                    result["beta_lag1_p"] = _safe_float(lag1["p_value"].median())
                    result["n_obs"] = int(r2_df["ETF"].nunique())
                    best = r2_df.groupby("lag")["r_squared"].median()
                    result["extra_json"] = json.dumps({
                        "best_lag": int(best.idxmax()) if len(best) > 0 else None,
                        "best_r2": _safe_float(best.max()),
                    })

            elif model_name in ("multilag_ols", "multilag_ols_month_fe"):
                min_n = df.groupby("ETF")[flow_col].apply(lambda x: x.notna().sum()).min()
                lags = auto_lags(min_n)
                add_dummies = kwargs.get("add_month_dummies", False)
                summary = lag_regression_all_etfs(df, flow_col, return_col, lags,
                                                     add_dummies, extra_controls)
                if not summary.empty:
                    detail = summary
                    result["r2"] = _safe_float(summary["R²"].median())
                    result["beta_lag1"] = _safe_float(summary.get("Return_lag1", pd.Series([np.nan])).median())
                    result["beta_lag1_p"] = _safe_float(summary.get("Return_lag1_pval", pd.Series([np.nan])).median())
                    result["f_stat"] = _safe_float(summary["F_p_value"].median())
                    result["n_obs"] = int(summary["N"].sum())
                    result["n_etfs"] = len(summary)

            elif model_name == "cross_correlation":
                cc = cross_correlation_all_etfs(df, flow_col, return_col, max_lag=20)
                if not cc.empty:
                    detail = cc
                    pos_lags = cc[cc["lag"] > 0]
                    lag1 = pos_lags[pos_lags["lag"] == 1]
                    result["beta_lag1"] = _safe_float(lag1["correlation"].median())
                    result["beta_lag1_p"] = _safe_float(lag1["p_value"].median())
                    result["n_etfs"] = cc["ETF"].nunique()
                    median_by_lag = pos_lags.groupby("lag")["correlation"].median()
                    if len(median_by_lag) > 0:
                        result["extra_json"] = json.dumps({
                            "best_lag": int(median_by_lag.idxmax()),
                            "best_corr": _safe_float(median_by_lag.max()),
                        })

            elif model_name.startswith("panel_"):
                use_excess = kwargs.pop("use_excess", False)
                exc = excess_col if use_excess else None
                min_n = df.groupby("ETF")[flow_col].apply(lambda x: x.notna().sum()).min()
                lags = auto_lags(min_n)
                panel_result = panel_regression(
                    df, flow_col, return_col,
                    excess_return_col=exc,
                    lags=lags,
                    extra_controls=extra_controls,
                    **kwargs,
                )
                if panel_result:
                    coefs = panel_result["coefficients"]
                    # Build detail: coefficient table + summary row
                    summary_row = pd.DataFrame([{
                        "Variable": "_SUMMARY_",
                        "r2_within": panel_result["r_squared_within"],
                        "r2_between": panel_result.get("r_squared_between", np.nan),
                        "r2_overall": panel_result.get("r_squared_overall", np.nan),
                        "f_statistic": panel_result.get("f_statistic", np.nan),
                        "f_pvalue": panel_result.get("f_pvalue", np.nan),
                        "n_obs": panel_result["n_obs"],
                        "n_entities": panel_result["n_entities"],
                    }])
                    detail = pd.concat([coefs, summary_row], ignore_index=True)

                    result["r2"] = _safe_float(panel_result["r_squared_within"])
                    lag1_row = coefs[coefs["Variable"] == "Return_lag1"]
                    if not lag1_row.empty:
                        result["beta_lag1"] = _safe_float(lag1_row["Coefficient"].iloc[0])
                        result["beta_lag1_p"] = _safe_float(lag1_row["p_value"].iloc[0])
                    result["f_stat"] = _safe_float(panel_result.get("f_statistic", np.nan))
                    result["n_obs"] = panel_result["n_obs"]
                    result["n_etfs"] = panel_result["n_entities"]
                    result["extra_json"] = json.dumps({
                        "r2_overall": _safe_float(panel_result.get("r_squared_overall", np.nan)),
                        "r2_between": _safe_float(panel_result.get("r_squared_between", np.nan)),
                    })

            elif model_name == "asymmetry":
                asym = asymmetry_all_etfs(df, flow_col, return_col,
                                              extra_controls)
                if not asym.empty:
                    detail = asym
                    result["r2"] = _safe_float(asym["R²"].median())
                    result["beta_lag1"] = _safe_float(asym["Beta_Pos"].median())
                    result["n_obs"] = int(asym["N"].sum())
                    result["n_etfs"] = len(asym)
                    result["extra_json"] = json.dumps({
                        "beta_pos_median": _safe_float(asym["Beta_Pos"].median()),
                        "beta_neg_median": _safe_float(asym["Beta_Neg"].median()),
                        "asymmetry_ratio_median": _safe_float(asym["Asymmetry_Ratio"].median()),
                        "wald_p_median": _safe_float(asym["Wald_P"].median()),
                    })

            elif model_name == "relative_performance":
                if excess_col and df[excess_col].notna().any():
                    rp = relative_performance_all_etfs(df, flow_col, return_col, excess_col)
                    if not rp.empty:
                        detail = rp
                        result["r2"] = _safe_float(rp["R²_Combined"].median())
                        result["n_obs"] = int(rp["N"].sum())
                        result["n_etfs"] = len(rp)
                        result["extra_json"] = json.dumps({
                            "r2_absolute_median": _safe_float(rp["R²_Absolute"].median()),
                            "r2_excess_median": _safe_float(rp["R²_Excess"].median()),
                            "r2_combined_median": _safe_float(rp["R²_Combined"].median()),
                        })

            elif model_name == "granger":
                granger_results = []
                for etf in df["ETF"].unique():
                    etf_df = df[df["ETF"] == etf][[flow_col, return_col]].dropna()
                    if len(etf_df) > 30:
                        gc = granger_causality_test(
                            etf_df.reset_index(drop=True),
                            flow_col, return_col, max_lag=5)
                        if not gc.empty:
                            gc["ETF"] = etf
                            granger_results.append(gc)
                if granger_results:
                    gc_all = pd.concat(granger_results, ignore_index=True)
                    detail = gc_all
                    ret_to_flow = gc_all[gc_all["direction"] == "Returns → Flows"]
                    if not ret_to_flow.empty:
                        lag1 = ret_to_flow[ret_to_flow["lag"] == 1]
                        result["f_stat"] = _safe_float(lag1["F_statistic"].median())
                        result["beta_lag1_p"] = _safe_float(lag1["p_value"].median())
                        result["n_etfs"] = gc_all["ETF"].nunique()
                        result["extra_json"] = json.dumps({
                            "pct_significant_lag1": _safe_float(
                                (lag1["p_value"] < 0.05).mean() if len(lag1) > 0 else np.nan
                            ),
                        })

            elif model_name == "seasonality":
                season = seasonality_analysis(df, flow_col)
                if not season.empty:
                    detail = season
                    result["n_obs"] = int(season["Count"].sum())
                    result["extra_json"] = json.dumps({
                        "jan_mean": _safe_float(season[season["Month"] == 1]["Mean"].iloc[0]),
                        "best_month": int(season.loc[season["Mean"].idxmax(), "Month"]),
                        "worst_month": int(season.loc[season["Mean"].idxmin(), "Month"]),
                    })

            elif model_name == "drawdown":
                dd = compute_etf_drawdowns(df, return_col, min_depth_pct=10.0)
                if not dd.empty:
                    analysis = drawdown_flow_analysis(df, dd, flow_col)
                    if not analysis.empty:
                        reg = drawdown_flow_regression(analysis)
                        if not reg.empty:
                            detail = reg
                            result["n_obs"] = int(reg["N"].sum())
                            result["n_etfs"] = dd["ETF"].nunique()
                            row_1m = reg[reg["Horizon"] == "1m"]
                            if not row_1m.empty:
                                result["beta_lag1"] = _safe_float(row_1m["β_Depth"].iloc[0])
                                result["beta_lag1_p"] = _safe_float(row_1m["β_Depth_p"].iloc[0])
                                result["r2"] = _safe_float(row_1m["R²"].iloc[0])
                            result["extra_json"] = json.dumps({
                                row["Horizon"]: {
                                    "beta_depth": _safe_float(row["β_Depth"]),
                                    "beta_depth_p": _safe_float(row["β_Depth_p"]),
                                    "r2": _safe_float(row["R²"]),
                                }
                                for _, row in reg.iterrows()
                            })

    except Exception as e:
        logger.warning("Model %s failed: %s", model_name, e)
        result["extra_json"] = json.dumps({"error": str(e)})

    # Gate check: β > 0 AND p < 0.05
    b = result["beta_lag1"]
    p = result["beta_lag1_p"]
    if not np.isnan(b) and not np.isnan(p):
        result["gate_pass"] = bool(b > 0 and p < 0.05)

    return result, detail


def run_experiment(experiment_id: str, factor_ids: list[str] | None,
                   df: pd.DataFrame, flow_col: str, return_col: str,
                   freq: str, flow_unit: str, benchmark: str,
                   model_names: list[str]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run all specified models for one experiment (one factor combo).

    Returns:
        summary_rows: list of summary dicts, one per model.
        details: {model_name: detail_df} for models that produced detail.
    """
    # Apply noise factors
    if factor_ids:
        df = apply_factors(df, factor_ids, flow_col=flow_col)

    rows = []
    details = {}
    for model_name in model_names:
        logger.info("  Running %s ...", model_name)
        result, detail = run_model(model_name, df, flow_col, return_col, freq)
        result["experiment_id"] = experiment_id
        result["factors"] = "+".join(factor_ids) if factor_ids else "(none)"
        result["freq"] = freq
        result["flow_unit"] = flow_unit
        result["benchmark"] = benchmark
        rows.append(result)
        if detail is not None and not detail.empty:
            details[model_name] = detail

    return rows, details


def _exp_dir(experiment_id: str) -> Path:
    """Return the output directory for an experiment."""
    if experiment_id == "baseline":
        return RESULTS_DIR / "baseline"
    return RESULTS_DIR / "noise" / experiment_id


def run_grid(config: dict) -> pd.DataFrame:
    """Run the full experiment grid.

    Returns:
        DataFrame with one row per experiment × freq × flow_unit × benchmark × model.
    """
    all_rows = []
    all_details: dict[str, dict[str, pd.DataFrame]] = {}  # {exp_id: {model: df}}
    total_experiments = 0

    for freq in config["frequencies"]:
        for benchmark in config["benchmarks"]:
            logger.info("Loading data: freq=%s, benchmark=%s", freq, benchmark)
            df = get_prepared_data_with_peers(
                freq=freq,
                zscore_type=config["zscore_type"],
                benchmark=benchmark,
            )
            logger.info("  Loaded %d rows, %d ETFs", len(df), df["ETF"].nunique())

            for flow_unit in config["flow_units"]:
                flow_col, return_col = _resolve_columns(freq, flow_unit)

                # Check that flow column exists and has data
                if flow_col not in df.columns or df[flow_col].notna().sum() == 0:
                    logger.warning("  Skipping flow_unit=%s: column %s empty",
                                   flow_unit, flow_col)
                    continue

                # Baseline
                if config["include_baseline"]:
                    total_experiments += 1
                    logger.info("Running baseline [%s/%s/%s]", freq, flow_unit, benchmark)
                    rows, details = run_experiment(
                        "baseline", None, df,
                        flow_col, return_col, freq, flow_unit, benchmark,
                        config["models"],
                    )
                    all_rows.extend(rows)
                    all_details["baseline"] = details

                # Factor combinations
                for exp_id, factor_ids in config["factor_combos"]:
                    total_experiments += 1
                    logger.info("Running %s [%s/%s/%s]", exp_id, freq, flow_unit, benchmark)
                    rows, details = run_experiment(
                        exp_id, factor_ids, df,
                        flow_col, return_col, freq, flow_unit, benchmark,
                        config["models"],
                    )
                    all_rows.extend(rows)
                    all_details[exp_id] = details

    logger.info("Completed %d experiments, %d total result rows",
                total_experiments, len(all_rows))

    results = pd.DataFrame(all_rows)

    # Compute R² delta vs baseline (in basis points)
    if not results.empty and "baseline" in results["experiment_id"].values:
        baseline_r2 = results[results["experiment_id"] == "baseline"].set_index(
            ["freq", "flow_unit", "benchmark", "model"])["r2"]
        results["r2_delta_bp"] = results.apply(
            lambda row: _compute_r2_delta(row, baseline_r2), axis=1)
    else:
        results["r2_delta_bp"] = np.nan

    # Save detail CSVs
    for exp_id, details in all_details.items():
        exp_dir = _exp_dir(exp_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        for model_name, detail_df in details.items():
            detail_df.to_csv(exp_dir / f"{model_name}.csv", index=False)
        logger.info("Saved %d detail CSVs to %s", len(details), exp_dir)

    return results


def _compute_r2_delta(row, baseline_r2: pd.Series) -> float:
    """Compute R² improvement vs baseline in basis points."""
    key = (row["freq"], row["flow_unit"], row["benchmark"], row["model"])
    try:
        base = baseline_r2.loc[key]
        if np.isnan(base) or np.isnan(row["r2"]):
            return np.nan
        return (row["r2"] - base) * 10000  # basis points
    except KeyError:
        return np.nan


def save_results(results: pd.DataFrame, tag: str = ""):
    """Save summary results to CSV files in experiments/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Master results table
    suffix = f"_{tag}" if tag else ""
    master_path = RESULTS_DIR / f"master_results{suffix}.csv"
    results.to_csv(master_path, index=False)
    logger.info("Saved master results to %s (%d rows)", master_path, len(results))

    # Per-experiment summary CSVs
    for exp_id in results["experiment_id"].unique():
        exp_df = results[results["experiment_id"] == exp_id]
        exp_dir = _exp_dir(exp_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_df.to_csv(exp_dir / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument("--full", action="store_true",
                        help="Run full grid (all freqs × units × benchmarks)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run baseline only (no noise factors)")
    parser.add_argument("--freq", nargs="+", default=None,
                        help="Frequencies to run (D W ME QE)")
    parser.add_argument("--flow", nargs="+", default=None,
                        help="Flow units to run (raw pct_aum)")
    parser.add_argument("--benchmark", nargs="+", default=None,
                        help="Benchmarks (SPY QQQ peer_avg)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to run")
    parser.add_argument("--tag", default="",
                        help="Tag for output files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    if args.full:
        config = FULL_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    if args.freq:
        config["frequencies"] = args.freq
    if args.flow:
        config["flow_units"] = args.flow
    if args.benchmark:
        config["benchmarks"] = args.benchmark
    if args.models:
        config["models"] = args.models
    if args.baseline_only:
        config["factor_combos"] = []
        config["include_baseline"] = True

    # Summary
    n_combos = len(config["factor_combos"]) + (1 if config["include_baseline"] else 0)
    n_dims = (len(config["frequencies"]) * len(config["flow_units"])
              * len(config["benchmarks"]))
    n_models = len(config["models"])
    total = n_combos * n_dims * n_models
    logger.info("Experiment grid: %d experiments × %d dimension combos × %d models = %d total runs",
                n_combos, n_dims, n_models, total)

    t0 = time.time()
    results = run_grid(config)
    elapsed = time.time() - t0

    if not results.empty:
        save_results(results, tag=args.tag)
        logger.info("Done in %.1fs. %d result rows.", elapsed, len(results))

        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        gate_pass = results[results["gate_pass"]]
        print(f"Total runs: {len(results)}")
        print(f"Gate pass (B>0, p<0.05): {len(gate_pass)} / {len(results)}")
        if not gate_pass.empty:
            print("\nTop experiments by R2:")
            top = gate_pass.nlargest(10, "r2")[
                ["experiment_id", "freq", "flow_unit", "model", "r2", "beta_lag1", "beta_lag1_p"]
            ]
            print(top.to_string(index=False))
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
