"""Statistical analysis: cross-correlation, regressions, Granger causality, seasonality,
relative performance, asymmetry, and panel regressions."""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import statsmodels.api as sm


def cross_correlation(flow_series: pd.Series, return_series: pd.Series,
                      max_lag: int = 20) -> pd.DataFrame:
    """
    Compute cross-correlation between flows and returns at various lags.

    Positive lag k: corr(flow(t), return(t-k)) — does past return predict current flow?
    Negative lag k: corr(flow(t), return(t+|k|)) — does current flow predict future return?

    Returns DataFrame with columns: lag, correlation, p_value.
    """
    results = []
    flow = flow_series.dropna()
    ret = return_series.dropna()

    # Align on common index
    common_idx = flow.index.intersection(ret.index)
    flow = flow.loc[common_idx]
    ret = ret.loc[common_idx]

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # Past returns → current flows
            shifted_ret = ret.shift(lag)
        elif lag < 0:
            # Current flows → future returns
            shifted_ret = ret.shift(lag)
        else:
            shifted_ret = ret

        valid = pd.DataFrame({"flow": flow, "ret": shifted_ret}).dropna()
        if len(valid) < 10:
            continue

        corr, p_val = stats.pearsonr(valid["flow"], valid["ret"])
        results.append({"lag": lag, "correlation": corr, "p_value": p_val})

    return pd.DataFrame(results)


def cross_correlation_all_etfs(df: pd.DataFrame, flow_col: str, return_col: str,
                               max_lag: int = 20) -> pd.DataFrame:
    """Compute cross-correlation for all ETFs in the DataFrame."""
    results = []
    for etf in df["ETF"].unique():
        etf_df = df[df["ETF"] == etf].set_index("Date").sort_index()
        cc = cross_correlation(etf_df[flow_col], etf_df[return_col], max_lag)
        cc["ETF"] = etf
        results.append(cc)
    return pd.concat(results, ignore_index=True)


def lag_regression(df: pd.DataFrame, flow_col: str, return_col: str,
                   lags: list[int], add_month_dummies: bool = False) -> dict:
    """
    OLS regression: flow(t) ~ return(t-lag1) + return(t-lag2) + ... [+ month_dummies]

    Returns dict with keys: coefficients, r_squared, adj_r_squared, f_pvalue, n_obs, summary_df
    """
    etf_df = df.copy().set_index("Date").sort_index()

    # Create lagged return columns
    X_data = {}
    for lag in lags:
        X_data[f"Return_lag{lag}"] = etf_df[return_col].shift(lag)

    X = pd.DataFrame(X_data, index=etf_df.index)

    if add_month_dummies:
        months = pd.get_dummies(etf_df.index.month, prefix="month", drop_first=True, dtype=float)
        months.index = etf_df.index
        X = pd.concat([X, months], axis=1)

    y = etf_df[flow_col]

    # Drop NaN
    valid = pd.concat([y, X], axis=1).dropna()
    if len(valid) < len(lags) + 5:
        return None

    y_clean = valid.iloc[:, 0]
    X_clean = valid.iloc[:, 1:]
    X_clean = sm.add_constant(X_clean)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y_clean, X_clean).fit()

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    return {
        "coefficients": coef_df,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_pvalue": model.f_pvalue,
        "n_obs": int(model.nobs),
        "aic": model.aic,
        "bic": model.bic,
    }


def lag_regression_all_etfs(df: pd.DataFrame, flow_col: str, return_col: str,
                            lags: list[int],
                            add_month_dummies: bool = False) -> pd.DataFrame:
    """Run lag regression for all ETFs and return summary table."""
    rows = []
    for etf in df["ETF"].unique():
        etf_df = df[df["ETF"] == etf]
        result = lag_regression(etf_df, flow_col, return_col, lags, add_month_dummies)
        if result is None:
            continue
        row = {"ETF": etf, "R²": result["r_squared"],
               "Adj_R²": result["adj_r_squared"],
               "F_p_value": result["f_pvalue"], "N": result["n_obs"]}
        # Add lag coefficients
        for _, coef_row in result["coefficients"].iterrows():
            if coef_row["Variable"].startswith("Return_lag"):
                row[coef_row["Variable"]] = coef_row["Coefficient"]
                row[f"{coef_row['Variable']}_pval"] = coef_row["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def granger_causality_test(df: pd.DataFrame, flow_col: str, return_col: str,
                           max_lag: int = 5) -> pd.DataFrame:
    """
    Granger causality test in both directions.
    Returns DataFrame with lag, direction, F-statistic, p-value.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    etf_df = df[[flow_col, return_col]].dropna()
    if len(etf_df) < max_lag * 3:
        return pd.DataFrame()

    results = []

    # Test: do returns Granger-cause flows?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gc1 = grangercausalitytests(
                etf_df[[flow_col, return_col]], maxlag=max_lag, verbose=False
            )
            for lag_val in range(1, max_lag + 1):
                f_stat = gc1[lag_val][0]["ssr_ftest"][0]
                p_val = gc1[lag_val][0]["ssr_ftest"][1]
                results.append({
                    "lag": lag_val,
                    "direction": "Returns → Flows",
                    "F_statistic": f_stat,
                    "p_value": p_val,
                })
        except Exception:
            pass

    # Test: do flows Granger-cause returns?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gc2 = grangercausalitytests(
                etf_df[[return_col, flow_col]], maxlag=max_lag, verbose=False
            )
            for lag_val in range(1, max_lag + 1):
                f_stat = gc2[lag_val][0]["ssr_ftest"][0]
                p_val = gc2[lag_val][0]["ssr_ftest"][1]
                results.append({
                    "lag": lag_val,
                    "direction": "Flows → Returns",
                    "F_statistic": f_stat,
                    "p_value": p_val,
                })
        except Exception:
            pass

    return pd.DataFrame(results)


def seasonality_analysis(df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
    """
    Compute average flows by calendar month across all years.
    Returns DataFrame with month, mean_flow, median_flow, std_flow, count.
    """
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    monthly = df.groupby(["Month", "Month_Name"])[flow_col].agg(
        Mean="mean", Median="median", Std="std", Count="count"
    ).reset_index()

    return monthly.sort_values("Month")


def r_squared_by_lag(df: pd.DataFrame, flow_col: str, return_col: str,
                     lag_range: range) -> pd.DataFrame:
    """
    Compute R² from simple OLS (flow ~ return_lag_k) for each lag k.
    Useful for finding optimal lag horizon.
    """
    etf_df = df.copy().set_index("Date").sort_index()
    results = []

    for lag in lag_range:
        X = etf_df[return_col].shift(lag)
        y = etf_df[flow_col]
        valid = pd.concat([y, X], axis=1).dropna()
        if len(valid) < 10:
            continue

        y_c = valid.iloc[:, 0]
        X_c = sm.add_constant(valid.iloc[:, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y_c, X_c).fit()

        results.append({
            "lag": lag,
            "r_squared": model.rsquared,
            "coefficient": model.params.iloc[1],
            "p_value": model.pvalues.iloc[1],
        })

    return pd.DataFrame(results)


# ============================================================
# Relative Performance Analysis
# ============================================================

def relative_performance_regression(df: pd.DataFrame, flow_col: str,
                                     return_col: str, excess_return_col: str,
                                     lags: list[int]) -> dict | None:
    """
    Compare absolute vs excess return as predictors of flows.

    Model 1: Flow ~ AbsoluteReturn(t-k)
    Model 2: Flow ~ ExcessReturn(t-k)
    Model 3: Flow ~ AbsoluteReturn(t-k) + ExcessReturn(t-k)
    """
    etf_df = df.copy().set_index("Date").sort_index()

    abs_lags = {f"Abs_lag{k}": etf_df[return_col].shift(k) for k in lags}
    exc_lags = {f"Exc_lag{k}": etf_df[excess_return_col].shift(k) for k in lags}

    X_abs = pd.DataFrame(abs_lags, index=etf_df.index)
    X_exc = pd.DataFrame(exc_lags, index=etf_df.index)
    X_both = pd.concat([X_abs, X_exc], axis=1)
    y = etf_df[flow_col]

    models = {}
    for name, X in [("absolute", X_abs), ("excess", X_exc), ("combined", X_both)]:
        valid = pd.concat([y, X], axis=1).dropna()
        if len(valid) < X.shape[1] + 5:
            return None
        y_c = valid.iloc[:, 0]
        X_c = sm.add_constant(valid.iloc[:, 1:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.OLS(y_c, X_c).fit()
        coef_df = pd.DataFrame({
            "Variable": result.params.index,
            "Coefficient": result.params.values,
            "Std_Error": result.bse.values,
            "t_stat": result.tvalues.values,
            "p_value": result.pvalues.values,
        })
        models[name] = {
            "coefficients": coef_df,
            "r_squared": result.rsquared,
            "adj_r_squared": result.rsquared_adj,
            "n_obs": int(result.nobs),
            "aic": result.aic,
        }

    return models


def relative_performance_all_etfs(df: pd.DataFrame, flow_col: str,
                                   return_col: str, excess_return_col: str,
                                   lags: list[int]) -> pd.DataFrame:
    """Summary table: ETF, R²_Absolute, R²_Excess, R²_Combined, N."""
    rows = []
    for etf in df["ETF"].unique():
        etf_df = df[df["ETF"] == etf]
        if etf_df[excess_return_col].dropna().empty:
            continue
        result = relative_performance_regression(
            etf_df, flow_col, return_col, excess_return_col, lags)
        if result is None:
            continue
        rows.append({
            "ETF": etf,
            "R²_Absolute": result["absolute"]["r_squared"],
            "R²_Excess": result["excess"]["r_squared"],
            "R²_Combined": result["combined"]["r_squared"],
            "N": result["absolute"]["n_obs"],
        })
    return pd.DataFrame(rows)


# ============================================================
# Asymmetry Analysis
# ============================================================

def asymmetry_regression(df: pd.DataFrame, flow_col: str, return_col: str,
                          lags: list[int]) -> dict | None:
    """
    Piecewise regression: Flow(t) = β₁·Return⁺(t-k) + β₂·Return⁻(t-k) + ε

    Return⁺ = max(Return, 0), Return⁻ = min(Return, 0)
    Wald test for H0: β₁ + β₂ = 0 (symmetric response)
    """
    etf_df = df.copy().set_index("Date").sort_index()

    pos_lags = {}
    neg_lags = {}
    for k in lags:
        shifted = etf_df[return_col].shift(k)
        pos_lags[f"Return_pos_lag{k}"] = shifted.clip(lower=0)
        neg_lags[f"Return_neg_lag{k}"] = shifted.clip(upper=0)

    X = pd.DataFrame({**pos_lags, **neg_lags}, index=etf_df.index)
    y = etf_df[flow_col]

    valid = pd.concat([y, X], axis=1).dropna()
    if len(valid) < len(lags) * 2 + 5:
        return None

    y_c = valid.iloc[:, 0]
    X_c = sm.add_constant(valid.iloc[:, 1:])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y_c, X_c).fit()

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    pos_cols = [c for c in model.params.index if "pos" in c]
    neg_cols = [c for c in model.params.index if "neg" in c]
    beta_pos = model.params[pos_cols].mean()
    beta_neg = model.params[neg_cols].mean()

    # Wald test: H0: sum of pos coeffs + sum of neg coeffs = 0
    R = np.zeros((1, len(model.params)))
    for col in pos_cols:
        R[0, list(model.params.index).index(col)] = 1.0
    for col in neg_cols:
        R[0, list(model.params.index).index(col)] = 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wald = model.wald_test(R, scalar=True)
        wald_stat = float(wald.statistic)
        wald_p = float(wald.pvalue)
    except Exception:
        wald_stat = np.nan
        wald_p = np.nan

    asym_ratio = beta_pos / abs(beta_neg) if abs(beta_neg) > 1e-10 else np.nan

    return {
        "coefficients": coef_df,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "beta_pos": beta_pos,
        "beta_neg": beta_neg,
        "asymmetry_ratio": asym_ratio,
        "wald_stat": wald_stat,
        "wald_p": wald_p,
    }


def asymmetry_all_etfs(df: pd.DataFrame, flow_col: str, return_col: str,
                        lags: list[int]) -> pd.DataFrame:
    """Summary table: ETF, Beta_Pos, Beta_Neg, Asymmetry_Ratio, Wald_P, R², N."""
    rows = []
    for etf in df["ETF"].unique():
        etf_df = df[df["ETF"] == etf]
        result = asymmetry_regression(etf_df, flow_col, return_col, lags)
        if result is None:
            continue
        rows.append({
            "ETF": etf,
            "Beta_Pos": result["beta_pos"],
            "Beta_Neg": result["beta_neg"],
            "Asymmetry_Ratio": result["asymmetry_ratio"],
            "Wald_P": result["wald_p"],
            "R²": result["r_squared"],
            "N": result["n_obs"],
        })
    return pd.DataFrame(rows)


# ============================================================
# Panel Regression
# ============================================================

def panel_regression(df: pd.DataFrame, flow_col: str, return_col: str,
                     excess_return_col: str | None = None,
                     lags: list[int] = [1],
                     entity_effects: bool = True,
                     time_effects: bool = False,
                     cluster_entity: bool = True,
                     add_controls: bool = False) -> dict | None:
    """
    Panel regression using linearmodels PanelOLS.

    Entity (ETF) fixed effects, optional time fixed effects,
    clustered standard errors by entity, optional volatility control.
    """
    from linearmodels.panel import PanelOLS, PooledOLS

    pdf = df.copy()

    # Build lagged return columns
    for k in lags:
        pdf[f"Return_lag{k}"] = pdf.groupby("ETF")[return_col].shift(k)
        if excess_return_col and excess_return_col in pdf.columns:
            pdf[f"Excess_lag{k}"] = pdf.groupby("ETF")[excess_return_col].shift(k)

    # Optional: rolling volatility control
    if add_controls:
        pdf["Volatility"] = pdf.groupby("ETF")[return_col].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )

    # Build X matrix
    x_cols = [f"Return_lag{k}" for k in lags]
    if excess_return_col and excess_return_col in pdf.columns:
        x_cols += [f"Excess_lag{k}" for k in lags]
    if add_controls:
        x_cols.append("Volatility")

    pdf = pdf.dropna(subset=[flow_col] + x_cols)
    if len(pdf) < len(x_cols) + 10:
        return None

    # Set multi-index for panel
    pdf["ETF_cat"] = pd.Categorical(pdf["ETF"])
    pdf = pdf.set_index(["ETF_cat", "Date"])

    y = pdf[flow_col]
    X = pdf[x_cols]
    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not entity_effects and not time_effects:
            model = PooledOLS(y, X).fit(
                cov_type="clustered" if cluster_entity else "unadjusted",
                cluster_entity=cluster_entity,
            )
        else:
            model = PanelOLS(
                y, X,
                entity_effects=entity_effects,
                time_effects=time_effects,
                drop_absorbed=True,
            ).fit(
                cov_type="clustered" if cluster_entity else "unadjusted",
                cluster_entity=cluster_entity,
            )

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.std_errors.values,
        "t_stat": model.tstats.values,
        "p_value": model.pvalues.values,
    })

    result = {
        "coefficients": coef_df,
        "r_squared_within": getattr(model, "rsquared_within", model.rsquared),
        "r_squared_between": getattr(model, "rsquared_between", np.nan),
        "r_squared_overall": getattr(model, "rsquared_overall", model.rsquared),
        "n_obs": int(model.nobs),
        "n_entities": int(model.entity_info.total) if hasattr(model, "entity_info") else pdf.index.get_level_values(0).nunique(),
    }

    # Extract entity effects if available
    if entity_effects and hasattr(model, "estimated_effects"):
        effects = model.estimated_effects.copy()
        effects = effects.reset_index()
        effects.columns = ["ETF", "Date", "Effect"]
        entity_avg = effects.groupby("ETF")["Effect"].mean().reset_index()
        entity_avg.columns = ["ETF", "Fixed_Effect"]
        result["entity_effects"] = entity_avg

    return result


def panel_regression_comparison(df: pd.DataFrame, flow_col: str,
                                 return_col: str,
                                 excess_return_col: str | None = None,
                                 lags: list[int] = [1]) -> pd.DataFrame:
    """
    Run 5 panel specifications side by side:
    1. Pooled OLS
    2. Entity FE
    3. Entity + Time FE
    4. Entity FE with excess return
    5. Entity FE with controls
    """
    specs = [
        ("Pooled OLS", dict(entity_effects=False, time_effects=False,
                            excess_return_col=None, add_controls=False)),
        ("Entity FE", dict(entity_effects=True, time_effects=False,
                           excess_return_col=None, add_controls=False)),
        ("Entity+Time FE", dict(entity_effects=True, time_effects=True,
                                excess_return_col=None, add_controls=False)),
        ("Entity FE + Excess", dict(entity_effects=True, time_effects=False,
                                    excess_return_col=excess_return_col,
                                    add_controls=False)),
        ("Entity FE + Controls", dict(entity_effects=True, time_effects=False,
                                      excess_return_col=None,
                                      add_controls=True)),
    ]

    rows = []
    for name, kwargs in specs:
        result = panel_regression(df, flow_col, return_col, lags=lags, **kwargs)
        if result is None:
            continue
        row = {"Specification": name,
               "R²_within": result["r_squared_within"],
               "R²_overall": result["r_squared_overall"],
               "N": result["n_obs"],
               "Entities": result["n_entities"]}
        for _, cr in result["coefficients"].iterrows():
            if cr["Variable"] != "const":
                row[f"{cr['Variable']}_coef"] = cr["Coefficient"]
                row[f"{cr['Variable']}_pval"] = cr["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)
