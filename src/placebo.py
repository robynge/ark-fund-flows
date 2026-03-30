"""Placebo tests and robustness checks for the flow-performance analysis."""
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _panel_ols_demeaned(df: pd.DataFrame, y_col: str, x_cols: list[str],
                         entity_col: str = "ETF") -> dict | None:
    """Entity-demeaned OLS with clustered SE. Returns params, bse, pvalues, etc."""
    keep = [entity_col, y_col] + x_cols
    sub = df[keep].dropna()
    if len(sub) < 30:
        return None

    y = sub[y_col] - sub.groupby(entity_col)[y_col].transform("mean")
    X = sub[x_cols].copy()
    for col in x_cols:
        X[col] = X[col] - sub.groupby(entity_col)[col].transform("mean")
    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": sub[entity_col]})

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    coef_df = coef_df[coef_df["Variable"] != "const"].reset_index(drop=True)

    return {
        "coefficients": coef_df,
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
        "n_entities": sub[entity_col].nunique(),
    }


def placebo_test(df: pd.DataFrame, flow_col: str, return_col: str,
                 lag_windows: list[tuple[int, int]] = [(1, 5), (6, 20), (21, 60)],
                 lead_windows: list[tuple[int, int]] = [(1, 5), (6, 20), (21, 60)],
                 ) -> dict:
    """Run main specification with LAG returns (real) vs LEAD returns (placebo).

    If lead (future) return coefficients are significant, it's a red flag
    for spurious correlation. They should be insignificant.

    Returns dict with 'real' and 'placebo' regression results.
    """
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col]) & np.isfinite(pdf[return_col])]

    # Build LAG cumulative returns (real specification)
    lag_cols = []
    for start, end in lag_windows:
        col = f"CumRet_lag{start}_{end}"
        pdf[col] = pdf.groupby("ETF")[return_col].transform(
            lambda x, s=start, e=end: sum(x.shift(k) for k in range(s, e + 1))
        )
        lag_cols.append(col)

    # Build LEAD cumulative returns (placebo — future returns)
    lead_cols = []
    for start, end in lead_windows:
        col = f"CumRet_lead{start}_{end}"
        pdf[col] = pdf.groupby("ETF")[return_col].transform(
            lambda x, s=start, e=end: sum(x.shift(-k) for k in range(s, e + 1))
        )
        lead_cols.append(col)

    real = _panel_ols_demeaned(pdf, flow_col, lag_cols)
    placebo = _panel_ols_demeaned(pdf, flow_col, lead_cols)

    return {"real": real, "placebo": placebo}


def leave_one_etf_out(df: pd.DataFrame, flow_col: str, return_col: str,
                       x_cols: list[str]) -> pd.DataFrame:
    """Run main specification dropping each ETF in turn.

    Tests whether results are driven by a single ETF (e.g. ARKK).

    Returns DataFrame with one row per excluded ETF:
        ETF_excluded, and coefficient + p-value for each regressor.
    """
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col])]
    etfs = pdf["ETF"].unique()

    rows = []
    for excluded in etfs:
        sub = pdf[pdf["ETF"] != excluded]
        result = _panel_ols_demeaned(sub, flow_col, x_cols)
        if result is None:
            continue
        row = {"ETF_excluded": excluded, "n_obs": result["n_obs"],
               "n_etfs": result["n_entities"], "r2": result["r_squared"]}
        for _, cr in result["coefficients"].iterrows():
            row[f"{cr['Variable']}_coef"] = cr["Coefficient"]
            row[f"{cr['Variable']}_pval"] = cr["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)


def subsample_comparison(df: pd.DataFrame, flow_col: str, return_col: str,
                          x_cols: list[str],
                          periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Run main specification on sub-samples (e.g. bull vs bear).

    Returns DataFrame comparing coefficients across periods.
    """
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col])]

    rows = []
    for name, (start, end) in periods.items():
        mask = (pdf["Date"] >= pd.Timestamp(start)) & (pdf["Date"] <= pd.Timestamp(end))
        sub = pdf[mask]
        result = _panel_ols_demeaned(sub, flow_col, x_cols)
        if result is None:
            continue
        row = {"period": name, "start": start, "end": end,
               "n_obs": result["n_obs"], "n_etfs": result["n_entities"],
               "r2": result["r_squared"]}
        for _, cr in result["coefficients"].iterrows():
            row[f"{cr['Variable']}_coef"] = cr["Coefficient"]
            row[f"{cr['Variable']}_pval"] = cr["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)


def fama_macbeth(df: pd.DataFrame, flow_col: str,
                 regressors: list[str],
                 date_col: str = "Date") -> dict | None:
    """Fama-MacBeth (1973) cross-sectional regression.

    Step 1: For each date, regress Flow_i ~ X_i across all ETFs.
    Step 2: Average the time series of coefficients.
    Step 3: t-stat = mean(beta) / (se(beta) / sqrt(T)).

    Returns dict with coefficients DataFrame and T (number of periods).
    """
    pdf = df[[date_col, "ETF", flow_col] + regressors].dropna()
    pdf = pdf[np.isfinite(pdf[flow_col])]

    dates = sorted(pdf[date_col].unique())
    all_betas = []

    for dt in dates:
        cross = pdf[pdf[date_col] == dt]
        if len(cross) < 5:
            continue

        y = cross[flow_col]
        X = sm.add_constant(cross[regressors])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.OLS(y, X).fit()
                betas = model.params.drop("const", errors="ignore")
                all_betas.append(betas)
            except Exception:
                continue

    if not all_betas:
        return None

    beta_df = pd.DataFrame(all_betas)
    T = len(beta_df)

    results = []
    for col in beta_df.columns:
        mean_b = beta_df[col].mean()
        se_b = beta_df[col].std() / np.sqrt(T)
        t_stat = mean_b / se_b if se_b > 1e-10 else np.nan
        p_val = 2 * (1 - __import__("scipy").stats.t.cdf(abs(t_stat), T - 1))
        results.append({
            "Variable": col,
            "Coefficient": mean_b,
            "Std_Error": se_b,
            "t_stat": t_stat,
            "p_value": p_val,
        })

    return {
        "coefficients": pd.DataFrame(results),
        "T": T,
        "n_etfs_per_period": pdf.groupby(date_col)["ETF"].nunique().median(),
    }


# ============================================================
# Heteroscedasticity tests
# ============================================================

def breusch_pagan_test(df: pd.DataFrame, y_col: str, x_cols: list[str],
                       entity_col: str = "ETF") -> dict:
    """Breusch-Pagan test for heteroscedasticity on entity-demeaned data."""
    from statsmodels.stats.diagnostic import het_breuschpagan

    keep = [entity_col, y_col] + x_cols
    sub = df[keep].dropna()
    if len(sub) < 30:
        return {"statistic": np.nan, "p_value": np.nan}

    y = sub[y_col] - sub.groupby(entity_col)[y_col].transform("mean")
    X = sub[x_cols].copy()
    for col in x_cols:
        X[col] = X[col] - sub.groupby(entity_col)[col].transform("mean")
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)

    return {"statistic": bp_stat, "p_value": bp_p}


def white_test(df: pd.DataFrame, y_col: str, x_cols: list[str],
               entity_col: str = "ETF") -> dict:
    """White's test for heteroscedasticity on entity-demeaned data."""
    from statsmodels.stats.diagnostic import het_white

    keep = [entity_col, y_col] + x_cols
    sub = df[keep].dropna()
    if len(sub) < 30:
        return {"statistic": np.nan, "p_value": np.nan}

    y = sub[y_col] - sub.groupby(entity_col)[y_col].transform("mean")
    X = sub[x_cols].copy()
    for col in x_cols:
        X[col] = X[col] - sub.groupby(entity_col)[col].transform("mean")
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    w_stat, w_p, _, _ = het_white(model.resid, model.model.exog)

    return {"statistic": w_stat, "p_value": w_p}


# ============================================================
# Driscoll-Kraay standard errors
# ============================================================

def driscoll_kraay_panel(df: pd.DataFrame, y_col: str, x_cols: list[str],
                         entity_col: str = "ETF", date_col: str = "Date",
                         maxlag: int | None = None) -> dict | None:
    """Panel OLS with Driscoll-Kraay (1998) standard errors.

    Uses linearmodels.PanelOLS with kernel covariance estimator.
    """
    from linearmodels.panel import PanelOLS

    keep = [entity_col, date_col, y_col] + x_cols
    sub = df[keep].dropna().copy()
    if len(sub) < 30:
        return None

    sub = sub.set_index([entity_col, date_col])
    y = sub[y_col]
    X = sub[x_cols]

    if maxlag is None:
        T = y.reset_index()[date_col].nunique()
        maxlag = int(T ** (1 / 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelOLS(y, X, entity_effects=True).fit(
            cov_type="kernel", kernel="bartlett", bandwidth=maxlag)

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.std_errors.values,
        "t_stat": model.tstats.values,
        "p_value": model.pvalues.values,
    }).reset_index(drop=True)

    return {
        "coefficients": coef_df,
        "r_squared_within": model.rsquared_within,
        "n_obs": int(model.nobs),
        "n_entities": model.entity_info.total,
        "bandwidth": maxlag,
    }
