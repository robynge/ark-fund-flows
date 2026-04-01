"""Jordà (2005) Local Projection impulse response estimation.

Estimates how fund flows respond to a return shock at each horizon h=0..H
by running separate panel regressions for each horizon:

    Flow_{i,t+h} = alpha_i + beta_h * Return_{i,t} + gamma * Controls + epsilon

The sequence of beta_h traces out the impulse response function.

Reference:
    Jordà, Ò. (2005). Estimation and Inference of Impulse Responses
    by Local Projections. American Economic Review, 95(1), 161-182.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _build_lead_flow(df: pd.DataFrame, flow_col: str,
                     max_horizon: int) -> pd.DataFrame:
    """Pre-compute Flow_{t+h} for h=0..max_horizon as separate columns."""
    pdf = df.copy()
    for h in range(max_horizon + 1):
        pdf[f"Flow_lead{h}"] = pdf.groupby("ETF")[flow_col].shift(-h)
    return pdf


def _run_single_horizon(pdf: pd.DataFrame, y_col: str,
                        x_cols: list[str], entity_col: str = "ETF",
                        cluster: bool = True,
                        hac_maxlags: int | None = None) -> dict | None:
    """Run one OLS regression with entity-demeaning.

    SE options:
    - cluster=True, hac_maxlags=None: entity-clustered (default)
    - hac_maxlags=int: Newey-West HAC with given maxlags (for LP overlap)
    """
    keep = [entity_col, y_col] + x_cols
    sub = pdf[keep].dropna()
    if len(sub) < 30:
        return None

    y = sub[y_col]
    X = sub[x_cols].copy()

    # Within transformation (entity-demean)
    y = y - sub.groupby(entity_col)[y_col].transform("mean")
    for col in x_cols:
        X[col] = X[col] - sub.groupby(entity_col)[col].transform("mean")

    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hac_maxlags is not None:
            model = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_maxlags},
            )
        elif cluster:
            model = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub[entity_col]},
            )
        else:
            model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "params": model.params,
        "bse": model.bse,
        "pvalues": model.pvalues,
        "nobs": int(model.nobs),
        "rsquared": model.rsquared,
    }


def local_projection(df: pd.DataFrame,
                     flow_col: str,
                     return_col: str,
                     max_horizon: int = 60,
                     controls: list[str] | None = None,
                     entity_fe: bool = True,
                     ci_level: float = 0.95) -> pd.DataFrame:
    """Estimate LP impulse response: effect of Return_t on Flow_{t+h}.

    Parameters:
        df: panel DataFrame with ETF, Date, flow_col, return_col columns
        flow_col: dependent variable (e.g. "Fund_Flow" or "Flow_Pct")
        return_col: shock variable (e.g. "Return")
        max_horizon: maximum horizon h (days/periods)
        controls: additional control variables
        ci_level: confidence interval level (default 95%)

    Returns DataFrame with columns:
        horizon, beta, se, ci_lower, ci_upper, p_value, n_obs, r2
    """
    # Filter out inf/extreme values
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col]) & np.isfinite(pdf[return_col])]

    # Build lead columns
    pdf = _build_lead_flow(pdf, flow_col, max_horizon)

    # Build regressor list
    x_cols = [return_col]
    if controls:
        x_cols += [c for c in controls if c in pdf.columns]

    z_crit = -1 * __import__("scipy").stats.norm.ppf((1 - ci_level) / 2)

    results = []
    for h in range(max_horizon + 1):
        y_col = f"Flow_lead{h}"
        # Use HAC SE with horizon-adaptive maxlags to handle LP overlap
        res = _run_single_horizon(pdf, y_col, x_cols,
                                  cluster=entity_fe,
                                  hac_maxlags=max(1, h))
        if res is None:
            results.append({
                "horizon": h, "beta": np.nan, "se": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "p_value": np.nan, "n_obs": 0, "r2": np.nan,
            })
            continue

        beta = res["params"].get(return_col, np.nan)
        se = res["bse"].get(return_col, np.nan)
        pval = res["pvalues"].get(return_col, np.nan)

        results.append({
            "horizon": h,
            "beta": beta,
            "se": se,
            "ci_lower": beta - z_crit * se,
            "ci_upper": beta + z_crit * se,
            "p_value": pval,
            "n_obs": res["nobs"],
            "r2": res["rsquared"],
        })

    return pd.DataFrame(results)


def local_projection_asymmetric(df: pd.DataFrame,
                                flow_col: str,
                                return_col: str,
                                max_horizon: int = 60,
                                controls: list[str] | None = None,
                                ci_level: float = 0.95) -> pd.DataFrame:
    """LP with separate positive and negative return shocks.

    Splits Return into Return_pos = max(Return, 0) and
    Return_neg = min(Return, 0), then estimates both in the same regression.

    Returns DataFrame with columns:
        horizon, beta_pos, se_pos, ci_lower_pos, ci_upper_pos, p_value_pos,
        beta_neg, se_neg, ci_lower_neg, ci_upper_neg, p_value_neg, n_obs, r2
    """
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col]) & np.isfinite(pdf[return_col])]

    # Create pos/neg return columns
    pdf["Return_pos"] = pdf[return_col].clip(lower=0)
    pdf["Return_neg"] = pdf[return_col].clip(upper=0)

    pdf = _build_lead_flow(pdf, flow_col, max_horizon)

    x_cols = ["Return_pos", "Return_neg"]
    if controls:
        x_cols += [c for c in controls if c in pdf.columns]

    z_crit = -1 * __import__("scipy").stats.norm.ppf((1 - ci_level) / 2)

    results = []
    for h in range(max_horizon + 1):
        y_col = f"Flow_lead{h}"
        res = _run_single_horizon(pdf, y_col, x_cols, hac_maxlags=max(1, h))
        if res is None:
            results.append({"horizon": h,
                            "beta_pos": np.nan, "se_pos": np.nan,
                            "ci_lower_pos": np.nan, "ci_upper_pos": np.nan,
                            "p_value_pos": np.nan,
                            "beta_neg": np.nan, "se_neg": np.nan,
                            "ci_lower_neg": np.nan, "ci_upper_neg": np.nan,
                            "p_value_neg": np.nan,
                            "n_obs": 0, "r2": np.nan})
            continue

        row = {"horizon": h, "n_obs": res["nobs"], "r2": res["rsquared"]}
        for tag, var in [("pos", "Return_pos"), ("neg", "Return_neg")]:
            b = res["params"].get(var, np.nan)
            s = res["bse"].get(var, np.nan)
            p = res["pvalues"].get(var, np.nan)
            row[f"beta_{tag}"] = b
            row[f"se_{tag}"] = s
            row[f"ci_lower_{tag}"] = b - z_crit * s
            row[f"ci_upper_{tag}"] = b + z_crit * s
            row[f"p_value_{tag}"] = p
        results.append(row)

    return pd.DataFrame(results)


def local_projection_subsample(df: pd.DataFrame,
                                flow_col: str,
                                return_col: str,
                                periods: dict[str, tuple[str, str]],
                                max_horizon: int = 60,
                                controls: list[str] | None = None,
                                ci_level: float = 0.95
                                ) -> dict[str, pd.DataFrame]:
    """Run LP separately for each sub-period.

    Parameters:
        periods: {"bull": ("2020-01-01", "2021-12-31"),
                  "bear": ("2022-01-01", "2024-12-31")}

    Returns dict mapping period name to LP results DataFrame.
    """
    results = {}
    for name, (start, end) in periods.items():
        mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
        sub = df[mask].copy()
        if len(sub) < 100:
            logger.warning("Sub-period %s has only %d rows, skipping", name, len(sub))
            continue
        lp = local_projection(sub, flow_col, return_col, max_horizon,
                              controls, ci_level=ci_level)
        results[name] = lp
    return results


def local_projection_cumulative(df: pd.DataFrame,
                                flow_col: str,
                                return_col: str,
                                max_horizon: int = 60,
                                controls: list[str] | None = None,
                                ci_level: float = 0.95) -> pd.DataFrame:
    """Cumulative LP: effect on cumulative flow over horizon [0, h].

    Instead of Flow_{t+h}, uses sum(Flow_{t}, Flow_{t+1}, ..., Flow_{t+h}).
    This shows the total cumulative flow response, not just the marginal response
    at each horizon.
    """
    pdf = df.copy()
    pdf = pdf[np.isfinite(pdf[flow_col]) & np.isfinite(pdf[return_col])]

    # Build cumulative flow leads
    for h in range(max_horizon + 1):
        # Cumulative flow from t to t+h
        pdf[f"CumFlow_lead{h}"] = pdf.groupby("ETF")[flow_col].transform(
            lambda x, _h=h: x.rolling(_h + 1, min_periods=_h + 1).sum().shift(-_h)
        )

    x_cols = [return_col]
    if controls:
        x_cols += [c for c in controls if c in pdf.columns]

    z_crit = -1 * __import__("scipy").stats.norm.ppf((1 - ci_level) / 2)

    results = []
    for h in range(max_horizon + 1):
        y_col = f"CumFlow_lead{h}"
        res = _run_single_horizon(pdf, y_col, x_cols, hac_maxlags=max(1, h))
        if res is None:
            results.append({"horizon": h, "beta_cum": np.nan, "se": np.nan,
                            "ci_lower": np.nan, "ci_upper": np.nan,
                            "p_value": np.nan, "n_obs": 0, "r2": np.nan})
            continue

        beta = res["params"].get(return_col, np.nan)
        se = res["bse"].get(return_col, np.nan)
        pval = res["pvalues"].get(return_col, np.nan)
        results.append({
            "horizon": h, "beta_cum": beta, "se": se,
            "ci_lower": beta - z_crit * se,
            "ci_upper": beta + z_crit * se,
            "p_value": pval,
            "n_obs": res["nobs"], "r2": res["rsquared"],
        })

    return pd.DataFrame(results)
