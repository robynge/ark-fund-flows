"""Sirri & Tufano (1998) piecewise linear flow-performance model.

Implements the classic convexity test: do investors chase top performers
disproportionately more than they flee bottom performers?

Reference:
    Sirri, E.R. & Tufano, P. (1998). Costly Search and Mutual Fund Flows.
    The Journal of Finance, 53(5), 1589-1622.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Try to import R engine; fall back to pure Python if unavailable
try:
    from .r_engine import _py_to_r, _r_to_py, R_AVAILABLE
    import rpy2.robjects as ro
except ImportError:
    try:
        from r_engine import _py_to_r, _r_to_py, R_AVAILABLE
        import rpy2.robjects as ro
    except ImportError:
        R_AVAILABLE = False


def compute_fractional_rank(df: pd.DataFrame,
                            return_col: str = "Return_Cum",
                            date_col: str = "Date",
                            etf_col: str = "ETF") -> pd.DataFrame:
    """Cross-sectional rank each ETF's return among all ETFs per period.

    Adds columns:
        RANK: fractional rank 0-1 (0 = worst, 1 = best)
        LOWPERF:  min(0.2, RANK)
        MIDPERF:  min(0.6, RANK - LOWPERF)
        HIGHPERF: RANK - LOWPERF - MIDPERF

    The piecewise linear specification allows different slopes for
    bottom 20%, middle 60%, and top 20% performers.
    """
    df = df.copy()
    df["RANK"] = df.groupby(date_col)[return_col].rank(pct=True)
    df["LOWPERF"] = df["RANK"].clip(upper=0.2)
    df["MIDPERF"] = (df["RANK"] - df["LOWPERF"]).clip(upper=0.6)
    df["HIGHPERF"] = df["RANK"] - df["LOWPERF"] - df["MIDPERF"]
    return df


def _sirri_tufano_python(df: pd.DataFrame,
                          flow_col: str = "Flow_Pct",
                          entity_effects: bool = True,
                          controls: list[str] | None = None) -> dict | None:
    """Pure Python implementation using statsmodels with entity-demeaning."""
    required = ["LOWPERF", "MIDPERF", "HIGHPERF", flow_col, "ETF", "Date"]
    for col in required:
        if col not in df.columns:
            logger.warning("Missing column: %s", col)
            return None

    x_cols = ["LOWPERF", "MIDPERF", "HIGHPERF"]
    if controls:
        x_cols += [c for c in controls if c in df.columns]

    pdf = df[["ETF", "Date", flow_col] + x_cols].dropna()
    # Remove inf and extreme outliers (Flow_Pct > 100% is likely data error)
    pdf = pdf[np.isfinite(pdf[flow_col])]
    pdf = pdf[pdf[flow_col].abs() < 100]
    if len(pdf) < 30:
        return None

    y = pdf[flow_col]
    X = pdf[x_cols].copy()

    if entity_effects:
        # Within transformation (entity-demean)
        for col in x_cols:
            X[col] = X[col] - pdf.groupby("ETF")[col].transform("mean")
        y = y - pdf.groupby("ETF")[flow_col].transform("mean")

    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Entity-clustered SE via cov_type='cluster'
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": pdf["ETF"]},
        )

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    # Remove const row for cleaner output
    coef_df = coef_df[coef_df["Variable"] != "const"].reset_index(drop=True)

    # R² within (from demeaned regression)
    r2_within = model.rsquared

    # R² overall (re-estimate without demeaning)
    y_raw = pdf[flow_col]
    X_raw = sm.add_constant(pdf[x_cols])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_overall = sm.OLS(y_raw, X_raw).fit()
    r2_overall = model_overall.rsquared

    # Convexity test
    high_row = coef_df[coef_df["Variable"] == "HIGHPERF"]
    low_row = coef_df[coef_df["Variable"] == "LOWPERF"]
    convexity = {}
    if not high_row.empty and not low_row.empty:
        h_coef = float(high_row["Coefficient"].iloc[0])
        l_coef = float(low_row["Coefficient"].iloc[0])
        convexity = {
            "HIGHPERF_coef": h_coef,
            "LOWPERF_coef": l_coef,
            "ratio": h_coef / l_coef if abs(l_coef) > 1e-10 else np.nan,
            "convex": h_coef > l_coef,
        }

    return {
        "coefficients": coef_df,
        "r_squared_within": r2_within,
        "r_squared_overall": r2_overall,
        "n_obs": int(model.nobs),
        "n_entities": pdf["ETF"].nunique(),
        "convexity_test": convexity,
    }


def _sirri_tufano_r(df: pd.DataFrame,
                     flow_col: str = "Flow_Pct",
                     entity_effects: bool = True,
                     vcov: str = "cluster",
                     controls: list[str] | None = None) -> dict | None:
    """R/fixest implementation (preferred when R is available)."""
    required = ["LOWPERF", "MIDPERF", "HIGHPERF", flow_col, "ETF", "Date"]
    for col in required:
        if col not in df.columns:
            return None

    keep_cols = required.copy()
    rhs_vars = ["LOWPERF", "MIDPERF", "HIGHPERF"]
    if controls:
        for c in controls:
            if c in df.columns:
                keep_cols.append(c)
                rhs_vars.append(c)

    pdf = df[keep_cols].dropna()
    pdf = pdf[np.isfinite(pdf[flow_col])]
    pdf = pdf[pdf[flow_col].abs() < 100]
    if len(pdf) < 30:
        return None

    pdf = pdf.rename(columns={flow_col: "Flow"})
    pdf["DateInt"] = pd.factorize(pdf["Date"])[0]

    rhs = " + ".join(rhs_vars)
    fe_part = "ETF" if entity_effects else ""
    formula = f"Flow ~ {rhs} | {fe_part}" if fe_part else f"Flow ~ {rhs}"

    vcov_map = {"cluster": "~ETF", "twoway": "~ETF + DateInt",
                "DK": '"DK"', "hetero": '"hetero"'}
    vcov_r = vcov_map.get(vcov, "~ETF")

    try:
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        m <- feols({formula}, data=df_r, vcov={vcov_r}, panel.id=~ETF+DateInt)
        b <- coef(m); v <- vcov(m); s <- sqrt(diag(v))
        tv <- b / s; pv <- 2 * pnorm(abs(tv), lower.tail=FALSE)
        coefs <- data.frame(Variable=names(b), Coefficient=as.numeric(b),
            Std_Error=as.numeric(s), t_stat=as.numeric(tv), p_value=as.numeric(pv),
            stringsAsFactors=FALSE)
        r2w <- tryCatch(as.numeric(fitstat(m,"wr2")[[1]]), error=function(e) NA_real_)
        r2o <- tryCatch(as.numeric(fitstat(m,"r2")[[1]]), error=function(e) NA_real_)
        list(coefs=coefs, r2_within=r2w, r2_overall=r2o,
             nobs=as.integer(m$nobs), n_entities=as.integer(length(unique(df_r$ETF))))
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        coef_df = _r_to_py(result[0])
        _s = lambda rv: float(rv[0] if hasattr(rv, '__len__') else rv)
        r2w = _s(result[1])
        r2o = _s(result[2])
        if np.isnan(r2w):
            r2w = r2o

        high_row = coef_df[coef_df["Variable"] == "HIGHPERF"]
        low_row = coef_df[coef_df["Variable"] == "LOWPERF"]
        convexity = {}
        if not high_row.empty and not low_row.empty:
            h = float(high_row["Coefficient"].iloc[0])
            l = float(low_row["Coefficient"].iloc[0])
            convexity = {"HIGHPERF_coef": h, "LOWPERF_coef": l,
                         "ratio": h / l if abs(l) > 1e-10 else np.nan,
                         "convex": h > l}

        return {
            "coefficients": coef_df,
            "r_squared_within": r2w,
            "r_squared_overall": r2o,
            "n_obs": int(result[3][0]),
            "n_entities": int(result[4][0]),
            "convexity_test": convexity,
        }
    except Exception as e:
        logger.warning("R Sirri-Tufano regression failed: %s", e)
        return None


def sirri_tufano_regression(df: pd.DataFrame,
                             flow_col: str = "Flow_Pct",
                             entity_effects: bool = True,
                             vcov: str = "cluster",
                             controls: list[str] | None = None) -> dict | None:
    """Run Sirri-Tufano piecewise linear regression.

    Uses R/fixest if available, falls back to pure Python/statsmodels.

    Flow_it = alpha_i + beta1*LOWPERF + beta2*MIDPERF + beta3*HIGHPERF
              + gamma*Controls + epsilon

    Returns dict with:
        coefficients, r_squared_within, r_squared_overall,
        n_obs, n_entities, convexity_test
    """
    if R_AVAILABLE:
        result = _sirri_tufano_r(df, flow_col, entity_effects, vcov, controls)
        if result is not None:
            return result

    return _sirri_tufano_python(df, flow_col, entity_effects, controls)


def sirri_tufano_table(df: pd.DataFrame,
                       flow_col: str = "Flow_Pct",
                       controls_sequence: list[tuple[str, list[str]]] | None = None
                       ) -> pd.DataFrame:
    """Generate multi-column regression table with incremental controls.

    Parameters:
        controls_sequence: list of (column_name, control_vars) tuples.
            Example: [("(1) Base", []),
                      ("(2) + VIX", ["VIX_Close"])]

    Returns DataFrame with one row per variable, one column per specification.
    """
    if controls_sequence is None:
        controls_sequence = [("(1) Base", [])]

    results = {}
    for col_name, ctrl_list in controls_sequence:
        ctrl = ctrl_list if ctrl_list else None
        reg = sirri_tufano_regression(df, flow_col=flow_col, controls=ctrl)
        if reg is not None:
            results[col_name] = reg

    if not results:
        return pd.DataFrame()

    # Collect all variable names in order
    all_vars = []
    for spec_result in results.values():
        for _, row in spec_result["coefficients"].iterrows():
            if row["Variable"] not in all_vars:
                all_vars.append(row["Variable"])

    table_rows = []
    for var in all_vars:
        row = {"Variable": var}
        for col_name, reg in results.items():
            coef_row = reg["coefficients"][
                reg["coefficients"]["Variable"] == var
            ]
            if not coef_row.empty:
                coef = coef_row["Coefficient"].iloc[0]
                pval = coef_row["p_value"].iloc[0]
                se = coef_row["Std_Error"].iloc[0]
                stars = ("***" if pval < 0.01 else "**" if pval < 0.05
                         else "*" if pval < 0.1 else "")
                row[col_name] = f"{coef:.4f}{stars}"
                row[f"{col_name}_se"] = f"({se:.4f})"
            else:
                row[col_name] = ""
                row[f"{col_name}_se"] = ""
        table_rows.append(row)

    # Add fit statistics
    for stat_name, stat_key in [("R² within", "r_squared_within"),
                                 ("N", "n_obs"), ("ETFs", "n_entities")]:
        row = {"Variable": stat_name}
        for col_name, reg in results.items():
            val = reg.get(stat_key, "")
            if isinstance(val, float):
                row[col_name] = f"{val:.4f}"
            else:
                row[col_name] = str(val)
            row[f"{col_name}_se"] = ""
        table_rows.append(row)

    return pd.DataFrame(table_rows)
