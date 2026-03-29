"""R-based regression engine using fixest via rpy2.

Replaces the Python panel/OLS regressions with R's fixest package,
which provides faster estimation, built-in multi-way clustering,
and diagnostic tests in far fewer lines of code.

Usage:
    from r_engine import panel_feols, ols_by_etf, diagnostics
"""
import os
import logging
import warnings
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure R_HOME is set for Homebrew R
if not os.environ.get("R_HOME"):
    import subprocess
    try:
        r_home = subprocess.check_output(["R", "RHOME"], text=True).strip()
        os.environ["R_HOME"] = r_home
    except FileNotFoundError:
        logger.warning("R not found. R-based models will be unavailable.")

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    _CONVERTER = ro.default_converter + pandas2ri.converter
    ro.r("suppressPackageStartupMessages(library(fixest))")
    R_AVAILABLE = True
except Exception as e:
    logger.warning("rpy2/fixest not available: %s", e)
    R_AVAILABLE = False


def _py_to_r(df: pd.DataFrame):
    """Convert pandas DataFrame to R data.frame."""
    with localconverter(_CONVERTER):
        return ro.conversion.get_conversion().py2rpy(df)


def _r_to_py(r_obj):
    """Convert R object to Python."""
    with localconverter(_CONVERTER):
        return ro.conversion.get_conversion().rpy2py(r_obj)


def _prep_panel(df: pd.DataFrame, flow_col: str, return_col: str,
                lags: list[int],
                extra_controls: list[str] | None = None,
                cum_windows: list[int] | None = None) -> pd.DataFrame:
    """Prepare panel data: build lag columns, cumulative return windows,
    and select relevant columns.

    Parameters:
        cum_windows: rolling cumulative return windows (in periods).
            E.g. [5, 20, 60] builds CumRet_5, CumRet_20, CumRet_60
            where CumRet_W = rolling W-period cumulative return shifted by 1
            (so it uses only past information, no look-ahead).
    """
    pdf = df.copy()

    # Build lagged return columns
    for k in lags:
        pdf[f"Return_lag{k}"] = pdf.groupby("ETF")[return_col].shift(k)

    # Columns to keep
    keep = ["ETF", "Date", flow_col] + [f"Return_lag{k}" for k in lags]

    # Build cumulative return windows (shifted by 1 to avoid look-ahead)
    if cum_windows:
        for w in cum_windows:
            pdf[f"CumRet_{w}"] = pdf.groupby("ETF")[return_col].transform(
                lambda x: x.rolling(w, min_periods=max(1, w // 2)).sum()
            ).groupby(pdf["ETF"]).shift(1)
            keep.append(f"CumRet_{w}")

    # Add extra control columns if present
    if extra_controls:
        for col in extra_controls:
            if col in pdf.columns:
                keep.append(col)

    # Add excess return lags if available
    if "Excess_Return" in pdf.columns:
        for k in lags:
            pdf[f"Excess_lag{k}"] = pdf.groupby("ETF")["Excess_Return"].shift(k)
            keep.append(f"Excess_lag{k}")

    pdf = pdf[keep].dropna()
    # Rename flow column to 'Flow' for R formula simplicity
    pdf = pdf.rename(columns={flow_col: "Flow"})
    return pdf


# ============================================================
# Panel regressions via fixest::feols
# ============================================================

def panel_feols(df: pd.DataFrame, flow_col: str, return_col: str,
                lags: list[int] = [1],
                entity_effects: bool = True,
                time_effects: bool = False,
                vcov: str = "cluster",
                extra_controls: list[str] | None = None,
                add_excess: bool = False,
                add_volatility: bool = False,
                cum_windows: list[int] | None = None) -> dict | None:
    """Run panel regression using fixest::feols.

    Parameters:
        vcov: "cluster" (by ETF), "twoway" (ETF+Date), "DK" (Driscoll-Kraay),
              "iid" (no correction), "hetero" (HC1)
        extra_controls: columns to include as additional regressors
        add_excess: include lagged excess return
        add_volatility: include rolling volatility control
        cum_windows: rolling cumulative return windows (e.g. [5, 20, 60])

    Returns dict with: coefficients (DataFrame), r2_within, r2_overall,
    f_statistic, f_pvalue, n_obs, n_entities.
    """
    if not R_AVAILABLE:
        return None

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return None

    # Add volatility if requested
    if add_volatility:
        pdf["Volatility"] = pdf.groupby("ETF")["Return_lag1"].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )
        pdf = pdf.dropna(subset=["Volatility"])

    # Build formula
    lag_vars = [f"Return_lag{k}" for k in lags]
    rhs_vars = list(lag_vars)

    # Add cumulative return windows
    if cum_windows:
        rhs_vars += [f"CumRet_{w}" for w in cum_windows if f"CumRet_{w}" in pdf.columns]

    if add_excess:
        excess_vars = [c for c in pdf.columns if c.startswith("Excess_lag")]
        rhs_vars += excess_vars

    if add_volatility:
        rhs_vars.append("Volatility")

    if extra_controls:
        for col in extra_controls:
            if col in pdf.columns:
                rhs_vars.append(col)

    rhs = " + ".join(rhs_vars)

    # Fixed effects
    if entity_effects and time_effects:
        fe_part = "ETF + Date"
    elif entity_effects:
        fe_part = "ETF"
    elif time_effects:
        fe_part = "Date"
    else:
        fe_part = None

    if fe_part:
        formula = f"Flow ~ {rhs} | {fe_part}"
    else:
        formula = f"Flow ~ {rhs}"

    # Variance-covariance
    vcov_map = {
        "cluster": "~ETF",
        "twoway": "~ETF + Date",
        "DK": '"DK"',
        "iid": '"iid"',
        "hetero": '"hetero"',
    }
    vcov_r = vcov_map.get(vcov, "~ETF")

    try:
        # Convert Date to integer for fixest panel.id (R Date objects can cause issues)
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)

        r_code = f'''
        m <- feols({formula}, data=df_r, vcov={vcov_r}, panel.id=~ETF+DateInt)
        b <- coef(m)
        v <- vcov(m)
        s <- sqrt(diag(v))
        tv <- b / s
        pv <- 2 * pnorm(abs(tv), lower.tail=FALSE)
        coefs <- data.frame(
            Variable = names(b),
            Coefficient = as.numeric(b),
            Std_Error = as.numeric(s),
            t_stat = as.numeric(tv),
            p_value = as.numeric(pv),
            stringsAsFactors = FALSE
        )
        r2w <- tryCatch(as.numeric(fitstat(m, "wr2")[[1]]), error=function(e) NA_real_)
        r2o <- tryCatch(as.numeric(fitstat(m, "r2")[[1]]), error=function(e) NA_real_)
        f_stat <- tryCatch(as.numeric(fitstat(m, "wf")[[1]]), error=function(e) NA_real_)
        f_pval <- tryCatch({{
            fo <- fitstat(m, "wf")
            # fitstat returns a list; extract the p-value from the F distribution
            fval <- as.numeric(fo[[1]])
            k <- length(coef(m))
            n <- m$nobs
            as.numeric(pf(fval, k, n - k, lower.tail=FALSE))
        }}, error=function(e) NA_real_)
        list(
            coefs = coefs,
            r2_within = r2w,
            r2_overall = r2o,
            f_stat = f_stat,
            f_pval = f_pval,
            nobs = as.integer(m$nobs),
            n_entities = as.integer(length(unique(df_r$ETF)))
        )
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        coef_df = _r_to_py(result[0])

        def _scalar(rv):
            """Extract scalar from R vector."""
            v = rv[0] if hasattr(rv, '__len__') else rv
            return float(v)

        r2w = _scalar(result[1])
        r2o = _scalar(result[2])
        if np.isnan(r2w):
            r2w = r2o

        return {
            "coefficients": coef_df,
            "r_squared_within": r2w,
            "r_squared_overall": r2o,
            "f_statistic": _scalar(result[3]),
            "f_pvalue": _scalar(result[4]),
            "n_obs": int(result[5][0]),
            "n_entities": int(result[6][0]),
        }

    except Exception as e:
        logger.warning("panel_feols failed: %s", e)
        return None


def panel_feols_comparison(df: pd.DataFrame, flow_col: str, return_col: str,
                           lags: list[int] = [1],
                           extra_controls: list[str] | None = None,
                           cum_windows: list[int] | None = None) -> pd.DataFrame:
    """Run multiple panel specifications side by side using fixest.

    Returns summary DataFrame comparing Pooled, Entity FE, Entity+Time FE,
    Entity FE + Excess, Entity FE + Controls, plus Driscoll-Kraay and
    two-way clustered SE variants.
    """
    specs = [
        ("Pooled OLS", dict(entity_effects=False, time_effects=False, vcov="hetero")),
        ("Entity FE", dict(entity_effects=True, time_effects=False, vcov="cluster")),
        ("Entity+Time FE", dict(entity_effects=True, time_effects=True, vcov="cluster")),
        ("Entity FE + Excess", dict(entity_effects=True, time_effects=False,
                                     vcov="cluster", add_excess=True)),
        ("Entity FE + Controls", dict(entity_effects=True, time_effects=False,
                                       vcov="cluster", add_volatility=True)),
        ("Entity FE (DK SE)", dict(entity_effects=True, time_effects=False, vcov="DK")),
        ("Entity FE (2way cluster)", dict(entity_effects=True, time_effects=False, vcov="twoway")),
    ]

    rows = []
    for name, kwargs in specs:
        result = panel_feols(df, flow_col, return_col, lags=lags,
                             extra_controls=extra_controls,
                             cum_windows=cum_windows, **kwargs)
        if result is None:
            continue
        row = {
            "Specification": name,
            "R²_within": result["r_squared_within"],
            "R²_overall": result["r_squared_overall"],
            "F_stat": result["f_statistic"],
            "F_pval": result["f_pvalue"],
            "N": result["n_obs"],
            "Entities": result["n_entities"],
        }
        for _, cr in result["coefficients"].iterrows():
            if "Return_lag" in str(cr["Variable"]):
                row[f"{cr['Variable']}_coef"] = cr["Coefficient"]
                row[f"{cr['Variable']}_pval"] = cr["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Per-ETF OLS via fixest (vectorized across ETFs)
# ============================================================

def ols_by_etf(df: pd.DataFrame, flow_col: str, return_col: str,
               lags: list[int] = [1],
               extra_controls: list[str] | None = None,
               cum_windows: list[int] | None = None) -> pd.DataFrame:
    """Run per-ETF OLS using fixest's split estimation.

    Equivalent to running lag_regression for each ETF, but done in one
    fixest call using the `split` argument.

    Returns DataFrame: ETF, R², Return_lag1 coefficient, p-value, N.
    """
    if not R_AVAILABLE:
        return pd.DataFrame()

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return pd.DataFrame()

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]

    rhs = " + ".join(lag_vars + cum_vars + ctrl_vars)

    try:
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        models <- feols(Flow ~ {rhs}, data=df_r, fsplit=~ETF, vcov="hetero")
        n_models <- length(models)
        results <- lapply(seq_len(n_models), function(i) {{
            m <- models[[i]]
            etf_name <- names(models)[i]
            b <- coef(m)
            v <- vcov(m)
            s <- sqrt(diag(v))
            tv <- b / s
            pv <- 2 * pnorm(abs(tv), lower.tail=FALSE)
            data.frame(
                ETF = etf_name,
                R2 = as.numeric(fitstat(m, "r2")[[1]]),
                Return_lag1 = as.numeric(b["Return_lag1"]),
                Return_lag1_pval = as.numeric(pv["Return_lag1"]),
                N = as.integer(m$nobs),
                stringsAsFactors = FALSE
            )
        }})
        do.call(rbind, results)
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        out = _r_to_py(result)
        # Clean ETF names: fixest prepends "sample.var: ETF; sample: "
        out["ETF"] = out["ETF"].str.replace(
            r"^sample\.var:.*?sample:\s*", "", regex=True
        )
        return out

    except Exception as e:
        logger.warning("ols_by_etf failed: %s", e)
        return pd.DataFrame()


# ============================================================
# Asymmetry regression via fixest
# ============================================================

def asymmetry_feols(df: pd.DataFrame, flow_col: str, return_col: str,
                    lags: list[int] = [1],
                    extra_controls: list[str] | None = None,
                    cum_windows: list[int] | None = None) -> dict | None:
    """Asymmetry regression: Flow ~ Return_pos + Return_neg [+ controls] | ETF

    Returns dict with beta_pos, beta_neg, asymmetry_ratio, wald_p, r2, n_obs.
    """
    if not R_AVAILABLE:
        return None

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return None

    # Create pos/neg return columns
    for k in lags:
        col = f"Return_lag{k}"
        pdf[f"Ret_pos_lag{k}"] = pdf[col].clip(lower=0)
        pdf[f"Ret_neg_lag{k}"] = pdf[col].clip(upper=0)

    pos_vars = [f"Ret_pos_lag{k}" for k in lags]
    neg_vars = [f"Ret_neg_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]

    rhs = " + ".join(pos_vars + neg_vars + cum_vars + ctrl_vars)

    try:
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        m <- feols(Flow ~ {rhs} | ETF, data=df_r, vcov=~ETF, panel.id=~ETF+DateInt)
        b <- coef(m)
        s <- sqrt(diag(vcov(m)))
        tv <- b / s
        pv <- 2 * pt(abs(tv), df=m$nobs - length(b), lower.tail=FALSE)
        coefs <- data.frame(
            Variable = names(b),
            Coefficient = as.numeric(b),
            Std_Error = as.numeric(s),
            t_stat = as.numeric(tv),
            p_value = as.numeric(pv),
            stringsAsFactors = FALSE
        )

        pos_names <- grep("pos", names(coef(m)), value=TRUE)
        neg_names <- grep("neg", names(coef(m)), value=TRUE)
        beta_pos <- mean(coef(m)[pos_names])
        beta_neg <- mean(coef(m)[neg_names])

        # Wald test: H0: sum of pos + sum of neg = 0
        wald_p <- tryCatch({{
            constraint <- paste(paste(pos_names, collapse=" + "), "+",
                              paste(neg_names, collapse=" + "), "= 0")
            w <- wald(m, constraint)
            w$p
        }}, error=function(e) NA)

        list(
            coefs = coefs,
            beta_pos = beta_pos,
            beta_neg = beta_neg,
            wald_p = wald_p,
            r2 = fitstat(m, "wr2")[[1]],
            nobs = m$nobs
        )
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        beta_pos = float(result[1][0])
        beta_neg = float(result[2][0])
        asym_ratio = beta_pos / abs(beta_neg) if abs(beta_neg) > 1e-10 else np.nan

        return {
            "coefficients": _r_to_py(result[0]),
            "beta_pos": beta_pos,
            "beta_neg": beta_neg,
            "asymmetry_ratio": asym_ratio,
            "wald_p": float(result[3][0]),
            "r_squared_within": float(result[4][0]),
            "n_obs": int(result[5][0]),
        }

    except Exception as e:
        logger.warning("asymmetry_feols failed: %s", e)
        return None


# ============================================================
# Diagnostic tests
# ============================================================

def diagnostic_tests(df: pd.DataFrame, flow_col: str, return_col: str,
                     lags: list[int] = [1],
                     extra_controls: list[str] | None = None,
                     cum_windows: list[int] | None = None) -> dict:
    """Run panel diagnostic tests (6 tests).

    Returns dict with:
        breusch_pagan: {statistic, p_value} — heteroskedasticity
        hausman: {statistic, p_value} — FE vs RE
        f_test_individual_effects: {statistic, p_value}
        serial_correlation: {statistic, p_value} — Breusch-Godfrey
        cross_sectional_dependence: {statistic, p_value} — Pesaran CD
        bp_lm_test: {statistic, p_value} — pooled vs panel
    """
    if not R_AVAILABLE:
        return {}

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 50:
        return {}

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]
    rhs = " + ".join(lag_vars + cum_vars + ctrl_vars)

    try:
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        library(plm)
        library(lmtest)

        pdata <- pdata.frame(df_r, index=c("ETF","Date"))

        # Panel FE model
        fe_model <- plm(Flow ~ {rhs}, data=pdata, model="within")
        re_model <- plm(Flow ~ {rhs}, data=pdata, model="random")
        pool_model <- plm(Flow ~ {rhs}, data=pdata, model="pooling")

        # Breusch-Pagan test for heteroskedasticity
        bp <- tryCatch({{
            t <- bptest(fe_model)
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        # Hausman test: FE vs RE
        haus <- tryCatch({{
            t <- phtest(fe_model, re_model)
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        # F-test for individual effects
        f_test <- tryCatch({{
            t <- pFtest(fe_model, pool_model)
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        # Serial correlation (Breusch-Godfrey)
        bg <- tryCatch({{
            t <- pbgtest(fe_model, order=1)
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        # Cross-sectional dependence (Pesaran CD)
        cd <- tryCatch({{
            t <- pcdtest(fe_model, test="cd")
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        # Breusch-Pagan LM test (Pooled vs Panel)
        bp_lm <- tryCatch({{
            t <- plmtest(pool_model, type="bp")
            list(stat=as.numeric(t$statistic), p=as.numeric(t$p.value))
        }}, error=function(e) list(stat=NA_real_, p=NA_real_))

        list(
            bp_stat = bp$stat, bp_p = bp$p,
            hausman_stat = haus$stat, hausman_p = haus$p,
            f_test_stat = f_test$stat, f_test_p = f_test$p,
            bg_stat = bg$stat, bg_p = bg$p,
            cd_stat = cd$stat, cd_p = cd$p,
            bp_lm_stat = bp_lm$stat, bp_lm_p = bp_lm$p
        )
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        def _safe(idx):
            try:
                return float(result[idx][0])
            except (IndexError, TypeError):
                return np.nan

        return {
            "breusch_pagan": {
                "statistic": _safe(0),
                "p_value": _safe(1),
            },
            "hausman": {
                "statistic": _safe(2),
                "p_value": _safe(3),
            },
            "f_test_individual_effects": {
                "statistic": _safe(4),
                "p_value": _safe(5),
            },
            "serial_correlation": {
                "statistic": _safe(6),
                "p_value": _safe(7),
            },
            "cross_sectional_dependence": {
                "statistic": _safe(8),
                "p_value": _safe(9),
            },
            "bp_lm_test": {
                "statistic": _safe(10),
                "p_value": _safe(11),
            },
        }

    except Exception as e:
        logger.warning("diagnostic_tests failed: %s", e)
        return {}


def variance_decomposition(df: pd.DataFrame,
                           columns: list[str]) -> pd.DataFrame:
    """Between/within variance decomposition (equivalent to Stata's xtsum).

    Parameters:
        df: Panel DataFrame with 'ETF' column.
        columns: Variable names to decompose.

    Returns DataFrame with columns: Variable, overall_sd, between_sd,
        within_sd, within_pct.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        x = df[col].dropna()
        if len(x) == 0:
            continue
        overall_sd = x.std()
        grp_means = df.groupby("ETF")[col].mean()
        between_sd = grp_means.std()
        within_vals = df.groupby("ETF")[col].transform(lambda g: g - g.mean())
        within_sd = within_vals.std()
        denom = within_sd**2 + between_sd**2
        within_pct = 100 * within_sd**2 / denom if denom > 0 else np.nan
        rows.append({
            "Variable": col,
            "overall_sd": overall_sd,
            "between_sd": between_sd,
            "within_sd": within_sd,
            "within_pct": within_pct,
        })
    return pd.DataFrame(rows)


def panel_unit_root(df: pd.DataFrame, flow_col: str,
                    return_col: str) -> dict:
    """Panel unit root tests (Im-Pesaran-Shin) on flow and return.

    Returns dict: {flow_col: {statistic, p_value}, return_col: {statistic, p_value}}
    """
    if not R_AVAILABLE:
        return {}

    pdf = df[["ETF", "Date", flow_col, return_col]].dropna().copy()
    pdf = pdf.rename(columns={flow_col: "Flow", return_col: "Ret"})

    try:
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = '''
        library(plm)
        pdata <- pdata.frame(df_r, index=c("ETF","Date"))

        ips_flow <- tryCatch({
            t <- purtest(pdata$Flow, test="ips", exo="intercept", lags="AIC", pmax=5)
            list(stat=as.numeric(t$statistic$statistic),
                 p=as.numeric(t$statistic$p.value))
        }, error=function(e) list(stat=NA_real_, p=NA_real_))

        ips_ret <- tryCatch({
            t <- purtest(pdata$Ret, test="ips", exo="intercept", lags="AIC", pmax=5)
            list(stat=as.numeric(t$statistic$statistic),
                 p=as.numeric(t$statistic$p.value))
        }, error=function(e) list(stat=NA_real_, p=NA_real_))

        list(flow_stat=ips_flow$stat, flow_p=ips_flow$p,
             ret_stat=ips_ret$stat, ret_p=ips_ret$p)
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        def _safe(idx):
            try:
                return float(result[idx][0])
            except (IndexError, TypeError):
                return np.nan

        return {
            flow_col: {"statistic": _safe(0), "p_value": _safe(1)},
            return_col: {"statistic": _safe(2), "p_value": _safe(3)},
        }

    except Exception as e:
        logger.warning("panel_unit_root failed: %s", e)
        return {}


# ============================================================
# Multi-spec summary (fixest etable equivalent)
# ============================================================

def multi_spec_summary(df: pd.DataFrame, flow_col: str, return_col: str,
                       lags: list[int] = [1],
                       extra_controls: list[str] | None = None,
                       cum_windows: list[int] | None = None) -> str:
    """Run all key specifications and return a formatted summary string,
    similar to fixest::etable output.
    """
    if not R_AVAILABLE:
        return "R not available"

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return "Not enough data"

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]
    rhs = " + ".join(lag_vars + cum_vars + ctrl_vars)

    try:
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        m1 <- feols(Flow ~ {rhs}, data=df_r, vcov="hetero", panel.id=~ETF+DateInt)
        m2 <- feols(Flow ~ {rhs} | ETF, data=df_r, vcov=~ETF, panel.id=~ETF+DateInt)
        m3 <- feols(Flow ~ {rhs} | ETF + DateInt, data=df_r, vcov=~ETF, panel.id=~ETF+DateInt)
        m4 <- feols(Flow ~ {rhs} | ETF, data=df_r, vcov=~ETF+DateInt, panel.id=~ETF+DateInt)
        m5 <- feols(Flow ~ {rhs} | ETF, data=df_r, vcov="DK", panel.id=~ETF+DateInt)
        capture.output(etable(m1, m2, m3, m4, m5,
                              headers=c("Pooled","Entity FE","Entity+Time FE",
                                        "2-way Cluster","Driscoll-Kraay"),
                              fitstat=c("r2","wr2","n")))
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        return "\n".join(list(result))

    except Exception as e:
        logger.warning("multi_spec_summary failed: %s", e)
        return f"Error: {e}"


# ============================================================
# GMM estimation (Arellano-Bond / Blundell-Bond)
# ============================================================

def panel_gmm(df: pd.DataFrame, flow_col: str, return_col: str,
              lags: list[int] = [1],
              extra_controls: list[str] | None = None,
              cum_windows: list[int] | None = None,
              transformation: str = "d") -> dict | None:
    """Dynamic panel GMM via plm::pgmm.

    Parameters:
        transformation: "d" for Arellano-Bond (first differences),
                        "ld" for Blundell-Bond (system GMM).

    Returns dict with: coefficients (DataFrame), sargan (stat, p),
        ar1 (stat, p), ar2 (stat, p), n_obs, n_entities.
    """
    if not R_AVAILABLE:
        return None

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 50:
        return None

    # GMM needs Flow_lag1 as dependent variable lag
    pdf["Flow_lag1"] = pdf.groupby("ETF")["Flow"].shift(1)
    pdf = pdf.dropna(subset=["Flow_lag1"])

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]
    exog_rhs = " + ".join(lag_vars + cum_vars + ctrl_vars)

    try:
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        library(plm)
        pdata <- pdata.frame(df_r, index=c("ETF","Date"))

        gmm_mod <- tryCatch({{
            pgmm(Flow ~ lag(Flow, 1) + {exog_rhs} | lag(Flow, 2:5),
                  data=pdata, effect="individual", model="onestep",
                  transformation="{transformation}")
        }}, error=function(e) NULL)

        if (is.null(gmm_mod)) {{
            list(ok=FALSE)
        }} else {{
            s <- summary(gmm_mod)
            coefs <- s$coefficients
            coef_df <- data.frame(
                Variable = rownames(coefs),
                Coefficient = as.numeric(coefs[,1]),
                Std_Error = as.numeric(coefs[,2]),
                t_stat = as.numeric(coefs[,3]),
                p_value = as.numeric(coefs[,4]),
                stringsAsFactors = FALSE
            )

            # Sargan test
            sargan_stat <- tryCatch(as.numeric(s$sargan$statistic), error=function(e) NA_real_)
            sargan_p <- tryCatch(as.numeric(s$sargan$p.value), error=function(e) NA_real_)

            # AR tests
            ar1_stat <- tryCatch(as.numeric(s$m1$statistic), error=function(e) NA_real_)
            ar1_p <- tryCatch(as.numeric(s$m1$p.value), error=function(e) NA_real_)
            ar2_stat <- tryCatch(as.numeric(s$m2$statistic), error=function(e) NA_real_)
            ar2_p <- tryCatch(as.numeric(s$m2$p.value), error=function(e) NA_real_)

            list(ok=TRUE, coefs=coef_df,
                 sargan_stat=sargan_stat, sargan_p=sargan_p,
                 ar1_stat=ar1_stat, ar1_p=ar1_p,
                 ar2_stat=ar2_stat, ar2_p=ar2_p,
                 nobs=as.integer(length(gmm_mod$residuals)),
                 n_entities=as.integer(length(unique(df_r$ETF))))
        }}
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        ok = bool(result[0][0])
        if not ok:
            logger.warning("panel_gmm: pgmm returned NULL (transformation=%s)",
                           transformation)
            return None

        coef_df = _r_to_py(result[1])

        def _s(idx):
            try:
                return float(result[idx][0])
            except (IndexError, TypeError):
                return np.nan

        return {
            "coefficients": coef_df,
            "sargan": {"statistic": _s(2), "p_value": _s(3)},
            "ar1": {"statistic": _s(4), "p_value": _s(5)},
            "ar2": {"statistic": _s(6), "p_value": _s(7)},
            "n_obs": int(_s(8)) if not np.isnan(_s(8)) else 0,
            "n_entities": int(_s(9)) if not np.isnan(_s(9)) else 0,
            "transformation": transformation,
        }

    except Exception as e:
        logger.warning("panel_gmm failed: %s", e)
        return None


# ============================================================
# Cluster bootstrap (pairs cluster, resampling ETFs)
# ============================================================

def cluster_bootstrap(df: pd.DataFrame, flow_col: str, return_col: str,
                      lags: list[int] = [1],
                      extra_controls: list[str] | None = None,
                      cum_windows: list[int] | None = None,
                      n_boot: int = 999) -> dict | None:
    """Pairs cluster bootstrap — resample entire ETFs.

    Returns dict with: original_beta, boot_se, boot_t, boot_p,
        ci_normal (tuple), ci_percentile (tuple).
    """
    if not R_AVAILABLE:
        return None

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 50:
        return None

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]
    rhs = " + ".join(lag_vars + cum_vars + ctrl_vars)

    try:
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)
        r_code = f'''
        library(boot)

        boot_panel <- function(data, indices) {{
            etfs <- unique(data$ETF)
            selected <- etfs[indices]
            boot_data <- do.call(rbind, lapply(seq_along(selected), function(i) {{
                d <- data[data$ETF == selected[i], ]
                d$ETF <- paste0("ETF_", i)
                d
            }}))
            tryCatch({{
                m <- feols(Flow ~ {rhs} | ETF, data=boot_data, vcov="iid")
                coef(m)["Return_lag1"]
            }}, error=function(e) NA)
        }}

        set.seed(42)
        br <- boot(df_r, boot_panel, R={n_boot}, sim="ordinary", stype="i")

        boot_se <- sd(br$t, na.rm=TRUE)
        ci <- tryCatch(boot.ci(br, type=c("norm","perc")), error=function(e) NULL)

        ci_norm_lo <- if (!is.null(ci) && !is.null(ci$normal)) ci$normal[2] else NA_real_
        ci_norm_hi <- if (!is.null(ci) && !is.null(ci$normal)) ci$normal[3] else NA_real_
        ci_perc_lo <- if (!is.null(ci) && !is.null(ci$percent)) ci$percent[4] else NA_real_
        ci_perc_hi <- if (!is.null(ci) && !is.null(ci$percent)) ci$percent[5] else NA_real_

        list(
            beta = as.numeric(br$t0),
            boot_se = as.numeric(boot_se),
            boot_t = as.numeric(br$t0 / boot_se),
            boot_p = as.numeric(2 * pnorm(-abs(br$t0 / boot_se))),
            ci_norm_lo = as.numeric(ci_norm_lo),
            ci_norm_hi = as.numeric(ci_norm_hi),
            ci_perc_lo = as.numeric(ci_perc_lo),
            ci_perc_hi = as.numeric(ci_perc_hi)
        )
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        def _s(idx):
            try:
                return float(result[idx][0])
            except (IndexError, TypeError):
                return np.nan

        return {
            "original_beta": _s(0),
            "boot_se": _s(1),
            "boot_t": _s(2),
            "boot_p": _s(3),
            "ci_normal": (_s(4), _s(5)),
            "ci_percentile": (_s(6), _s(7)),
            "n_boot": n_boot,
        }

    except Exception as e:
        logger.warning("cluster_bootstrap failed: %s", e)
        return None


# ============================================================
# Interacted FE (ETF-specific time trends)
# ============================================================

def panel_feols_trend(df: pd.DataFrame, flow_col: str, return_col: str,
                      lags: list[int] = [1],
                      extra_controls: list[str] | None = None,
                      cum_windows: list[int] | None = None,
                      vcov: str = "cluster") -> dict | None:
    """Panel regression with ETF-specific linear time trends.

    fixest syntax: feols(Flow ~ ... | ETF[DateInt])

    Returns same dict structure as panel_feols.
    """
    if not R_AVAILABLE:
        return None

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return None

    lag_vars = [f"Return_lag{k}" for k in lags]
    rhs_vars = list(lag_vars)
    if cum_windows:
        rhs_vars += [f"CumRet_{w}" for w in cum_windows if f"CumRet_{w}" in pdf.columns]
    if extra_controls:
        rhs_vars += [col for col in extra_controls if col in pdf.columns]
    rhs = " + ".join(rhs_vars)

    vcov_map = {
        "cluster": "~ETF",
        "twoway": "~ETF + DateInt",
        "DK": '"DK"',
        "iid": '"iid"',
        "hetero": '"hetero"',
    }
    vcov_r = vcov_map.get(vcov, "~ETF")

    try:
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)

        r_code = f'''
        m <- feols(Flow ~ {rhs} | ETF[DateInt], data=df_r, vcov={vcov_r},
                   panel.id=~ETF+DateInt)
        b <- coef(m)
        v <- vcov(m)
        s <- sqrt(diag(v))
        tv <- b / s
        pv <- 2 * pnorm(abs(tv), lower.tail=FALSE)
        coefs <- data.frame(
            Variable = names(b),
            Coefficient = as.numeric(b),
            Std_Error = as.numeric(s),
            t_stat = as.numeric(tv),
            p_value = as.numeric(pv),
            stringsAsFactors = FALSE
        )
        r2w <- tryCatch(as.numeric(fitstat(m, "wr2")[[1]]), error=function(e) NA_real_)
        r2o <- tryCatch(as.numeric(fitstat(m, "r2")[[1]]), error=function(e) NA_real_)
        f_stat <- tryCatch(as.numeric(fitstat(m, "wf")[[1]]), error=function(e) NA_real_)
        f_pval <- tryCatch({{
            fval <- as.numeric(fitstat(m, "wf")[[1]])
            k <- length(coef(m))
            n <- m$nobs
            as.numeric(pf(fval, k, n - k, lower.tail=FALSE))
        }}, error=function(e) NA_real_)
        list(
            coefs = coefs,
            r2_within = r2w,
            r2_overall = r2o,
            f_stat = f_stat,
            f_pval = f_pval,
            nobs = as.integer(m$nobs),
            n_entities = as.integer(length(unique(df_r$ETF)))
        )
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        coef_df = _r_to_py(result[0])

        def _scalar(rv):
            v = rv[0] if hasattr(rv, '__len__') else rv
            return float(v)

        r2w = _scalar(result[1])
        r2o = _scalar(result[2])
        if np.isnan(r2w):
            r2w = r2o

        return {
            "coefficients": coef_df,
            "r_squared_within": r2w,
            "r_squared_overall": r2o,
            "f_statistic": _scalar(result[3]),
            "f_pvalue": _scalar(result[4]),
            "n_obs": int(result[5][0]),
            "n_entities": int(result[6][0]),
        }

    except Exception as e:
        logger.warning("panel_feols_trend failed: %s", e)
        return None


# ============================================================
# LaTeX table export via fixest::etable
# ============================================================

def export_latex_table(df: pd.DataFrame, flow_col: str, return_col: str,
                       lags: list[int] = [1],
                       extra_controls: list[str] | None = None,
                       cum_windows: list[int] | None = None,
                       output_path: str = "table_main_results.tex") -> str:
    """Generate publication-quality LaTeX table with multiple specs.

    Returns the LaTeX string (also writes to output_path).
    """
    if not R_AVAILABLE:
        return ""

    pdf = _prep_panel(df, flow_col, return_col, lags, extra_controls,
                      cum_windows=cum_windows)
    if len(pdf) < 20:
        return ""

    # Build Flow_lag1 for the dynamic spec
    pdf["Flow_lag1"] = pdf.groupby("ETF")["Flow"].shift(1)
    pdf = pdf.dropna(subset=["Flow_lag1"])

    lag_vars = [f"Return_lag{k}" for k in lags]
    cum_vars = [f"CumRet_{w}" for w in (cum_windows or []) if f"CumRet_{w}" in pdf.columns]
    ctrl_vars = []
    if extra_controls:
        ctrl_vars = [col for col in extra_controls if col in pdf.columns]
    rhs_base = " + ".join(lag_vars + ctrl_vars)
    rhs_cum = " + ".join(lag_vars + cum_vars + ctrl_vars) if cum_vars else rhs_base
    rhs_dyn = " + ".join(lag_vars + cum_vars + ["Flow_lag1"] + ctrl_vars) if cum_vars else rhs_base + " + Flow_lag1"

    try:
        pdf["DateInt"] = pd.factorize(pdf["Date"])[0]
        ro.globalenv["df_r"] = _py_to_r(pdf)
        ro.globalenv["tex_path"] = output_path

        r_code = f'''
        m1 <- feols(Flow ~ {rhs_base}, data=df_r, vcov="hetero",
                    panel.id=~ETF+DateInt)
        m2 <- feols(Flow ~ {rhs_base} | ETF, data=df_r, vcov=~ETF,
                    panel.id=~ETF+DateInt)
        m3 <- feols(Flow ~ {rhs_cum} | ETF, data=df_r, vcov=~ETF,
                    panel.id=~ETF+DateInt)
        m4 <- feols(Flow ~ {rhs_dyn} | ETF, data=df_r, vcov=~ETF,
                    panel.id=~ETF+DateInt)
        m5 <- feols(Flow ~ {rhs_cum} | ETF, data=df_r, vcov="DK",
                    panel.id=~ETF+DateInt)
        m6 <- feols(Flow ~ {rhs_cum} | ETF[DateInt], data=df_r, vcov=~ETF,
                    panel.id=~ETF+DateInt)

        etable(m1, m2, m3, m4, m5, m6,
               headers=c("Pooled","Entity FE","+ CumRet","Dynamic","DK SE","Trend FE"),
               fitstat=c("r2","wr2","n"),
               se.below=TRUE,
               signif.code=c("***"=0.01, "**"=0.05, "*"=0.10),
               file=tex_path, replace=TRUE)

        paste(readLines(tex_path), collapse="\n")
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ro.r(r_code)

        return str(result[0]) if result else ""

    except Exception as e:
        logger.warning("export_latex_table failed: %s", e)
        return ""
