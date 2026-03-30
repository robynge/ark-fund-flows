"""Summary statistics for panel data (Table 1 of the paper)."""
import pandas as pd
import numpy as np


def summary_statistics(df: pd.DataFrame, variables: list[str],
                       groupby: str | None = None) -> pd.DataFrame:
    """Compute descriptive statistics for specified variables.

    Returns DataFrame with columns: Variable, N, Mean, SD, Min, P25, P50, P75, Max.
    If groupby is specified, adds a Group column.
    """
    def _stats(series: pd.Series) -> dict:
        s = series.dropna()
        return {
            "N": len(s),
            "Mean": s.mean(),
            "SD": s.std(),
            "Min": s.min(),
            "P25": s.quantile(0.25),
            "P50": s.median(),
            "P75": s.quantile(0.75),
            "Max": s.max(),
        }

    if groupby is None:
        rows = []
        for var in variables:
            if var in df.columns:
                row = {"Variable": var, **_stats(df[var])}
                rows.append(row)
        return pd.DataFrame(rows)

    rows = []
    for group_val, group_df in df.groupby(groupby):
        for var in variables:
            if var in group_df.columns:
                row = {"Group": group_val, "Variable": var,
                       **_stats(group_df[var])}
                rows.append(row)
    return pd.DataFrame(rows)


def panel_summary(df: pd.DataFrame, flow_col: str, return_col: str,
                  extra_vars: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Compute full panel summary: overall + by-ETF tables.

    Returns {'overall': DataFrame, 'by_etf': DataFrame,
             'between_within': DataFrame}.
    """
    variables = [flow_col, return_col]
    if extra_vars:
        variables += [v for v in extra_vars if v in df.columns]

    overall = summary_statistics(df, variables)
    by_etf = summary_statistics(df, variables, groupby="ETF")

    # Between/within decomposition
    bw_rows = []
    for var in variables:
        if var not in df.columns:
            continue
        overall_sd = df[var].std()
        etf_means = df.groupby("ETF")[var].mean()
        between_sd = etf_means.std()
        within_vals = df[var] - df.groupby("ETF")[var].transform("mean")
        within_sd = within_vals.std()
        total_var = overall_sd ** 2 if overall_sd > 0 else 1e-10
        within_pct = (within_sd ** 2 / total_var) * 100

        bw_rows.append({
            "Variable": var,
            "Overall_SD": overall_sd,
            "Between_SD": between_sd,
            "Within_SD": within_sd,
            "Within_Pct": within_pct,
        })

    between_within = pd.DataFrame(bw_rows)

    return {
        "overall": overall,
        "by_etf": by_etf,
        "between_within": between_within,
    }


def correlation_matrix(df: pd.DataFrame,
                       variables: list[str]) -> pd.DataFrame:
    """Compute pairwise Pearson correlation matrix for specified variables."""
    cols = [v for v in variables if v in df.columns]
    return df[cols].corr()


def to_latex_summary(stats_df: pd.DataFrame, caption: str = "Summary Statistics",
                     label: str = "tab:summary") -> str:
    """Convert summary statistics DataFrame to LaTeX table string."""
    lines = []
    lines.append(r"\begin{table}[htbp]\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    n_cols = len(stats_df.columns)
    col_fmt = "l" + "r" * (n_cols - 1)
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join(str(c) for c in stats_df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for _, row in stats_df.iterrows():
        cells = []
        for i, val in enumerate(row):
            if i == 0:
                cells.append(str(val))
            elif isinstance(val, (int, np.integer)):
                cells.append(f"{val:,}")
            elif isinstance(val, (float, np.floating)):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}\small")
    lines.append(r"\item Notes: N = number of observations, SD = standard deviation,")
    lines.append(r"P25/P50/P75 = 25th/50th/75th percentiles.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
