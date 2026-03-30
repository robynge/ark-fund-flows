"""Generate publication-quality LaTeX tables for the paper."""
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS = Path(__file__).parent.parent / "experiments" / "results_v2"
TABLES = RESULTS / "tables"
TABLES.mkdir(exist_ok=True)


def _stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "^{***}"
    if p < 0.05: return "^{**}"
    if p < 0.10: return "^{*}"
    return ""


def _fmt_coef(coef, p, se=None):
    """Format coefficient with stars and SE in parentheses."""
    s = f"${coef:.2f}{_stars(p)}$"
    if se is not None:
        s += f"\n& $({se:.2f})$"
    return s


# ============================================================
# Table 1: Summary Statistics
# ============================================================

def table_1():
    overall = pd.read_csv(RESULTS / "table_1_summary.csv")
    bw = pd.read_csv(RESULTS / "table_1_bw.csv")

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Summary Statistics}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variable & N & Mean & SD & Min & P25 & P50 & P75 & Max \\",
        r"\midrule",
    ]

    for _, row in overall.iterrows():
        var = row["Variable"].replace("_", r"\_")
        n = f"{int(row['N']):,}"
        vals = [f"{row[c]:.2f}" for c in ["Mean", "SD", "Min", "P25", "P50", "P75", "Max"]]
        lines.append(f"{var} & {n} & {' & '.join(vals)} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Between/Within Decomposition}} \\")
    lines.append(r"Variable & \multicolumn{2}{c}{Overall SD} & \multicolumn{2}{c}{Between SD} & \multicolumn{2}{c}{Within SD} & \multicolumn{2}{c}{Within \%} \\")
    lines.append(r"\midrule")

    for _, row in bw.iterrows():
        var = row["Variable"].replace("_", r"\_")
        lines.append(f"{var} & \\multicolumn{{2}}{{c}}{{{row['Overall_SD']:.2f}}} "
                      f"& \\multicolumn{{2}}{{c}}{{{row['Between_SD']:.2f}}} "
                      f"& \\multicolumn{{2}}{{c}}{{{row['Within_SD']:.2f}}} "
                      f"& \\multicolumn{{2}}{{c}}{{{row['Within_Pct']:.1f}\\%}} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\small",
              r"\item Notes: Daily data for 9 ARK ETFs. Fund\_Flow in millions USD.",
              r"Return is daily log return. Within \% is the share of total variance",
              r"explained by within-ETF variation.",
              r"\end{tablenotes}", r"\end{table}"]

    tex = "\n".join(lines)
    (TABLES / "table_1_summary.tex").write_text(tex)
    print("Table 1 saved")


# ============================================================
# Table 3: Main Panel Specification
# ============================================================

def table_3():
    df = pd.read_csv(RESULTS / "table_3_main_panel.csv")
    specs = df["spec"].unique()

    # Extract variables (not R² or N)
    var_rows = df[~df["Variable"].isin(["R²", "N"])]
    stat_rows = df[df["Variable"].isin(["R²", "N"])]
    variables = var_rows["Variable"].unique()

    # Filter to CumRet variables for main display
    main_vars = [v for v in variables if v.startswith("CumRet")]
    control_vars = [v for v in variables if not v.startswith("CumRet")]

    n_specs = len(specs)
    col_fmt = "l" + "c" * n_specs

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Main Panel Specification: Daily Fund Flows and Past Returns}",
        r"\label{tab:main_panel}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    # Header
    header = " & " + " & ".join(specs) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Coefficient rows
    for var in main_vars:
        var_data = var_rows[var_rows["Variable"] == var]
        cells = []
        se_cells = []
        for spec in specs:
            row = var_data[var_data["spec"] == spec]
            if row.empty:
                cells.append("")
                se_cells.append("")
            else:
                r = row.iloc[0]
                coef = r["Coefficient"]
                p = r["p_value"]
                se = r["Std_Error"]
                cells.append(f"${coef:.2f}{_stars(p)}$")
                se_cells.append(f"$({se:.2f})$")

        var_label = var.replace("_", r"\_")
        lines.append(f"{var_label} & " + " & ".join(cells) + r" \\")
        lines.append(" & " + " & ".join(se_cells) + r" \\[3pt]")

    # Controls indicator
    lines.append(r"\midrule")
    controls = ["VIX", "Calendar", "Peer Flow", "Events"]
    ctrl_specs = {
        "VIX": [False, True, True, True, True],
        "Calendar": [False, False, True, True, True],
        "Peer Flow": [False, False, False, True, True],
        "Events": [False, False, False, False, True],
    }
    for ctrl in controls:
        flags = ctrl_specs[ctrl][:n_specs]
        cells = ["Yes" if f else "" for f in flags]
        lines.append(f"{ctrl} & " + " & ".join(cells) + r" \\")

    # Fit statistics
    lines.append(r"\midrule")
    lines.append("Entity FE & " + " & ".join(["Yes"] * n_specs) + r" \\")

    for stat_var, label in [("R²", "$R^2$ (within)"), ("N", "Observations")]:
        cells = []
        for spec in specs:
            row = stat_rows[(stat_rows["spec"] == spec) & (stat_rows["Variable"] == stat_var)]
            if row.empty:
                cells.append("")
            else:
                val = row.iloc[0]["Coefficient"]
                if stat_var == "R²":
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(f"{int(val):,}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\small",
              r"\item Notes: Entity-demeaned OLS with standard errors clustered by ETF",
              r"in parentheses. CumRet\_$k$\_$l$ is the cumulative return from day $t-k$",
              r"to $t-l$. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
              r"\end{tablenotes}", r"\end{table}"]

    tex = "\n".join(lines)
    (TABLES / "table_3_main_panel.tex").write_text(tex)
    print("Table 3 saved")


# ============================================================
# Table 4: Sub-sample
# ============================================================

def table_4():
    df = pd.read_csv(RESULTS / "table_4_subsample.csv")

    cumret_cols = [c for c in df.columns if c.endswith("_coef") and "CumRet" in c]
    pval_cols = [c.replace("_coef", "_pval") for c in cumret_cols]

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Sub-sample Analysis: Bull vs.\ Bear Market}",
        r"\label{tab:subsample}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & Full Sample & Pre-COVID & Bull (2020--2021) & Bear (2022--2024) \\",
        r"\midrule",
    ]

    for coef_col, pval_col in zip(cumret_cols, pval_cols):
        var_name = coef_col.replace("_coef", "").replace("_", r"\_")
        cells = []
        for _, row in df.iterrows():
            c = row[coef_col]
            p = row[pval_col]
            cells.append(f"${c:.2f}{_stars(p)}$")
        lines.append(f"{var_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # R² and N
    r2_cells = [f"{r:.4f}" for r in df["r2"]]
    n_cells = [f"{int(n):,}" for n in df["n_obs"]]
    lines.append(f"$R^2$ (within) & " + " & ".join(r2_cells) + r" \\")
    lines.append(f"Observations & " + " & ".join(n_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\small",
              r"\item Notes: Same specification as Table \ref{tab:main_panel} Column (5).",
              r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
              r"\end{tablenotes}", r"\end{table}"]

    tex = "\n".join(lines)
    (TABLES / "table_4_subsample.tex").write_text(tex)
    print("Table 4 saved")


# ============================================================
# Table 5: Robustness
# ============================================================

def table_5():
    # 5a: Placebo
    real = pd.read_csv(RESULTS / "table_5a_placebo_real.csv")
    fake = pd.read_csv(RESULTS / "table_5a_placebo_fake.csv")

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness: Placebo Test (Future Returns)}",
        r"\label{tab:placebo}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & Real (Lag Returns) & Placebo (Lead Returns) \\",
        r"\midrule",
    ]

    for _, rr in real.iterrows():
        var = rr["Variable"].replace("_", r"\_")
        fr = fake[fake["Variable"] == rr["Variable"].replace("lag", "lead")]
        real_cell = f"${rr['Coefficient']:.2f}{_stars(rr['p_value'])}$"
        if not fr.empty:
            fake_cell = f"${fr.iloc[0]['Coefficient']:.2f}{_stars(fr.iloc[0]['p_value'])}$"
        else:
            fake_cell = ""
        lines.append(f"{var} & {real_cell} & {fake_cell} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\small",
              r"\item Notes: Placebo specification uses future (lead) returns instead of past (lag)",
              r"returns. Insignificant placebo coefficients confirm that the relationship is not spurious.",
              r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
              r"\end{tablenotes}", r"\end{table}"]

    tex = "\n".join(lines)
    (TABLES / "table_5a_placebo.tex").write_text(tex)

    # 5b: Leave-one-out
    loo = pd.read_csv(RESULTS / "table_5b_leave_one_out.csv")
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness: Leave-One-ETF-Out}",
        r"\label{tab:leaveoneout}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"ETF Excluded & CumRet\_1\_5 & CumRet\_6\_20 & CumRet\_21\_60 & $R^2$ & N \\",
        r"\midrule",
    ]

    cumret_vars = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]
    for _, row in loo.iterrows():
        etf = row["ETF_excluded"]
        cells = [etf]
        for v in cumret_vars:
            c = row.get(f"{v}_coef", np.nan)
            p = row.get(f"{v}_pval", np.nan)
            cells.append(f"${c:.2f}{_stars(p)}$")
        cells.append(f"{row['r2']:.4f}")
        cells.append(f"{int(row['n_obs']):,}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\small",
              r"\item Notes: Each row drops one ETF and re-estimates the full specification.",
              r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
              r"\end{tablenotes}", r"\end{table}"]

    tex = "\n".join(lines)
    (TABLES / "table_5b_leave_one_out.tex").write_text(tex)

    print("Table 5 saved")


# ============================================================
# Table 2: Sirri-Tufano Piecewise Linear
# ============================================================

def table_2():
    df = pd.read_csv(RESULTS / "table_2_sirri_tufano.csv")

    specs = ["(1) Base", "(2) + VIX", "(3) + Calendar", "(4) + Peer"]
    variables = df["Variable"].tolist()

    # Separate coefficient rows from stats rows
    stat_vars = {"R² within", "N", "ETFs"}
    coef_mask = ~df["Variable"].isin(stat_vars)

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Sirri--Tufano Piecewise Linear Specification (Monthly)}",
        r"\label{tab:sirri_tufano}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & (1) Base & (2) + VIX & (3) + Calendar & (4) + Peer \\",
        r"\midrule",
    ]

    for _, row in df[coef_mask].iterrows():
        var = row["Variable"].replace("_", r"\_")
        cells = []
        for spec in specs:
            coef_val = row.get(spec, "")
            se_val = row.get(f"{spec}_se", "")
            if pd.isna(coef_val) or coef_val == "":
                cells.append("")
            else:
                cells.append(str(coef_val))
            # SE on next line handled below
        lines.append(f"{var} & " + " & ".join(cells) + r" \\")
        # SE row
        se_cells = []
        for spec in specs:
            se_val = row.get(f"{spec}_se", "")
            if pd.isna(se_val) or se_val == "":
                se_cells.append("")
            else:
                se_cells.append(str(se_val))
        lines.append(" & " + " & ".join(se_cells) + r" \\[3pt]")

    lines.append(r"\midrule")

    for _, row in df[~coef_mask].iterrows():
        var = row["Variable"].replace("²", "$^2$").replace("_", r"\_")
        cells = []
        for spec in specs:
            val = row.get(spec, "")
            if pd.isna(val) or val == "":
                cells.append("")
            else:
                try:
                    fval = float(val)
                    if var.startswith("N") or var.startswith("ETF"):
                        cells.append(f"{int(fval):,}")
                    else:
                        cells.append(f"{fval:.4f}")
                except ValueError:
                    cells.append(str(val))
        lines.append(f"{var} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: Monthly frequency, 44 ETFs (9 ARK + 35 peers).",
        r"LOWPERF, MIDPERF, HIGHPERF are fractional performance ranks",
        r"(bottom 20\%, middle 60\%, top 20\%) following Sirri \& Tufano (1998).",
        r"Entity-clustered standard errors in parentheses.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_2_sirri_tufano.tex").write_text(tex)
    print("Table 2 saved")


# ============================================================
# Table 5c: Fama-MacBeth
# ============================================================

def table_5c():
    df = pd.read_csv(RESULTS / "table_5c_fama_macbeth.csv")

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness: Fama--MacBeth Cross-Sectional Regression}",
        r"\label{tab:fama_macbeth}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variable & Coefficient & Std.\ Error & $t$-stat & $p$-value \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        if pd.isna(row["Coefficient"]):
            continue
        var = row["Variable"].replace("_", r"\_")
        coef = row["Coefficient"]
        se = row["Std_Error"]
        t = row["t_stat"]
        p = row["p_value"]
        lines.append(
            f"{var} & ${coef:.2f}{_stars(p)}$ & $({se:.2f})$ & "
            f"${t:.2f}$ & ${p:.3f}$ \\\\"
        )

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: Fama--MacBeth (1973) procedure: cross-sectional regressions",
        r"estimated each period, coefficients averaged over time. Standard errors",
        r"account for time-series variation in cross-sectional estimates.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_5c_fama_macbeth.tex").write_text(tex)
    print("Table 5c saved")


# ============================================================
# Table 5d: Flow % AUM
# ============================================================

def table_5d():
    df = pd.read_csv(RESULTS / "table_5d_flow_pct.csv")

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness: Alternative Dependent Variable (Flow \% of AUM)}",
        r"\label{tab:flow_pct}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variable & Coefficient & Std.\ Error & $t$-stat & $p$-value \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        if pd.isna(row["Coefficient"]):
            continue
        var = row["Variable"].replace("_", r"\_")
        coef = row["Coefficient"]
        se = row["Std_Error"]
        t = row["t_stat"]
        p = row["p_value"]
        lines.append(
            f"{var} & ${coef:.4f}{_stars(p)}$ & $({se:.4f})$ & "
            f"${t:.2f}$ & ${p:.4f}$ \\\\"
        )

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: Same specification as Table \ref{tab:main_panel} Column (5)",
        r"with dependent variable rescaled as fund flow divided by lagged AUM.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_5d_flow_pct.tex").write_text(tex)
    print("Table 5d saved")


# ============================================================
# Table: Economic Significance
# ============================================================

def table_econ():
    df = pd.read_csv(RESULTS / "economic_significance.csv")
    row = df.iloc[0]

    cumret_vars = ["CumRet_1_5", "CumRet_6_20", "CumRet_21_60"]

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Economic Significance of Return--Flow Relationship}",
        r"\label{tab:econ_sig}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Return Window & 1-SD Shock (\$M) & Standardized $\beta$ & Mean AUM (\$M) \\",
        r"\midrule",
    ]

    for var in cumret_vars:
        label = var.replace("_", r"\_")
        effect = row[f"{var}_1sd_effect"]
        std_beta = row[f"{var}_std_beta"]
        aum = row["mean_aum_millions"]
        lines.append(f"{label} & ${effect:.2f}$ & ${std_beta:.4f}$ & ${aum:.0f}$ \\\\")

    lines.append(r"\midrule")
    lines.append(f"SD(Flow) & \\multicolumn{{3}}{{c}}{{\\${row['sd_flow']:.2f}M}} \\\\")

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: 1-SD Shock shows the fund flow response (in \$M) to a",
        r"one-standard-deviation increase in cumulative returns over each window.",
        r"Standardized $\beta$ is the coefficient scaled by $\sigma_x / \sigma_y$.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_econ_significance.tex").write_text(tex)
    print("Economic significance table saved")


# ============================================================
# Table 5e: Driscoll-Kraay SE
# ============================================================

def table_5e():
    f = RESULTS / "table_5e_driscoll_kraay.csv"
    if not f.exists():
        print("Skipping Table 5e: CSV not found")
        return

    df = pd.read_csv(f)

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Robustness: Driscoll--Kraay Standard Errors}",
        r"\label{tab:driscoll_kraay}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variable & Coefficient & DK Std.\ Error & $t$-stat & $p$-value \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        var = row["Variable"].replace("_", r"\_")
        coef = row["Coefficient"]
        se = row["Std_Error"]
        t = row["t_stat"]
        p = row["p_value"]
        lines.append(
            f"{var} & ${coef:.2f}{_stars(p)}$ & $({se:.2f})$ & "
            f"${t:.2f}$ & ${p:.3f}$ \\\\"
        )

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: Same specification as Table \ref{tab:main_panel} Column (1)",
        r"with Driscoll--Kraay (1998) standard errors, which are robust to both",
        r"cross-sectional and temporal dependence. Bartlett kernel with automatic",
        r"bandwidth selection.",
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_5e_driscoll_kraay.tex").write_text(tex)
    print("Table 5e saved")


# ============================================================
# Table 5f: Heteroscedasticity Diagnostics
# ============================================================

def table_5f():
    f = RESULTS / "table_5f_diagnostics.csv"
    if not f.exists():
        print("Skipping Table 5f: CSV not found")
        return

    df = pd.read_csv(f)

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Diagnostic Tests: Heteroscedasticity}",
        r"\label{tab:het_tests}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Test & Statistic & $p$-value \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        name = row["Test"]
        stat = row["statistic"]
        p = row["p_value"]
        sig = " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.10 else ""
        lines.append(f"{name} & ${stat:.2f}$ & ${p:.4f}{sig}$ \\\\")

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Notes: Both tests applied to the entity-demeaned residuals from the",
        r"main specification. Rejection ($p < 0.05$) indicates heteroscedasticity,",
        r"justifying the use of clustered or Driscoll--Kraay standard errors.",
        r"\end{tablenotes}", r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "table_5f_diagnostics.tex").write_text(tex)
    print("Table 5f saved")


if __name__ == "__main__":
    table_1()
    table_2()
    table_3()
    table_4()
    table_5()
    table_5c()
    table_5d()
    table_5e()
    table_5f()
    table_econ()
    print("All LaTeX tables generated.")
