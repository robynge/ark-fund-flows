"""Generate publication-quality figures for the paper."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Scientific figure style
sys.path.insert(0, str(Path.home() / ".claude/skills/scientific-figure-pro/scripts"))
from scientific_figure_pro import (
    apply_publication_style,
    FigureStyle,
    finalize_figure,
    PALETTE,
)

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "experiments" / "results_v2"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2))


def figure_1_impulse_response():
    """Figure 1: Local Projection impulse response (h=0..40 days)."""
    lp = pd.read_csv(RESULTS / "figure_1_lp.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    h = lp["horizon"]
    beta = lp["beta"]
    ci_lo = lp["ci_lower"]
    ci_hi = lp["ci_upper"]

    # Confidence band
    ax.fill_between(h, ci_lo, ci_hi, alpha=0.2, color=PALETTE["blue_main"],
                    label="95% CI")

    # Point estimates
    ax.plot(h, beta, color=PALETTE["blue_main"], linewidth=2.0,
            marker="o", markersize=3, label=r"$\hat{\beta}_h$")

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Mark significant horizons
    sig = lp[lp["p_value"] < 0.05]
    if not sig.empty:
        ax.scatter(sig["horizon"], sig["beta"], color=PALETTE["red_strong"],
                   s=25, zorder=5, label="p < 0.05")

    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Response of Fund Flow ($M)")
    ax.set_title("Impulse Response: Return Shock → Fund Flow", fontsize=16,
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")

    for path in [FIGURES / "figure_1_impulse_response.png",
                 FIGURES / "figure_1_impulse_response.pdf"]:
        finalize_figure(fig, str(path), dpi=300, pad=2)

    plt.close(fig)
    print(f"Figure 1 saved to {FIGURES}")


def figure_2_asymmetric():
    """Figure 2: Asymmetric impulse response (positive vs negative shocks)."""
    lp = pd.read_csv(RESULTS / "figure_2_asymmetric_lp.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    h = lp["horizon"]

    # Positive shock path
    ax.fill_between(h, lp["ci_lower_pos"], lp["ci_upper_pos"],
                    alpha=0.15, color=PALETTE["blue_main"])
    ax.plot(h, lp["beta_pos"], color=PALETTE["blue_main"], linewidth=2.0,
            marker="o", markersize=3, label="Positive return shock (chasing)")

    # Negative shock path
    ax.fill_between(h, lp["ci_lower_neg"], lp["ci_upper_neg"],
                    alpha=0.15, color=PALETTE["red_strong"])
    ax.plot(h, lp["beta_neg"], color=PALETTE["red_strong"], linewidth=2.0,
            marker="s", markersize=3, label="Negative return shock (fleeing)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Response of Fund Flow ($M)")
    ax.set_title("Asymmetric Response: Chasing vs. Fleeing", fontsize=16,
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")

    for path in [FIGURES / "figure_2_asymmetric.png",
                 FIGURES / "figure_2_asymmetric.pdf"]:
        finalize_figure(fig, str(path), dpi=300, pad=2)

    plt.close(fig)
    print(f"Figure 2 saved to {FIGURES}")


def figure_3_subsample_lp():
    """Figure 3 (bonus): Bull vs Bear LP comparison."""
    bull_f = RESULTS / "table_4_lp_bull.csv"
    bear_f = RESULTS / "table_4_lp_bear.csv"
    if not bull_f.exists() or not bear_f.exists():
        print("Skipping Figure 3: sub-sample LP data not found")
        return

    bull = pd.read_csv(bull_f)
    bear = pd.read_csv(bear_f)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bull
    ax.fill_between(bull["horizon"], bull["ci_lower"], bull["ci_upper"],
                    alpha=0.15, color=PALETTE["blue_main"])
    ax.plot(bull["horizon"], bull["beta"], color=PALETTE["blue_main"],
            linewidth=2.0, marker="o", markersize=3,
            label="Bull market (2020-2021)")

    # Bear
    ax.fill_between(bear["horizon"], bear["ci_lower"], bear["ci_upper"],
                    alpha=0.15, color=PALETTE["red_strong"])
    ax.plot(bear["horizon"], bear["beta"], color=PALETTE["red_strong"],
            linewidth=2.0, marker="s", markersize=3,
            label="Bear market (2022-2024)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Response of Fund Flow ($M)")
    ax.set_title("Performance Chasing: Bull vs. Bear Market", fontsize=16,
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")

    for path in [FIGURES / "figure_3_bull_vs_bear.png",
                 FIGURES / "figure_3_bull_vs_bear.pdf"]:
        finalize_figure(fig, str(path), dpi=300, pad=2)

    plt.close(fig)
    print(f"Figure 3 saved to {FIGURES}")


if __name__ == "__main__":
    figure_1_impulse_response()
    figure_2_asymmetric()
    figure_3_subsample_lp()
    print("All figures generated.")
