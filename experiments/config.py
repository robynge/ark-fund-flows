"""Experiment grid definition.

Enumerates all factor combinations, frequencies, flow units, and model
specifications. The runner iterates this grid and writes results to CSV.
"""
from itertools import combinations

# ============================================================
# Factor combinations: all 2^5 - 1 = 31 non-empty subsets of {A,B,C,D,E}
# ============================================================

ALL_FACTORS = ["A", "B", "C", "D", "E"]


def all_factor_combos() -> list[tuple[str, list[str]]]:
    """Return (experiment_id, factor_list) for all 31 non-empty combinations."""
    combos = []
    for r in range(1, len(ALL_FACTORS) + 1):
        for combo in combinations(ALL_FACTORS, r):
            factor_list = list(combo)
            exp_id = "N-" + "".join(factor_list)
            combos.append((exp_id, factor_list))
    return combos


FACTOR_COMBOS = all_factor_combos()  # 31 entries

# ============================================================
# Frequencies
# ============================================================

FREQUENCIES = ["D", "W", "ME", "QE"]

# ============================================================
# Flow units
# ============================================================

FLOW_UNITS = {
    "raw": {
        "daily_col": "Fund_Flow",
        "agg_col": "Flow_Sum",
        "zscore_suffix": "_Z",
    },
    "pct_aum": {
        "daily_col": "Flow_Pct",
        "agg_col": "Flow_Pct",
        "zscore_suffix": "_Z",
    },
}

# ============================================================
# Benchmarks (for excess return computation)
# ============================================================

BENCHMARKS = ["SPY", "QQQ", "peer_avg"]

# ============================================================
# Model specifications
# ============================================================

MODELS = {
    "univariate_r2_by_lag": {
        "description": "R² from OLS (flow ~ return_lag_k) for each lag k",
        "function": "r_squared_by_lag_all_etfs",
        "scope": "per_etf",
    },
    "multilag_ols": {
        "description": "Multi-lag OLS: flow ~ Σ βₖ·return(t-k)",
        "function": "lag_regression_all_etfs",
        "scope": "per_etf",
        "kwargs": {"add_month_dummies": False},
    },
    "multilag_ols_month_fe": {
        "description": "Multi-lag OLS + month dummies",
        "function": "lag_regression_all_etfs",
        "scope": "per_etf",
        "kwargs": {"add_month_dummies": True},
    },
    "cross_correlation": {
        "description": "Pearson cross-correlation at lags -20..+20",
        "function": "cross_correlation_all_etfs",
        "scope": "per_etf",
    },
    "panel_pooled": {
        "description": "Pooled OLS (no fixed effects)",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": False, "time_effects": False},
    },
    "panel_entity_fe": {
        "description": "Entity (ETF) fixed effects",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": False},
    },
    "panel_entity_time_fe": {
        "description": "Entity + time fixed effects",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": True},
    },
    "panel_entity_fe_excess": {
        "description": "Entity FE + excess return",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": False,
                   "use_excess": True},
    },
    "panel_entity_fe_controls": {
        "description": "Entity FE + volatility control",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": False,
                   "add_controls": True},
    },
    "asymmetry": {
        "description": "Piecewise regression: β⁺ vs β⁻",
        "function": "asymmetry_all_etfs",
        "scope": "per_etf",
    },
    "relative_performance": {
        "description": "Absolute vs excess vs combined R² comparison",
        "function": "relative_performance_all_etfs",
        "scope": "per_etf",
    },
    "granger": {
        "description": "Granger causality test (both directions)",
        "function": "granger_causality_test",
        "scope": "per_etf",
    },
    "seasonality": {
        "description": "Average flow by calendar month",
        "function": "seasonality_analysis",
        "scope": "pooled",
    },
    "drawdown": {
        "description": "Drawdown event study + flow regression",
        "function": "drawdown_flow_analysis",
        "scope": "per_etf",
    },
    # --- Cumulative return models ---
    "panel_entity_fe_cum": {
        "description": "Entity FE with cumulative return windows (5, 20, 60)",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": False,
                   "cum_windows": [5, 20, 60]},
    },
    "panel_entity_fe_cum_long": {
        "description": "Entity FE with longer cumulative windows (20, 60, 120)",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": False,
                   "cum_windows": [20, 60, 120]},
    },
    "panel_entity_time_fe_cum": {
        "description": "Entity+Time FE with cumulative return windows",
        "function": "panel_regression",
        "scope": "panel",
        "kwargs": {"entity_effects": True, "time_effects": True,
                   "cum_windows": [5, 20, 60]},
    },
}

# ============================================================
# Default run configuration (can be overridden from CLI)
# ============================================================

DEFAULT_CONFIG = {
    "frequencies": ["ME"],
    "flow_units": ["raw"],
    "benchmarks": ["SPY"],
    "zscore_type": "full",
    "models": list(MODELS.keys()),
    "include_baseline": True,
    "factor_combos": FACTOR_COMBOS,
}

FULL_CONFIG = {
    "frequencies": FREQUENCIES,
    "flow_units": list(FLOW_UNITS.keys()),
    "benchmarks": BENCHMARKS,
    "zscore_type": "full",
    "models": list(MODELS.keys()),
    "include_baseline": True,
    "factor_combos": FACTOR_COMBOS,
}
