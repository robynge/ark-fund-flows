"""Noise factor transforms (A–E).

Each factor is a standalone function that takes a DataFrame and returns a
transformed DataFrame. Factors are composable: applying A then C means
calling both functions sequentially.

Factor registry maps factor IDs to their transform functions so the
experiment runner can apply arbitrary combinations programmatically.
"""
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from .macro_events import exclude_events, add_event_dummies, get_event_ids
except ImportError:
    from macro_events import exclude_events, add_event_dummies, get_event_ids

DATA_DIR = Path(__file__).parent.parent / "data" / "input"


# ============================================================
# Factor A: Macro event exclusion (all events as one factor)
# ============================================================

def apply_factor_A(df: pd.DataFrame, method: str = "exclude",
                   date_col: str = "Date") -> pd.DataFrame:
    """Factor A: Remove or dummy-flag ALL macro event windows.

    Treats all registered macro events (COVID, Ukraine war, Fed hikes,
    banking crisis, Bitcoin ETF approval, etc.) as a single noise factor.
    Individual event impact analysis is a separate investigation,
    not a noise factor toggle.

    Methods:
        "exclude": Drop rows inside any event window.
        "dummy": Add per-event binary columns (1 = inside window).
    """
    all_event_ids = get_event_ids()
    if method == "exclude":
        return exclude_events(df, all_event_ids, date_col)
    else:
        return add_event_dummies(df, all_event_ids, date_col)


# ============================================================
# Factor B: Market-wide flow trends
# ============================================================

def apply_factor_B(df: pd.DataFrame, flow_col: str = "Fund_Flow",
                   date_col: str = "Date") -> pd.DataFrame:
    """Factor B: Subtract cross-sectional mean flow per date.

    Removes the market-wide component of fund flows so that only the
    ETF-specific deviation remains.
    """
    df = df.copy()
    market_flow = df.groupby(date_col)[flow_col].transform("mean")
    df[flow_col] = df[flow_col] - market_flow
    return df


# ============================================================
# Factor C: Volatility regime (VIX-based)
# ============================================================

def _load_vix() -> pd.DataFrame:
    """Load VIX data from local CSV."""
    vix_path = DATA_DIR / "VIX.csv"
    if not vix_path.exists():
        raise FileNotFoundError(
            f"VIX data not found at {vix_path}. "
            "Download it first: yfinance ^VIX → data/VIX.csv"
        )
    vix = pd.read_csv(vix_path, parse_dates=["Date"])
    return vix[["Date", "VIX_Close"]].copy()


def apply_factor_C(df: pd.DataFrame, method: str = "control",
                   date_col: str = "Date",
                   high_vix_threshold: float = 25.0) -> pd.DataFrame:
    """Factor C: Control for volatility regime using VIX.

    Methods:
        "control": Add VIX_Close as a control variable column.
        "exclude_high": Remove dates where VIX > threshold.
        "regime_dummy": Add binary column VIX_High (1 if VIX > threshold).
    """
    df = df.copy()
    vix = _load_vix()

    df = df.merge(vix, on=date_col, how="left")
    # Forward-fill VIX for non-trading days that might appear in aggregated data
    df["VIX_Close"] = df["VIX_Close"].ffill()

    if method == "control":
        # VIX_Close column is already added
        pass
    elif method == "exclude_high":
        df = df[df["VIX_Close"] <= high_vix_threshold].copy()
    elif method == "regime_dummy":
        df["VIX_High"] = (df["VIX_Close"] > high_vix_threshold).astype(int)
    else:
        raise ValueError(f"Unknown method for factor C: {method}")

    return df


# ============================================================
# Factor D: Month/quarter-end rebalancing dummies
# ============================================================

def apply_factor_D(df: pd.DataFrame,
                   date_col: str = "Date") -> pd.DataFrame:
    """Factor D: Add calendar dummy variables for rebalancing effects.

    Adds columns:
        month_end: 1 if last 3 trading days of month
        quarter_end: 1 if last 5 trading days of quarter
        january: 1 if January (January effect)
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    # Month-end: last 3 calendar days of each month
    month_end_date = dates + pd.offsets.MonthEnd(0)
    days_to_month_end = (month_end_date - dates).dt.days
    df["month_end"] = (days_to_month_end <= 3).astype(int)

    # Quarter-end: last 5 calendar days of each quarter
    quarter_end_date = dates + pd.offsets.QuarterEnd(0)
    days_to_quarter_end = (quarter_end_date - dates).dt.days
    df["quarter_end"] = (days_to_quarter_end <= 5).astype(int)

    # January dummy
    df["january"] = (dates.dt.month == 1).astype(int)

    return df


# ============================================================
# Factor E: Peer aggregate flow control
# ============================================================

def apply_factor_E(df: pd.DataFrame, flow_col: str = "Fund_Flow",
                   date_col: str = "Date") -> pd.DataFrame:
    """Factor E: Add per-peer-group aggregate flow as control variable.

    For each ARK ETF on each date, computes the mean flow of its OWN
    peer group (from PEER_MAPPING) and adds it as Peer_Agg_Flow.
    Falls back to cross-sectional leave-one-out mean if no peer mapping
    is available.
    """
    from data_loader import PEER_MAPPING, ETF_NAMES

    df = df.copy()
    df["Peer_Agg_Flow"] = np.nan

    if PEER_MAPPING:
        # Build a lookup: for each date, the flow of every ETF
        flow_by_date_etf = df.set_index([date_col, "ETF"])[flow_col]

        for ark_etf, peers in PEER_MAPPING.items():
            ark_mask = df["ETF"] == ark_etf
            if not ark_mask.any():
                continue
            # Get flows of this ARK ETF's peers on each date
            peer_flows = df[df["ETF"].isin(peers)].groupby(date_col)[flow_col].mean()
            peer_flows.name = "_peer_mean"
            # Map to ARK ETF rows
            df.loc[ark_mask, "Peer_Agg_Flow"] = df.loc[ark_mask, date_col].map(peer_flows).values
    else:
        # Fallback: cross-sectional leave-one-out (old behavior)
        date_totals = df.groupby(date_col)[flow_col].agg(["sum", "count"])
        date_totals.columns = ["_total_flow", "_n_etfs"]
        df = df.merge(date_totals, on=date_col, how="left")
        df["Peer_Agg_Flow"] = (df["_total_flow"] - df[flow_col]) / (df["_n_etfs"] - 1)
        df = df.drop(columns=["_total_flow", "_n_etfs"])

    return df


# ============================================================
# Factor registry
# ============================================================

FACTOR_REGISTRY = {
    "A": apply_factor_A,
    "B": apply_factor_B,
    "C": apply_factor_C,
    "D": apply_factor_D,
    "E": apply_factor_E,
}


def apply_factors(df: pd.DataFrame, factor_ids: list[str],
                  flow_col: str = "Fund_Flow",
                  date_col: str = "Date") -> pd.DataFrame:
    """Apply a list of noise factors sequentially.

    Parameters:
        df: Input DataFrame
        factor_ids: e.g., ["A", "C", "E"] to apply factors A, C, and E
        flow_col: Name of the flow column (needed for factors B, E)
        date_col: Name of the date column

    Returns:
        Transformed DataFrame with all specified factors applied.
    """
    result = df.copy()
    for fid in sorted(factor_ids):
        if fid not in FACTOR_REGISTRY:
            raise ValueError(
                f"Unknown factor: {fid}. Available: {list(FACTOR_REGISTRY.keys())}"
            )
        func = FACTOR_REGISTRY[fid]
        if fid == "A":
            result = func(result, method="exclude", date_col=date_col)
        elif fid in ("B", "E"):
            result = func(result, flow_col=flow_col, date_col=date_col)
        elif fid == "C":
            result = func(result, method="control", date_col=date_col)
        elif fid == "D":
            result = func(result, date_col=date_col)
    return result
