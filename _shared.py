"""Shared helpers for all Streamlit pages."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
from data_loader import get_prepared_data_with_peers, ETF_NAMES

# Also make new_runner helpers importable
sys.path.insert(0, str(_ROOT / "experiments"))

FREQ_OPTIONS = {"D": "Daily", "W": "Weekly", "ME": "Monthly"}
FREQ_LABELS = {"D": "days", "W": "weeks", "ME": "months"}

ETF_DISPLAY = {
    "ARKK": "ARK Innovation", "ARKF": "ARK Fintech", "ARKG": "ARK Genomic",
    "ARKX": "ARK Space & Defense", "ARKQ": "ARK Autonomous Tech",
    "ARKW": "ARK Next Gen Internet",
}


def sidebar_freq(key: str = "freq") -> str:
    return st.sidebar.selectbox(
        "Frequency", list(FREQ_OPTIONS),
        format_func=FREQ_OPTIONS.get, index=2, key=key)


def sidebar_etf(key: str = "etf") -> str:
    return st.sidebar.selectbox(
        "Select ETF", ETF_NAMES, index=0,
        format_func=lambda x: f"{x} — {ETF_DISPLAY.get(x, x)}", key=key)


@st.cache_data(show_spinner="Loading data...")
def load_data(freq: str):
    return get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark="SPY")


@st.cache_data(show_spinner="Preparing controls...")
def load_data_with_controls(freq: str):
    from new_runner import prepare_controls
    df = get_prepared_data_with_peers(freq=freq, zscore_type="full", benchmark="SPY")
    df = prepare_controls(df, freq=freq)
    return df


def get_cols(freq: str) -> tuple[str, str]:
    if freq == "D":
        return "Fund_Flow", "Return"
    return "Flow_Sum", "Return_Cum"


def get_cumret_windows(freq: str) -> list[tuple[int, int]]:
    if freq == "D":
        return [(1, 5), (6, 20), (21, 60)]
    elif freq == "W":
        return [(1, 4), (5, 12), (13, 26)]
    return [(1, 3), (4, 6), (7, 12)]


def get_lp_horizon(freq: str) -> int:
    if freq == "D":
        return 40
    elif freq == "W":
        return 20
    return 12


def get_rolling_window(freq: str) -> int:
    """Rolling window size in periods (for rolling regression)."""
    if freq == "D":
        return 504  # ~2 years
    elif freq == "W":
        return 104  # ~2 years
    return 24  # 2 years


def build_cumret(df: pd.DataFrame, return_col: str,
                 windows: list[tuple[int, int]]) -> pd.DataFrame:
    """Build non-overlapping cumulative return windows."""
    pdf = df.copy()
    for start, end in windows:
        col = f"CumRet_{start}_{end}"
        pdf[col] = pdf.groupby("ETF")[return_col].transform(
            lambda x, s=start, e=end: sum(x.shift(k) for k in range(s, e + 1)))
    return pdf


def cumret_cols(windows: list[tuple[int, int]]) -> list[str]:
    return [f"CumRet_{s}_{e}" for s, e in windows]


def stars(p) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
