"""Data loading, feature engineering, and normalization for ARK ETF fund flows."""
import pandas as pd
import numpy as np
from pathlib import Path

EXCEL_PATH = Path(__file__).parent.parent / "ARK ETF Fund Flows.xlsx"

ETF_NAMES = ["ARKK", "ARKF", "ARKG", "ARKX", "ARKB", "ARKQ", "ARKW", "PRNT", "IZRL"]



def load_single_etf(sheet_name: str) -> pd.DataFrame:
    """Load a single ETF sheet from the Excel file."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
    df.columns = ["Date", "Fund_Flow", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])
    df["ETF"] = sheet_name
    return df


def load_all_etfs() -> pd.DataFrame:
    """Load all ETF sheets and concatenate into one DataFrame."""
    frames = []
    for name in ETF_NAMES:
        try:
            frames.append(load_single_etf(name))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily return column, computed per ETF."""
    df = df.copy()
    df["Return"] = df.groupby("ETF")["Close"].pct_change()
    return df


def aggregate_to_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate daily data to weekly (W), monthly (ME), or quarterly (QE).
    Returns DataFrame with columns: Date, ETF, Flow_Sum, Return_Cum, Close_Last.
    """
    df = df.copy()
    df = df.set_index("Date")
    groups = df.groupby("ETF")

    result_frames = []
    for etf, group in groups:
        resampled = group.resample(freq).agg(
            Flow_Sum=("Fund_Flow", "sum"),
            Return_Cum=("Return", lambda x: (1 + x).prod() - 1),
            Close_Last=("Close", "last"),
            Volume_Sum=("Volume", "sum"),
        )
        resampled = resampled.dropna(subset=["Close_Last"])
        resampled["ETF"] = etf
        result_frames.append(resampled)

    result = pd.concat(result_frames)
    result = result.reset_index()
    return result


def zscore_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize a series (subtract mean, divide by std)."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std


def add_zscore_columns(df: pd.DataFrame, flow_col: str = "Fund_Flow",
                       return_col: str = "Return") -> pd.DataFrame:
    """Add z-score normalized flow and return columns, per ETF."""
    df = df.copy()
    df[f"{flow_col}_Z"] = df.groupby("ETF")[flow_col].transform(zscore_normalize)
    df[f"{return_col}_Z"] = df.groupby("ETF")[return_col].transform(zscore_normalize)
    return df


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score normalization."""
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window, min_periods=window // 2).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def add_rolling_zscore_columns(df: pd.DataFrame, window: int = 252,
                               flow_col: str = "Fund_Flow",
                               return_col: str = "Return") -> pd.DataFrame:
    """Add rolling z-score normalized columns, per ETF."""
    df = df.copy()
    df[f"{flow_col}_RZ"] = df.groupby("ETF")[flow_col].transform(
        lambda x: rolling_zscore(x, window)
    )
    df[f"{return_col}_RZ"] = df.groupby("ETF")[return_col].transform(
        lambda x: rolling_zscore(x, window)
    )
    return df


def get_prepared_data(freq: str = "D", zscore_type: str = "full") -> pd.DataFrame:
    """
    Main entry point: load data, compute returns, aggregate, normalize.

    Parameters:
        freq: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'QE' (quarterly)
        zscore_type: 'full' (full-sample z-score) or 'rolling' (252-day rolling)

    Returns:
        DataFrame with original and normalized columns.
    """
    df = load_all_etfs()
    df = add_returns(df)

    if freq != "D":
        df = aggregate_to_frequency(df, freq)
        flow_col = "Flow_Sum"
        return_col = "Return_Cum"
    else:
        flow_col = "Fund_Flow"
        return_col = "Return"

    if zscore_type == "full":
        df = add_zscore_columns(df, flow_col=flow_col, return_col=return_col)
    else:
        window = {"D": 252, "W": 52, "ME": 12, "QE": 4}.get(freq, 252)
        df = add_rolling_zscore_columns(df, window=window,
                                        flow_col=flow_col, return_col=return_col)

    return df
