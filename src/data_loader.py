"""Data loading, feature engineering, and normalization for ARK ETF fund flows."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "input"

EXCEL_PATH = DATA_DIR / "ARK ETF Fund Flows.xlsx"
PEER_EXCEL_PATH = DATA_DIR / "Peer Fund Flows.xlsx"  # TODO: re-acquire with sector-matched peers
ARK_AUM_PATH = DATA_DIR / "ARK ETF AUM.xlsx"
PEER_AUM_PATH = DATA_DIR / "Peers AUM.xlsx"  # TODO: re-acquire with sector-matched peers

ETF_NAMES = ["ARKK", "ARKF", "ARKG", "ARKX", "ARKB", "ARKQ", "ARKW", "PRNT", "IZRL"]
PEER_ETF_NAMES = [
    "FTXL", "PSI", "SMH", "SOXX", "PTF", "XSD", "PSCT", "IGPT", "KNCT",
    "IXN", "IGM", "IYW", "XLK", "FTEC", "VGT", "TDIV", "QTEC", "FID",
    "FXL", "ERTH", "XT", "GAMR", "CQQQ", "FDN", "HACK", "PNQI", "SKYY",
    "CIBR", "SOCL",
]
ALL_ETF_NAMES = ETF_NAMES + PEER_ETF_NAMES



def _load_single_etf_from_file(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load a single ETF sheet from an Excel file."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = ["Date", "Fund_Flow", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])
    df["ETF"] = sheet_name
    return df


def load_single_etf(sheet_name: str) -> pd.DataFrame:
    """Load a single ETF sheet from the ARK Excel file."""
    return _load_single_etf_from_file(EXCEL_PATH, sheet_name)


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
    Returns DataFrame with columns: Date, ETF, Flow_Sum, Return_Cum, Open_First, High_Max, Low_Min, Close_Last.
    """
    df = df.copy()
    df = df.set_index("Date")
    groups = df.groupby("ETF")

    result_frames = []
    for etf, group in groups:
        resampled = group.resample(freq).agg(
            Flow_Sum=("Fund_Flow", "sum"),
            Return_Cum=("Return", lambda x: (1 + x).prod() - 1),
            Open_First=("Open", "first"),
            High_Max=("High", "max"),
            Low_Min=("Low", "min"),
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


def load_peer_etfs() -> pd.DataFrame:
    """Load all peer ETF sheets from the peer Excel file."""
    if not PEER_EXCEL_PATH.exists():
        logger.warning("Peer file not found: %s", PEER_EXCEL_PATH)
        return pd.DataFrame()
    frames = []
    for name in PEER_ETF_NAMES:
        try:
            frames.append(_load_single_etf_from_file(PEER_EXCEL_PATH, name))
        except Exception as e:
            logger.warning("Failed to load peer ETF %s: %s", name, e)
            continue
    if not frames:
        logger.warning("No peer ETFs loaded from %s", PEER_EXCEL_PATH)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_etfs_with_peers() -> pd.DataFrame:
    """Load all 9 ARK + 29 tech peer ETFs."""
    ark = load_all_etfs()
    peers = load_peer_etfs()
    parts = [df for df in [ark, peers] if len(df) > 0]
    return pd.concat(parts, ignore_index=True)


def add_source_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add Is_ARK boolean column."""
    df = df.copy()
    df["Is_ARK"] = df["ETF"].isin(ETF_NAMES)
    return df


def add_peer_benchmark(df: pd.DataFrame, return_col: str,
                       min_etfs: int = 10) -> pd.DataFrame:
    """
    Compute cross-sectional mean return per date as Benchmark_Return,
    then Excess_Return = Return - Benchmark_Return.
    Dates with fewer than min_etfs ETFs get NaN benchmark.
    """
    df = df.copy()
    bench = df.groupby("Date")[return_col].agg(["mean", "count"])
    bench.columns = ["Benchmark_Return", "_count"]
    bench.loc[bench["_count"] < min_etfs, "Benchmark_Return"] = np.nan
    bench = bench[["Benchmark_Return"]]
    df = df.merge(bench, on="Date", how="left")
    df["Excess_Return"] = df[return_col] - df["Benchmark_Return"]
    return df


def load_market_benchmark(ticker: str) -> pd.DataFrame:
    """Load pre-downloaded SPY or QQQ benchmark data from data/{ticker}.csv.

    Returns DataFrame with columns: Date, Benchmark_Return (daily return).
    """
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        logger.warning("Benchmark file not found: %s", csv_path)
        return pd.DataFrame(columns=["Date", "Benchmark_Return"])
    return pd.read_csv(csv_path, parse_dates=["Date"])


def add_market_benchmark(df: pd.DataFrame, return_col: str,
                         benchmark: str = "SPY",
                         freq: str = "D") -> pd.DataFrame:
    """Compute Excess_Return using market index or peer average.

    Parameters:
        benchmark: "SPY", "QQQ", or "peer_avg"
        freq: frequency code for aggregation of benchmark returns
    """
    if benchmark == "peer_avg":
        return add_peer_benchmark(df, return_col)

    bench = load_market_benchmark(benchmark)

    if bench.empty:
        logger.warning("No benchmark data for %s, falling back to peer average", benchmark)
        return add_peer_benchmark(df, return_col)

    df = df.copy()

    if freq != "D":
        # Aggregate benchmark returns to match data frequency
        bench = bench.set_index("Date")
        bench = bench.resample(freq).agg(
            Benchmark_Return=("Benchmark_Return", lambda x: (1 + x).prod() - 1),
        ).dropna().reset_index()

    df = df.merge(bench[["Date", "Benchmark_Return"]], on="Date", how="left")
    df["Excess_Return"] = df[return_col] - df["Benchmark_Return"]
    return df


def load_aum_data() -> pd.DataFrame:
    """Load AUM data from both ARK and Peers Excel files.

    Returns long-format DataFrame with columns: Date, ETF, AUM.
    AUM values are in millions of dollars.
    """
    frames = []
    for path in [ARK_AUM_PATH, PEER_AUM_PATH]:
        if not path.exists():
            logger.warning("AUM file not found: %s", path)
            continue
        wide = pd.read_excel(path, sheet_name=0)
        wide["Date"] = pd.to_datetime(wide["Date"])
        # Melt to long format
        value_cols = [c for c in wide.columns if c != "Date"]
        long = wide.melt(id_vars="Date", value_vars=value_cols,
                         var_name="BBG_Ticker", value_name="AUM")
        # Extract ETF ticker: "ARKK US Equity" -> "ARKK", "ARKB US Equity  (L3)" -> "ARKB"
        long["ETF"] = long["BBG_Ticker"].str.extract(r"^(\w+)\s+US", expand=False)
        long = long.dropna(subset=["ETF", "AUM"])
        long = long[["Date", "ETF", "AUM"]].copy()
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["Date", "ETF", "AUM"])
    return pd.concat(frames, ignore_index=True)


def merge_aum(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Merge AUM onto the main DataFrame and compute Flow_Pct = Fund_Flow / AUM.

    For aggregated frequencies, AUM is the first value in each period
    (beginning-of-period AUM) to avoid endogeneity with current-period flows.
    Flow_Pct is expressed as a percentage (×100).
    """
    aum = load_aum_data()
    if aum.empty:
        df = df.copy()
        df["AUM"] = np.nan
        df["Flow_Pct"] = np.nan
        return df

    if freq != "D":
        # Aggregate AUM to matching frequency: use first available value per period
        aum = aum.set_index("Date")
        aum_agg = aum.groupby("ETF").resample(freq)["AUM"].first().reset_index()
        aum = aum_agg.dropna(subset=["AUM"])

    df = df.merge(aum[["Date", "ETF", "AUM"]], on=["Date", "ETF"], how="left")

    flow_col = "Fund_Flow" if freq == "D" else "Flow_Sum"
    df["Flow_Pct"] = df[flow_col] / df["AUM"] * 100
    return df


def get_prepared_data_with_peers(freq: str = "D",
                                  zscore_type: str = "full",
                                  benchmark: str = "SPY") -> pd.DataFrame:
    """
    Main entry point for combined ARK + peer data.
    Load → returns → aggregate → benchmark → z-score.

    Parameters:
        benchmark: "SPY", "QQQ", or "peer_avg"
    """
    df = load_all_etfs_with_peers()
    df = add_returns(df)

    if freq != "D":
        df = aggregate_to_frequency(df, freq)
        flow_col = "Flow_Sum"
        return_col = "Return_Cum"
    else:
        flow_col = "Fund_Flow"
        return_col = "Return"

    df = add_source_flag(df)
    df = add_market_benchmark(df, return_col, benchmark=benchmark, freq=freq)
    df = merge_aum(df, freq=freq)

    if zscore_type == "full":
        df = add_zscore_columns(df, flow_col=flow_col, return_col=return_col)
        # Also z-score normalize Flow_Pct if available
        if "Flow_Pct" in df.columns and df["Flow_Pct"].notna().any():
            df["Flow_Pct_Z"] = df.groupby("ETF")["Flow_Pct"].transform(zscore_normalize)
    else:
        window = {"D": 252, "W": 52, "ME": 12, "QE": 4}.get(freq, 252)
        df = add_rolling_zscore_columns(df, window=window,
                                        flow_col=flow_col, return_col=return_col)

    return df
