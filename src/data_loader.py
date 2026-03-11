"""Data loading, feature engineering, and normalization for ARK ETF fund flows."""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

logger = logging.getLogger(__name__)

EXCEL_PATH = Path(__file__).parent.parent / "ARK ETF Fund Flows.xlsx"
PEER_EXCEL_PATH = Path(__file__).parent.parent / "Peer Fund Flows.xlsx"
ARK_AUM_PATH = Path(__file__).parent.parent / "ARK ETF AUM.xlsx"
PEER_AUM_PATH = Path(__file__).parent.parent / "Peers AUM.xlsx"
DATA_DIR = Path(__file__).parent.parent / "data"

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


def load_market_benchmark(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download SPY or QQQ daily data via yfinance, cache to CSV.

    Returns DataFrame with columns: Date, Benchmark_Return (daily return).
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{ticker}.csv"

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        cached_end = cached["Date"].max()
        # Re-download if cache is more than 3 days stale
        if cached_end >= pd.Timestamp(end_date) - pd.Timedelta(days=3):
            return cached

    logger.info("Downloading %s from yfinance...", ticker)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        # Return stale cache if available, otherwise empty
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["Date"])
        return pd.DataFrame(columns=["Date", "Benchmark_Return"])
    if hist.empty:
        logger.warning("No data returned for %s", ticker)
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["Date"])
        return pd.DataFrame(columns=["Date", "Benchmark_Return"])

    bench = pd.DataFrame({
        "Date": hist.index.tz_localize(None),
        "Close": hist["Close"].values,
    })
    bench = bench.sort_values("Date").reset_index(drop=True)
    bench["Benchmark_Return"] = bench["Close"].pct_change()
    bench = bench[["Date", "Benchmark_Return"]].dropna()

    bench.to_csv(cache_path, index=False)
    logger.info("Cached %s data (%d rows) to %s", ticker, len(bench), cache_path)
    return bench


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

    start_date = df["Date"].min().strftime("%Y-%m-%d")
    end_date = (df["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bench = load_market_benchmark(benchmark, start_date, end_date)

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


def _clean_aum_column_name(col: str) -> str:
    """Strip Bloomberg suffixes: 'ARKK US Equity' -> 'ARKK', 'ARKB US Equity  (L3)' -> 'ARKB'."""
    col = col.split("(")[0].strip()  # remove (L3) etc.
    for suffix in [" US Equity", " NA Equity"]:
        col = col.replace(suffix, "")
    return col.strip()


def load_aum_data() -> pd.DataFrame:
    """Load AUM data from ARK and Peers Excel files.

    Returns long-form DataFrame: Date, ETF, AUM (in millions).
    """
    frames = []

    for path, label in [(ARK_AUM_PATH, "ARK"), (PEER_AUM_PATH, "Peers")]:
        if not path.exists():
            logger.warning("AUM file not found: %s", path)
            continue
        raw = pd.read_excel(path, sheet_name="Sheet1")
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date")
        raw.columns = [_clean_aum_column_name(c) for c in raw.columns]
        # Keep only known tickers
        known = [c for c in raw.columns if c in ALL_ETF_NAMES]
        raw = raw[known]
        melted = raw.reset_index().melt(id_vars="Date", var_name="ETF", value_name="AUM")
        frames.append(melted)

    if not frames:
        return pd.DataFrame(columns=["Date", "ETF", "AUM"])

    aum = pd.concat(frames, ignore_index=True)
    # Drop duplicates (prefer ARK file for ARK tickers, loaded first)
    aum = aum.drop_duplicates(subset=["Date", "ETF"], keep="first")
    aum = aum.dropna(subset=["AUM"])
    return aum.sort_values(["ETF", "Date"]).reset_index(drop=True)


def add_aum(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Merge AUM data and compute Flow_Pct = Fund_Flow / AUM."""
    aum = load_aum_data()
    if aum.empty:
        df = df.copy()
        df["AUM"] = np.nan
        df["Flow_Pct"] = np.nan
        return df

    if freq != "D":
        # Aggregate AUM to period-end
        aum = aum.set_index("Date")
        aum = aum.groupby("ETF").resample(freq)["AUM"].last().reset_index()
        aum = aum.dropna(subset=["AUM"])

    df = df.merge(aum[["Date", "ETF", "AUM"]], on=["Date", "ETF"], how="left")

    flow_col = "Flow_Sum" if freq != "D" else "Fund_Flow"
    df["Flow_Pct"] = df[flow_col] / df["AUM"]
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
    df = add_aum(df, freq=freq)

    if zscore_type == "full":
        df = add_zscore_columns(df, flow_col=flow_col, return_col=return_col)
    else:
        window = {"D": 252, "W": 52, "ME": 12, "QE": 4}.get(freq, 252)
        df = add_rolling_zscore_columns(df, window=window,
                                        flow_col=flow_col, return_col=return_col)

    return df
