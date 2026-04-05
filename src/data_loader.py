"""Data loading, feature engineering, and normalization for ARK ETF fund flows."""
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "input"

EXCEL_PATH = DATA_DIR / "ARK ETF Fund Flows.xlsx"
ARK_AUM_PATH = DATA_DIR / "ARK ETF AUM.xlsx"

ETF_NAMES = ["ARKK", "ARKF", "ARKG", "ARKX", "ARKQ", "ARKW"]

_BBG_TICKER_RE = re.compile(r"^(\w+)\s+US\s+Equity")


# ============================================================
# Bloomberg wide-format peer file loading
# ============================================================

def _parse_bbg_ticker(col: str) -> str | None:
    """Extract ticker from Bloomberg column header.

    'ARKK US Equity - Last Price (R1)' → 'ARKK'
    'ARKK US Equity - Fund Flow (L2)' → 'ARKK'
    'ROBO US Equity  - Fund Flow '    → 'ROBO'
    """
    if not isinstance(col, str):
        return None
    m = _BBG_TICKER_RE.match(col.strip())
    return m.group(1) if m else None


def _load_bbg_wide_sheet(path: Path, sheet: str, value_name: str) -> pd.DataFrame:
    """Load a wide Bloomberg sheet and melt to long format.

    Returns DataFrame with columns: Date, ETF, {value_name}.
    Rows with NaN values are dropped.
    """
    wide = pd.read_excel(path, sheet_name=sheet)
    wide.rename(columns={wide.columns[0]: "Date"}, inplace=True)
    wide["Date"] = pd.to_datetime(wide["Date"], errors="coerce")
    wide = wide.dropna(subset=["Date"])

    # Parse ticker from each non-Date column
    col_map = {}
    for col in wide.columns[1:]:
        ticker = _parse_bbg_ticker(col)
        if ticker:
            col_map[col] = ticker

    # Select and rename columns
    subset = wide[["Date"] + list(col_map.keys())].copy()
    subset.columns = ["Date"] + list(col_map.values())

    # Handle duplicate ticker columns (same ETF may appear twice in a sheet)
    subset = subset.T.groupby(level=0).first().T

    # Melt to long format
    etf_cols = [c for c in subset.columns if c != "Date"]
    long = subset.melt(id_vars="Date", value_vars=etf_cols,
                       var_name="ETF", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=[value_name])
    long["Date"] = pd.to_datetime(long["Date"])
    return long


def _parse_peers_list(path: Path) -> list[str]:
    """Extract peer tickers from the 'peers list' sheet.

    Handles varying header formats across Bloomberg exports.
    Returns list of non-ARK peer tickers.
    """
    df = pd.read_excel(path, sheet_name="peers list", header=None)
    tickers = []
    for _, row in df.iterrows():
        cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if "US Equity" in cell:
            m = _BBG_TICKER_RE.match(cell)
            if m:
                ticker = m.group(1)
                if ticker not in ETF_NAMES:
                    tickers.append(ticker)
    return tickers


def discover_peer_files() -> dict[str, Path]:
    """Find all Bloomberg peer files in DATA_DIR.

    Looks for files matching '*peers.xlsx' or '*peers*.xlsx'.
    Returns dict mapping ARK ticker to file path, e.g. {'ARKK': Path(...)}.
    """
    result = {}
    for p in sorted(DATA_DIR.glob("*peers*.xlsx")):
        if p.name.startswith("~"):
            continue
        # Extract ARK ticker from filename: "ARKK peers.xlsx" → "ARKK"
        name_part = p.stem.split("peers")[0].strip().upper()
        if name_part in ETF_NAMES:
            result[name_part] = p
    return result


def load_bbg_peer_data() -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Load all Bloomberg peer files.

    Returns:
        peer_mapping: {ARK_ticker: [peer_tickers]} (excludes ARK ETFs)
        data_df: long-format DataFrame [Date, ETF, Close, Fund_Flow]
                 (peers only, ARK ETFs excluded)
        aum_df:  long-format DataFrame [Date, ETF, AUM]
                 (all ETFs including ARK)
    """
    peer_files = discover_peer_files()
    if not peer_files:
        return {}, pd.DataFrame(), pd.DataFrame()

    peer_mapping = {}
    all_price = []
    all_flow = []
    all_aum = []

    for ark_etf, path in peer_files.items():
        logger.info("Loading peer file: %s", path.name)
        peers = _parse_peers_list(path)
        peer_mapping[ark_etf] = peers

        try:
            price = _load_bbg_wide_sheet(path, "price", "Close")
            all_price.append(price)
        except Exception as e:
            logger.warning("Failed to load price sheet from %s: %s", path.name, e)

        # Try standardized name first, fall back to legacy name
        for flow_sheet in ("fund flow", "fundflow"):
            try:
                flow = _load_bbg_wide_sheet(path, flow_sheet, "Fund_Flow")
                all_flow.append(flow)
                break
            except Exception:
                continue
        else:
            logger.warning("Failed to load fund flow sheet from %s", path.name)

        for aum_sheet in ("total assets", "totalassets"):
            try:
                aum = _load_bbg_wide_sheet(path, aum_sheet, "AUM")
                all_aum.append(aum)
                break
            except Exception:
                continue
        else:
            logger.warning("Failed to load total assets sheet from %s", path.name)

    # Combine and deduplicate
    if all_price:
        price_df = pd.concat(all_price, ignore_index=True).drop_duplicates(
            subset=["Date", "ETF"], keep="first")
    else:
        price_df = pd.DataFrame(columns=["Date", "ETF", "Close"])

    if all_flow:
        flow_df = pd.concat(all_flow, ignore_index=True).drop_duplicates(
            subset=["Date", "ETF"], keep="first")
    else:
        flow_df = pd.DataFrame(columns=["Date", "ETF", "Fund_Flow"])

    if all_aum:
        aum_df = pd.concat(all_aum, ignore_index=True).drop_duplicates(
            subset=["Date", "ETF"], keep="first")
    else:
        aum_df = pd.DataFrame(columns=["Date", "ETF", "AUM"])

    # Merge price and flow into one DataFrame
    if not price_df.empty and not flow_df.empty:
        data_df = price_df.merge(flow_df, on=["Date", "ETF"], how="outer")
    elif not price_df.empty:
        data_df = price_df
    elif not flow_df.empty:
        data_df = flow_df
    else:
        data_df = pd.DataFrame(columns=["Date", "ETF", "Close", "Fund_Flow"])

    # Filter out ARK ETFs from data_df (they're loaded from ARK Excel separately)
    data_df = data_df[~data_df["ETF"].isin(ETF_NAMES)].copy()

    return peer_mapping, data_df, aum_df


# ============================================================
# Dynamic peer list initialization
# ============================================================

def _build_peer_names() -> list[str]:
    """Build PEER_ETF_NAMES dynamically from peer files."""
    peer_files = discover_peer_files()
    if not peer_files:
        return []
    all_peers = set()
    for path in peer_files.values():
        try:
            all_peers.update(_parse_peers_list(path))
        except Exception:
            continue
    return sorted(all_peers)


PEER_ETF_NAMES = _build_peer_names()
ALL_ETF_NAMES = ETF_NAMES + PEER_ETF_NAMES
PEER_MAPPING: dict[str, list[str]] = {}  # populated by load_bbg_peer_data()


# ============================================================
# ARK ETF loading (original format, unchanged)
# ============================================================

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
    """Load all 9 ARK ETF sheets and concatenate into one DataFrame."""
    frames = []
    for name in ETF_NAMES:
        try:
            frames.append(load_single_etf(name))
        except Exception:
            continue
    if not frames:
        logger.warning("No ARK ETFs loaded from %s", EXCEL_PATH)
        return pd.DataFrame(columns=["Date", "Fund_Flow", "Open", "High",
                                      "Low", "Close", "Volume", "ETF"])
    return pd.concat(frames, ignore_index=True)


# ============================================================
# Peer ETF loading (Bloomberg wide format)
# ============================================================

def load_peer_etfs() -> pd.DataFrame:
    """Load peer ETFs from Bloomberg wide-format files.

    Returns DataFrame with same schema as load_all_etfs():
    [Date, Fund_Flow, Open, High, Low, Close, Volume, ETF]
    Open/High/Low are set to Close; Volume is NaN.
    """
    global PEER_MAPPING
    peer_mapping, data_df, _ = load_bbg_peer_data()
    PEER_MAPPING.update(peer_mapping)

    if data_df.empty:
        logger.warning("No peer data loaded from Bloomberg files")
        return pd.DataFrame()

    # Ensure required columns exist
    if "Close" not in data_df.columns:
        data_df["Close"] = np.nan
    if "Fund_Flow" not in data_df.columns:
        data_df["Fund_Flow"] = np.nan

    # Drop rows without price data
    data_df = data_df.dropna(subset=["Close"])

    # Build schema-compatible DataFrame
    result = pd.DataFrame({
        "Date": data_df["Date"],
        "Fund_Flow": data_df["Fund_Flow"],
        "Open": data_df["Close"],
        "High": data_df["Close"],
        "Low": data_df["Close"],
        "Close": data_df["Close"],
        "Volume": np.nan,
        "ETF": data_df["ETF"],
    })
    result = result.sort_values(["ETF", "Date"]).reset_index(drop=True)
    return result


def load_all_etfs_with_peers() -> pd.DataFrame:
    """Load all ARK ETFs + Bloomberg peer ETFs."""
    ark = load_all_etfs()
    peers = load_peer_etfs()
    parts = [df for df in [ark, peers] if len(df) > 0]
    return pd.concat(parts, ignore_index=True)


# ============================================================
# Returns, aggregation, normalization
# ============================================================

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


# ============================================================
# Source flags, benchmarks, AUM
# ============================================================

def add_source_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add Is_ARK boolean column."""
    df = df.copy()
    df["Is_ARK"] = df["ETF"].isin(ETF_NAMES)
    return df


def add_peer_benchmark(df: pd.DataFrame, return_col: str,
                       min_etfs: int = 3) -> pd.DataFrame:
    """Compute per-peer-group mean return as Benchmark_Return.

    For each ARK ETF, the benchmark is the mean return of its OWN peer
    group (from PEER_MAPPING) on each date.  Falls back to cross-sectional
    mean if no peer mapping is available.
    """
    df = df.copy()
    df["Benchmark_Return"] = np.nan

    if PEER_MAPPING:
        for ark_etf, peers in PEER_MAPPING.items():
            ark_mask = df["ETF"] == ark_etf
            if not ark_mask.any():
                continue
            peer_ret = df[df["ETF"].isin(peers)].groupby("Date")[return_col].agg(["mean", "count"])
            peer_ret.columns = ["_peer_ret", "_count"]
            peer_ret.loc[peer_ret["_count"] < min_etfs, "_peer_ret"] = np.nan
            df.loc[ark_mask, "Benchmark_Return"] = (
                df.loc[ark_mask, "Date"].map(peer_ret["_peer_ret"]).values
            )
    else:
        # Fallback: cross-sectional mean (old behavior)
        bench = df.groupby("Date")[return_col].agg(["mean", "count"])
        bench.columns = ["Benchmark_Return", "_count"]
        bench.loc[bench["_count"] < min_etfs, "Benchmark_Return"] = np.nan
        df = df.merge(bench[["Benchmark_Return"]], on="Date", how="left")

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
    """Load AUM data from ARK AUM file and Bloomberg peer files.

    Returns long-format DataFrame with columns: Date, ETF, AUM.
    AUM values are in millions of dollars.
    """
    frames = []

    # 1. ARK ETF AUM from dedicated file
    if ARK_AUM_PATH.exists():
        wide = pd.read_excel(ARK_AUM_PATH, sheet_name=0)
        wide["Date"] = pd.to_datetime(wide["Date"])
        value_cols = [c for c in wide.columns if c != "Date"]
        long = wide.melt(id_vars="Date", value_vars=value_cols,
                         var_name="BBG_Ticker", value_name="AUM")
        long["ETF"] = long["BBG_Ticker"].str.extract(r"^(\w+)\s+US", expand=False)
        long = long.dropna(subset=["ETF", "AUM"])
        long = long[["Date", "ETF", "AUM"]].copy()
        frames.append(long)
    else:
        logger.warning("ARK AUM file not found: %s", ARK_AUM_PATH)

    # 2. Peer AUM from Bloomberg peer files (totalassets sheets)
    peer_files = discover_peer_files()
    for ark_etf, path in peer_files.items():
        for aum_sheet in ("total assets", "totalassets"):
            try:
                aum = _load_bbg_wide_sheet(path, aum_sheet, "AUM")
                aum = aum[~aum["ETF"].isin(ETF_NAMES)].copy()
                frames.append(aum)
                break
            except Exception:
                continue
        else:
            logger.warning("Failed to load AUM from %s", path.name)

    if not frames:
        return pd.DataFrame(columns=["Date", "ETF", "AUM"])

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["Date", "ETF"], keep="first")
    return result


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
    aum_safe = df["AUM"].replace(0, np.nan)
    df["Flow_Pct"] = df[flow_col] / aum_safe * 100
    return df


# ============================================================
# Main entry points
# ============================================================

def get_prepared_data(freq: str = "D", zscore_type: str = "full") -> pd.DataFrame:
    """
    Main entry point: load ARK-only data, compute returns, aggregate, normalize.

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
