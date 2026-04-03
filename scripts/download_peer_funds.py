"""
Download OHLCV data for tech peer ETFs from yfinance, merge with
Bloomberg fund flow data from 'Tech Peers.xlsx'.

Saves to 'Peer Fund Flows.xlsx' with one sheet per ETF,
matching the format of 'ARK ETF Fund Flows.xlsx':
  Date | Fund Flow | Open | High | Low | Close | Volume

Skips tickers that already have data in the output file.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
import time

TECH_PEERS_PATH = Path(__file__).parent.parent / "data" / "input" / "Tech Peers.xlsx"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "input" / "Peer Fund Flows.xlsx"
ARK_TICKERS = {"ARKK", "ARKF", "ARKG", "ARKX", "ARKQ", "ARKW"}


def read_peer_tickers_and_flows() -> tuple[list[str], dict[str, pd.Series]]:
    """Read ticker list and fund flow data from tech peers Excel.

    Returns:
        tickers: list of peer ticker strings (excluding ARK)
        flows: dict mapping ticker -> Series(index=Date, values=fund_flow)
    """
    df = pd.read_excel(TECH_PEERS_PATH, sheet_name="Sheet1")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    tickers = []
    flows = {}
    for col in df.columns:
        ticker = col.replace(" US Equity", "").strip()
        if ticker in ARK_TICKERS:
            continue
        tickers.append(ticker)
        flows[ticker] = df[col]

    return tickers, flows


def load_existing_data() -> dict[str, pd.DataFrame]:
    """Load already-downloaded data from output file, if it exists."""
    if not OUTPUT_PATH.exists():
        return {}
    existing = {}
    xls = pd.ExcelFile(OUTPUT_PATH)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if len(df) > 0:
            existing[sheet] = df
    return existing


def download_etf(ticker: str, flow_series: pd.Series) -> pd.DataFrame | None:
    """Download OHLCV from yfinance and merge with Bloomberg fund flows."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=False)

        if hist.empty or len(hist) < 5:
            print(f"  {ticker}: no data or too few rows ({len(hist)})")
            return None

        # Build OHLCV DataFrame
        ohlcv = pd.DataFrame({
            "Date": hist.index.tz_localize(None),
            "Open": hist["Open"].values,
            "High": hist["High"].values,
            "Low": hist["Low"].values,
            "Close": hist["Close"].values,
            "Volume": hist["Volume"].values,
        })

        # Merge with Bloomberg fund flows
        flow_df = flow_series.dropna().reset_index()
        flow_df.columns = ["Date", "Fund Flow"]
        flow_df["Date"] = pd.to_datetime(flow_df["Date"])

        merged = ohlcv.merge(flow_df, on="Date", how="left")
        # Reorder columns to match ARK format
        merged = merged[["Date", "Fund Flow", "Open", "High", "Low", "Close", "Volume"]]
        # Sort descending by date (matching ARK format)
        merged = merged.sort_values("Date", ascending=False).reset_index(drop=True)
        return merged

    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")
        return None


def main():
    tickers, flows = read_peer_tickers_and_flows()
    print(f"Tech peer tickers: {len(tickers)}")
    print(f"  {', '.join(tickers)}")

    # Load existing data to avoid re-downloading
    existing = load_existing_data()
    if existing:
        print(f"\nAlready have data for {len(existing)} tickers: {', '.join(existing.keys())}")

    # Determine which tickers need downloading
    to_download = [t for t in tickers if t not in existing]
    already_have = [t for t in tickers if t in existing]

    if not to_download:
        print("\nAll tickers already downloaded. Nothing to do.")
        print(f"Output file: {OUTPUT_PATH}")
        return

    print(f"\nSkipping {len(already_have)} already-downloaded tickers")
    print(f"Downloading {len(to_download)} new tickers...\n")

    # Download new tickers
    new_data = {}
    for i, ticker in enumerate(to_download, 1):
        print(f"[{i}/{len(to_download)}] {ticker}...", end=" ")
        df = download_etf(ticker, flows[ticker])
        if df is not None:
            new_data[ticker] = df
            n_flows = df["Fund Flow"].notna().sum()
            print(f"OK - {len(df)} rows, {n_flows} flow obs, "
                  f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        if i < len(to_download):
            time.sleep(0.5)

    # Merge existing + new data
    all_data = {**existing, **new_data}

    if not all_data:
        print("\nNo data downloaded. Check ticker validity.")
        return

    # Sort sheets in ticker order
    ordered_sheets = [t for t in tickers if t in all_data]
    for t in all_data:
        if t not in ordered_sheets:
            ordered_sheets.append(t)

    print(f"\nWriting {len(ordered_sheets)} sheets to {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for ticker in ordered_sheets:
            all_data[ticker].to_excel(writer, sheet_name=ticker, index=False)

    print("Done!")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Ticker':<8} {'Rows':>6}  {'Flows':>6}  {'Start':>12}  {'End':>12}")
    print(f"{'-'*60}")
    for ticker in ordered_sheets:
        df = all_data[ticker]
        n_flows = df["Fund Flow"].notna().sum()
        print(f"{ticker:<8} {len(df):>6}  {n_flows:>6}  "
              f"{df['Date'].min().date()!s:>12}  {df['Date'].max().date()!s:>12}")


if __name__ == "__main__":
    main()
