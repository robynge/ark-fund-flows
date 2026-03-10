"""
Download OHLCV data for peer fund ETFs from yfinance.
Saves to 'Peer Fund Flows.xlsx' with one sheet per ETF,
matching the format of 'ARK ETF Fund Flows.xlsx'.

Fund Flow column is left as NaN (yfinance doesn't provide it).
Skips tickers that already have data in the output file.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
import time
import sys

PEER_FUNDS_PATH = Path(__file__).parent / "peer funds.xlsx"
OUTPUT_PATH = Path(__file__).parent / "Peer Fund Flows.xlsx"
ARK_TICKERS = {"ARKK", "ARKF", "ARKG", "ARKX", "ARKB", "ARKQ", "ARKW", "PRNT", "IZRL"}

# Columns to match ARK format: Date, Fund Flow, Open, High, Low, Close, Volume
OUTPUT_COLUMNS = ["Date", "Fund Flow", "Open", "High", "Low", "Close", "Volume"]


def read_peer_tickers() -> list[str]:
    """Read ticker list from peer funds Excel."""
    df = pd.read_excel(PEER_FUNDS_PATH, sheet_name="Sheet3", header=None)
    tickers = [t.replace(" US Equity", "").strip() for t in df[0].tolist()]
    # Exclude ARK tickers (we already have their data)
    tickers = [t for t in tickers if t not in ARK_TICKERS]
    return tickers


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


def download_etf(ticker: str) -> pd.DataFrame | None:
    """Download full history OHLCV for one ETF from yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=False)

        if hist.empty or len(hist) < 5:
            print(f"  {ticker}: no data or too few rows ({len(hist)})")
            return None

        # Build output DataFrame matching ARK format
        df = pd.DataFrame({
            "Date": hist.index.tz_localize(None),  # remove timezone
            "Fund Flow": float("nan"),
            "Open": hist["Open"].values,
            "High": hist["High"].values,
            "Low": hist["Low"].values,
            "Close": hist["Close"].values,
            "Volume": hist["Volume"].values,
        })

        # Sort descending by date (matching ARK format)
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")
        return None


def main():
    tickers = read_peer_tickers()
    print(f"Peer tickers to download: {len(tickers)}")
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
        df = download_etf(ticker)
        if df is not None:
            new_data[ticker] = df
            print(f"OK - {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
        # Small delay to be nice to Yahoo
        if i < len(to_download):
            time.sleep(0.5)

    # Merge existing + new data
    all_data = {**existing, **new_data}

    # Write to Excel (one sheet per ETF, in ticker order)
    if not all_data:
        print("\nNo data downloaded. Check ticker validity.")
        return

    # Sort sheets: tickers in the order they appear in the peer funds list
    ordered_sheets = [t for t in tickers if t in all_data]
    # Add any extras from existing that might not be in current ticker list
    for t in all_data:
        if t not in ordered_sheets:
            ordered_sheets.append(t)

    print(f"\nWriting {len(ordered_sheets)} sheets to {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for ticker in ordered_sheets:
            all_data[ticker].to_excel(writer, sheet_name=ticker, index=False)

    print("Done!")

    # Summary
    print(f"\n{'='*50}")
    print(f"{'Ticker':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}")
    print(f"{'-'*50}")
    for ticker in ordered_sheets:
        df = all_data[ticker]
        start = df["Date"].min()
        end = df["Date"].max()
        print(f"{ticker:<8} {len(df):>6}  {start.date()!s:>12}  {end.date()!s:>12}")


if __name__ == "__main__":
    main()
