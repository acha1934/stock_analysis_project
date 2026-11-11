"""Download Indian stock market data using yfinance.

Downloads historical OHLCV data for ~20 major NSE-listed stocks
and saves a combined CSV to ../data/indian_stocks.csv
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# Major Indian stocks across sectors
TICKERS = [
    "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "WIPRO.NS",
    "HCLTECH.NS", "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HINDUNILVR.NS",
]

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "indian_stocks.csv"


def main():
    print(f"Downloading data for {len(TICKERS)} stocks from {START_DATE} to {END_DATE}...")

    frames = []
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df.empty:
                print(f"  [SKIP] No data for {ticker}")
                continue
            df = df.reset_index()
            df["Ticker"] = ticker
            # Keep only needed columns, rename to lowercase
            df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            frames.append(df)
            print(f"  [OK] {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH}")
    print(f"Stocks: {combined['Ticker'].nunique()} | Date range: {combined['Date'].min()} to {combined['Date'].max()}")


if __name__ == "__main__":
    main()
