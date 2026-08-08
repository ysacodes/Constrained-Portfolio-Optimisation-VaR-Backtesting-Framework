"""
Phase 1: Data acquisition & cleaning
Step 1 — pull raw adjusted prices and inspect what we actually got
before deciding how to handle any gaps.
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["VOO", "KWEB", "QQQ", "URA", "VT", "GLD"]
START = "2016-01-01"


def fetch_prices(tickers, start):
    """Pull daily adjusted close prices for all tickers in one call."""
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return data


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels to daily log returns.

    Log returns are time-additive (an n-day return is just the sum of
    n daily log returns), which is what the Phase 3+ optimizer and the
    VaR models want to work with -- simple returns don't compound that
    cleanly across a multi-asset portfolio.
    """
    return np.log(prices / prices.shift(1)).dropna()


def inspect(prices: pd.DataFrame):
    print(f"Shape: {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}\n")

    print("First valid trading day per ticker:")
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        print(f"  {col:6s}  {first_valid.date() if first_valid is not None else 'NO DATA'}")

    print("\nMissing values per ticker:")
    print(prices.isna().sum())

    # rows where some (not all) tickers are missing -- these are the
    # interesting gaps, since "all missing" usually just means a holiday
    partial_gaps = prices[prices.isna().any(axis=1) & ~prices.isna().all(axis=1)]
    print(f"\nRows with a partial gap (some tickers NaN, not all): {len(partial_gaps)}")
    if len(partial_gaps) > 0:
        print(partial_gaps.isna().sum())


if __name__ == "__main__":
    prices = fetch_prices(TICKERS, START)
    inspect(prices)
    prices.to_csv("raw_prices.csv")
    print("\nSaved raw_prices.csv")

    log_returns = to_log_returns(prices)
    print(f"\nLog returns shape: {log_returns.shape[0]} rows x {log_returns.shape[1]} tickers")
    log_returns.to_csv("log_returns.csv")
    print("Saved log_returns.csv")