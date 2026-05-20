"""Resample OHLCV data to higher timeframes.

Reads data from TS Store, resamples to the target interval,
and writes back to TS Store.

Usage:
    python -m src.resample --symbol BTCUSDT --to 5m
    python -m src.resample --symbol BTCUSDT --to 1h --days 30
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.ts_store import read as ts_read, write as ts_write

_INTERVAL_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
}


def resample(
    exchange: str,
    symbol: str,
    to_interval: str,
    days: int | None = None,
) -> pd.DataFrame:
    """Read raw data from TS Store, resample, and return the result."""
    freq = _INTERVAL_MAP[to_interval]
    asset_id = f"market:{exchange}:{symbol.lower()}"
    df = ts_read(asset_id, frequency="raw")
    if df.empty:
        return df
    if days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["ts"] >= cutoff]
    df.set_index("ts", inplace=True)
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    resampled.reset_index(inplace=True)
    resampled["ts"] = resampled["ts"].dt.tz_convert("UTC")
    return resampled


def to_file(
    exchange: str,
    symbol: str,
    to_interval: str,
    days: int | None = None,
) -> int:
    """Resample and write to TS Store.

    Returns row count written.
    """
    df = resample(exchange, symbol, to_interval, days)
    if df.empty:
        return 0
    asset_id = f"market:{exchange}:{symbol.lower()}"
    n = ts_write(asset_id, df, frequency=to_interval)
    return n


def main():
    parser = argparse.ArgumentParser(description="Resample OHLCV to higher timeframe")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    parser.add_argument("--to", required=True, help="Target interval (5m, 15m, 1h, etc)")
    parser.add_argument("--days", type=int, help="Only use last N days of data")
    parser.add_argument("--info", action="store_true", help="Show available intervals and exit")
    args = parser.parse_args()

    if args.info:
        print("Available target intervals:", ", ".join(_INTERVAL_MAP.keys()))
        return

    n = to_file("binance", args.symbol.lower(), args.to, args.days)
    if n:
        print(f"Resampled to {args.to}: {n:,} rows → TS Store")
    else:
        print("No data to resample")


if __name__ == "__main__":
    main()
