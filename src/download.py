"""Download historical OHLCV from Binance REST API into the DataStore.

Usage:
    python -m src.download --symbol BTCUSDT --days 365
    python -m src.download --symbol BTCUSDT --interval 1m --days 90
    python -m src.download --fill                          # gap-fill only
    python -m src.download --info                          # show store status
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pandas as pd

from src.ts_store import write as ts_write
from src.store import read, available_range, row_count  # legacy — only for metadata

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com"
MAX_PER_REQUEST = 1000
EXCHANGE = "binance"


def _print_info():
    from src.ts_catalog import get_catalog
    cat = get_catalog(force_refresh=True)
    if cat.empty:
        print("TS Store is empty")
        return
    found = False
    for _, r in cat.iterrows():
        print(f"  {r['asset_id']:30s}  {r['frequency']:10s}  {int(r['rows']):>8,} rows  {str(r.get('min_ts',''))[:10]} → {str(r.get('max_ts',''))[:10]}")
        found = True
    if not found:
        print("TS Store is empty")


_INTERVAL_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
    "3d": "3D", "1w": "1W", "1M": "1ME",
}


def _to_pandas_freq(interval: str) -> str:
    """Convert Binance interval string to pandas-compatible offset alias."""
    return _INTERVAL_MAP.get(interval, interval)


def _ohlcv_to_dataframe(klines: list, interval: str) -> pd.DataFrame:
    """Convert Binance kline list to DataFrame with store-compatible schema."""
    rows = []
    for k in klines:
        rows.append({
            "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        freq = _to_pandas_freq(interval)
        df.set_index("ts", inplace=True)
        df = df.resample(freq).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df.reset_index(inplace=True)
    return df


async def download_symbol(
    symbol: str,
    days: int = 90,
    interval: str = "1m",
    fill: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Download historical OHLCV for a symbol and save to DataStore.

    Returns the DataFrame that was saved. If fill=True, only fetches missing data.
    """
    binance_interval = interval.replace("min", "m")
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    if fill:
        _, latest = available_range(EXCHANGE, symbol)
        if latest is None:
            start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        else:
            start_time = int(latest.timestamp() * 1000) + 60_000
            if start_time >= end_time:
                logger.info(f"{symbol}: already up to date")
                return pd.DataFrame()
    else:
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_klines: list = []
    async with httpx.AsyncClient() as client:
        while start_time < end_time:
            resp = await client.get(
                f"{BINANCE_REST}/api/v3/klines",
                params={
                    "symbol": symbol.upper(),
                    "interval": binance_interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": MAX_PER_REQUEST,
                },
                timeout=30,
            )
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                break
            all_klines.extend(klines)
            last_ts = int(klines[-1][0])
            start_time = last_ts + 60_000
            if progress:
                last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                logger.info(f"{symbol}: fetched {len(klines)} rows, up to {last_dt.isoformat()}")
            if len(klines) < MAX_PER_REQUEST:
                break
            await asyncio.sleep(0.2)

    if not all_klines:
        logger.info(f"{symbol}: no new data")
        return pd.DataFrame()

    df = _ohlcv_to_dataframe(all_klines, interval)
    asset_id = f"market:{EXCHANGE}:{symbol.lower()}"
    n = ts_write(asset_id, df, frequency="raw")
    logger.info(f"{symbol}: saved {n} rows to TS Store (%s)", asset_id)
    return df


async def _run(args):
    if args.info:
        _print_info()
        return
    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    for symbol in symbols:
        print(f"\nDownloading {symbol} ({args.interval}, {args.days}d)...")
        df = await download_symbol(
            symbol,
            days=args.days,
            interval=args.interval,
            fill=args.fill,
            progress=True,
        )
        if not df.empty:
            print(f"  Done: {len(df):,} rows")
    print("\nStore status:")
    _print_info()


def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV from Binance")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol(s), comma-separated")
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--fill", action="store_true", help="Only download missing data (gap-fill)")
    parser.add_argument("--info", action="store_true", help="Show DataStore status and exit")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
