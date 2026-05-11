"""Build/update Feature Store from raw Parquet data.

Reads raw OHLCV, computes indicators via calc_all(), and writes
the enriched DataFrame to data/features/{symbol}/{YYYY-MM}.parquet.

Incremental: only processes months that don't exist in features yet.
Use --rebuild to force a full rebuild.

Usage:
    python -m src.features.builder --symbol BTCUSDT
    python -m src.features.builder --symbol BTCUSDT --rebuild
    python -m src.features.builder --symbol all
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.store import read as raw_read, available_range as raw_range
from src.features.store import write as f_write, available_range as f_range
from src.indicators.calculator import calc_all

logger = logging.getLogger(__name__)


def build_features(symbol: str, rebuild: bool = False) -> int:
    """Build features for a symbol. Returns rows written."""
    sym = symbol.lower()

    # Determine which data to process
    if rebuild:
        raw_start, raw_end = raw_range("binance", sym)
        start = raw_start
    else:
        f_start, f_end = f_range(sym)
        raw_start, raw_end = raw_range("binance", sym)
        if f_end and raw_end and f_end >= raw_end:
            logger.info("features %s: already up to date (latest=%s)", sym, f_end)
            return 0
        start = f_end if f_end else raw_start

    if start is None:
        logger.info("features %s: no raw data available", sym)
        return 0

    logger.info("features %s: building from %s", sym, start)
    df = raw_read("binance", sym, start=start)
    if df.empty:
        return 0

    t0 = time.time()
    df = calc_all(df)
    calc_time = time.time() - t0

    n = f_write(sym, df)
    logger.info("features %s: wrote %d rows (%.1fM) in %.1fs",
                sym, n, n / 1_000_000, calc_time)
    return n


def build_all(rebuild: bool = False) -> None:
    """Build features for all symbols that have raw data."""
    base = Path("data/raw/binance")
    if not base.exists():
        logger.info("No raw data found")
        return

    from pathlib import Path
    symbols = [d.name for d in base.iterdir() if d.is_dir() and d.name != ".gitkeep"]
    total = 0
    for sym in symbols:
        total += build_features(sym, rebuild)
    logger.info("features: total rows written: %d", total)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Build Feature Store from raw Parquet")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol or 'all'")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    args = parser.parse_args()

    if args.symbol == "all":
        build_all(rebuild=args.rebuild)
    else:
        n = build_features(args.symbol, rebuild=args.rebuild)
        if n:
            print(f"Done: {n:,} rows written to data/features/{args.symbol.lower()}/")


if __name__ == "__main__":
    main()
