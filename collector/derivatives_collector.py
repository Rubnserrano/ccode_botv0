"""Derivatives data collector — standalone script for scheduled data accumulation.

Runs hourly (via cron/systemd) to accumulate Binance Futures data.
Idempotent: safe to run multiple times, no duplicate data.

Usage:
    python collector/derivatives_collector.py --symbol BTCUSDT --period 1h

Recommended schedule:
    # crontab: run every hour at :05
    5 * * * * cd /home/rserrano/project/ccode_botv0 && \
        .venv/bin/python3 collector/derivatives_collector.py >> data/logs/collector.log 2>&1
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("derivatives_collector")


def collect(symbol: str = "BTCUSDT", period: str = "1h") -> dict[str, int]:
    """Collect all derivatives data and align. Returns {source: row_count}."""
    from src.intelligence.binance_futures import (
        fetch_funding_rate_history,
        fetch_open_interest_history,
        fetch_taker_ratio_history,
        fetch_long_short_ratio,
    )

    results = {}

    # Funding rate (8h updates, fetch last 2 days)
    try:
        df = fetch_funding_rate_history(symbol, days=2)
        results["funding_rate"] = len(df)
    except Exception as e:
        logger.error("funding_rate: %s", e)
        results["funding_rate"] = 0

    # OI (1h updates, fetch last 2 days)
    try:
        df = fetch_open_interest_history(symbol, period=period, days=2)
        results["open_interest"] = len(df)
    except Exception as e:
        logger.error("open_interest: %s", e)
        results["open_interest"] = 0

    # Taker ratio (1h updates)
    try:
        df = fetch_taker_ratio_history(symbol, period=period, days=2)
        results["taker_ratio"] = len(df)
    except Exception as e:
        logger.error("taker_ratio: %s", e)
        results["taker_ratio"] = 0

    # Long/short ratio (1h updates)
    try:
        df = fetch_long_short_ratio(symbol, period=period, days=2)
        results["long_short_ratio"] = len(df)
    except Exception as e:
        logger.error("long_short_ratio: %s", e)
        results["long_short_ratio"] = 0

    return results


def align(symbol: str = "btcusdt", timeframe: str = "15m") -> dict[str, int]:
    """Align derivatives data to target timeframe. Returns {source: row_count}."""
    from src.intelligence.aligner import align_derivatives

    results = align_derivatives(symbol, timeframe=timeframe)
    return {name: len(df) for name, df in results.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and align Binance Futures derivatives data"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--period", default="1h",
        help="OHLCV period: 5m, 15m, 1h (default: 1h)"
    )
    parser.add_argument(
        "--align-only", action="store_true",
        help="Only align existing data (skip fetch)"
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Only fetch new data (skip align)"
    )
    args = parser.parse_args()

    logger.info(
        "=== Derivatives collector started: %s period=%s ===",
        args.symbol, args.period
    )

    collect_results = {}
    align_results = {}

    if not args.align_only:
        logger.info("Fetching derivatives data...")
        collect_results = collect(args.symbol, args.period)
        for name, count in collect_results.items():
            logger.info("  %s: %d rows", name, count)

    if not args.fetch_only:
        logger.info("Aligning derivatives data...")
        align_results = align(args.symbol.lower(), timeframe=args.period)
        for name, count in align_results.items():
            logger.info("  aligned/%s: %d rows", name, count)

    total_fetched = sum(collect_results.values())
    total_aligned = sum(align_results.values())

    logger.info(
        "=== Done: %d fetched, %d aligned ===",
        total_fetched, total_aligned
    )


if __name__ == "__main__":
    main()