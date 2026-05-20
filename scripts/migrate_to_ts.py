"""Migrate legacy data stores to the Universal TS Store.

Reads from data/raw/ and data/features/ and writes to data/ts/.

Usage:
    .venv/bin/python3 scripts/migrate_to_ts.py
    .venv/bin/python3 scripts/migrate_to_ts.py --symbol btcusdt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ts_store import write as ts_write

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("migrate")

EXCHANGE = "binance"
RAW_ROOT = Path("data/raw")
FEATURES_ROOT = Path("data/features")


def migrate_raw(symbol: str) -> int:
    """Migrate raw OHLCV from data/raw/ to TS Store. Returns rows migrated."""
    asset_id = f"market:{EXCHANGE}:{symbol}"
    src_dir = RAW_ROOT / EXCHANGE / symbol
    if not src_dir.exists():
        logger.warning("No raw data for %s", symbol)
        return 0

    total = 0
    for p in sorted(src_dir.glob("*.parquet")):
        if p.name == ".gitkeep":
            continue
        try:
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            n = ts_write(asset_id, df, frequency="raw")
            total += n
            logger.info("  raw: %s → %s (%d rows)", p.name, f"raw/{p.name}", n)
        except Exception as e:
            logger.warning("  raw: %s failed: %s", p.name, e)

    return total


def migrate_features(symbol: str) -> int:
    """Migrate features from data/features/ to TS Store. Returns rows migrated."""
    asset_id = f"market:{EXCHANGE}:{symbol}"
    src_dir = FEATURES_ROOT / symbol
    if not src_dir.exists():
        return 0

    total = 0
    for p in sorted(src_dir.glob("*.parquet")):
        if p.name == ".gitkeep":
            continue
        try:
            df = pd.read_parquet(p)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            n = ts_write(asset_id, df, frequency="features")
            total += n
            logger.info("  features: %s → features/%s (%d rows)", p.name, p.name, n)
        except Exception as e:
            logger.warning("  features: %s failed: %s", p.name, e)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy stores to TS Store")
    parser.add_argument("--symbol", default=None, help="Single symbol to migrate (default: all)")
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol.lower()]
    else:
        symbols = sorted(d.name for d in (RAW_ROOT / EXCHANGE).iterdir()
                        if d.is_dir() and d.name != ".gitkeep")

    total_raw = 0
    total_feat = 0
    for sym in symbols:
        logger.info("Migrating %s...", sym)
        n = migrate_raw(sym)
        total_raw += n
        n = migrate_features(sym)
        total_feat += n

    logger.info("Done: %d raw rows, %d feature rows migrated", total_raw, total_feat)

    # Refresh catalog
    from src.ts_catalog import refresh_catalog
    refresh_catalog()
    logger.info("Catalog refreshed")


if __name__ == "__main__":
    main()
