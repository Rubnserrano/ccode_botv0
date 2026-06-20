"""Migrate existing data to the Universal TimeSeries Store.

Usage:
    python scripts/migrate_to_ts.py

Migrates:
  - data/raw/binance/btcusdt/       → ts/market/binance_btcusdt/raw/
  - data/raw/binance/*/15m/         → ts/market/binance_*_usdt/aligned/15m/
  - data/external/fear_greed/       → ts/sentiment/alternative_fear_greed/raw/
  - data/external_aligned/15m/      → ts/sentiment/alternative_fear_greed/aligned/15m/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ts_store import write as ts_write, _TS_ROOT

RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external")
ALIGNED_DIR = Path("data/external_aligned")

EXCHANGES = ["binance", "test_exchange"]


def _migrate_ohlcv():
    """Migrate all OHLCV monthly partitions."""
    total = 0
    for exchange in EXCHANGES:
        raw_exchange = RAW_DIR / exchange
        if not raw_exchange.exists():
            continue
        for symbol_dir in sorted(raw_exchange.iterdir()):
            if not symbol_dir.is_dir() or symbol_dir.name == ".gitkeep":
                continue
            symbol = symbol_dir.name
            asset_id = f"market:{exchange}:{symbol.lower()}"

            for parquet_file in sorted(symbol_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(parquet_file)
                    if df.empty:
                        continue
                    n = ts_write(asset_id, df, frequency="raw")
                    total += n
                    print(f"  {asset_id}/raw/{parquet_file.stem}: {n} rows")
                except Exception as e:
                    print(f"  WARN: {parquet_file}: {e}")

    print(f"  Total OHLCV rows migrated: {total}")
    return total


def _migrate_15m():
    """Migrate any existing 15m resampled data."""
    total = 0
    for exchange in EXCHANGES:
        raw_exchange = RAW_DIR / exchange
        if not raw_exchange.exists():
            continue
        for symbol_dir in sorted(raw_exchange.iterdir()):
            if not symbol_dir.is_dir() or symbol_dir.name == ".gitkeep":
                continue
            symbol = symbol_dir.name
            asset_id = f"market:{exchange}:{symbol.lower()}"
            resampled_path = symbol_dir / "15m" / "data.parquet"
            if resampled_path.exists():
                try:
                    df = pd.read_parquet(resampled_path)
                    if df.empty:
                        continue
                    n = ts_write(asset_id, df, frequency="15m")
                    total += n
                    print(f"  {asset_id}/15m: {n} rows (from resampled)")
                except Exception as e:
                    print(f"  WARN: {resampled_path}: {e}")
    return total


def _migrate_external():
    """Migrate external data sources by type detection."""
    if not EXTERNAL_DIR.exists():
        return 0

    # Map known external sources to asset_ids
    SOURCE_MAP = {
        "fear_greed": "sentiment:alternative:fear_greed",
        "vix_test": "macro:fred:vix",
        "fred_vix_test": "macro:fred:vix",
        "test_manual": "macro:test:unknown",
    }

    total = 0
    for src_dir in sorted(EXTERNAL_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        name = src_dir.name
        asset_id = SOURCE_MAP.get(name, f"external:{name}:data")

        for parquet_file in sorted(src_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                if df.empty:
                    continue
                n = ts_write(asset_id, df, frequency="raw")
                total += n
                print(f"  {asset_id}/raw/{parquet_file.stem}: {n} rows")
            except Exception as e:
                print(f"  WARN: {parquet_file}: {e}")

    return total


def _migrate_aligned_external():
    """Migrate already-aligned external data."""
    if not ALIGNED_DIR.exists():
        return 0

    total = 0
    for tf_dir in sorted(ALIGNED_DIR.iterdir()):
        if not tf_dir.is_dir():
            continue
        freq = tf_dir.name

        for parquet_file in sorted(tf_dir.glob("*.parquet")):
            name = parquet_file.stem
            if name == "fear_greed":
                asset_id = "sentiment:alternative:fear_greed"
            else:
                continue

            try:
                df = pd.read_parquet(parquet_file)
                if df.empty:
                    continue
                n = ts_write(asset_id, df, frequency=freq)
                total += n
                print(f"  {asset_id}/{freq}: {n} rows (from aligned external)")
            except Exception as e:
                print(f"  WARN: {parquet_file}: {e}")

    return total


def _refresh_catalog():
    """Rebuild the ts_catalog cache."""
    from src.ts_catalog import refresh_catalog
    cat = refresh_catalog()
    print(f"  Catalog: {len(cat)} entries across {cat['source_type'].nunique()} source types")
    return len(cat)


def main():
    print("=== Migrating to Universal TimeSeries Store ===\n")

    print("Phase 1: OHLCV monthly partitions...")
    ohlcv = _migrate_ohlcv()

    print("\nPhase 2: Resampled 15m data...")
    resampled = _migrate_15m()

    print("\nPhase 3: External raw data...")
    external = _migrate_external()

    print("\nPhase 4: External aligned data...")
    aligned = _migrate_aligned_external()

    print("\nPhase 5: Rebuilding catalog...")
    n_cat = _refresh_catalog()

    grand_total = ohlcv + resampled + external + aligned
    print(f"\n=== Migration complete: {grand_total:,} rows migrated, {n_cat} catalog entries ===")

    print("\nSummary of available assets:")
    from src.ts_catalog import get_catalog
    cat = get_catalog()
    if not cat.empty:
        for _, r in cat.iterrows():
            print(f"  {r['asset_id']:45s}  {r['frequency']:6s}  {r['rows']:>8,} rows  {str(r['min_ts'])[:10]} → {str(r['max_ts'])[:10]}")


if __name__ == "__main__":
    main()
