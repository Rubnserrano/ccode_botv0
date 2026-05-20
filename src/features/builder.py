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
from pathlib import Path

import pandas as pd

from src.ts_store import read as ts_read, write as ts_write
from src.ts_catalog import get_catalog
from src.indicators.calculator import calc_all

logger = logging.getLogger(__name__)


def build_features(symbol: str, rebuild: bool = False) -> int:
    """Build features for a symbol. Returns rows written."""
    sym = symbol.lower()
    asset_id = f"market:binance:{sym}"

    cat = get_catalog()
    asset_rows = cat[cat["asset_id"] == asset_id] if not cat.empty else pd.DataFrame()
    raw_rows = asset_rows[asset_rows["frequency"] == "raw"] if not asset_rows.empty else pd.DataFrame()
    feat_rows = asset_rows[asset_rows["frequency"] == "features"] if not asset_rows.empty else pd.DataFrame()

    # Determine which data to process
    if rebuild:
        start = None
    elif not feat_rows.empty:
        feat_max = feat_rows["max_ts"].max()
        raw_max = raw_rows["max_ts"].max() if not raw_rows.empty else None
        if feat_max and raw_max and feat_max >= raw_max:
            logger.info("features %s: already up to date (latest=%s)", sym, feat_max)
            return 0
        start = pd.Timestamp(feat_max) if feat_max else None
    else:
        start = None

    logger.info("features %s: building from %s", sym, start or "beginning")
    df = ts_read(asset_id, frequency="raw", start=start)
    if df.empty:
        logger.info("features %s: no raw data available", sym)
        return 0

    t0 = time.time()
    df = calc_all(df)

    ext_dir = Path("data/external_aligned/15m")
    if ext_dir.exists():
        for ext_file in sorted(ext_dir.glob("*.parquet")):
            src_name = ext_file.stem
            try:
                ext_df = pd.read_parquet(ext_file)
                ext_df["ts"] = pd.to_datetime(ext_df["ts"], utc=True)
                df = df.merge(ext_df, on="ts", how="left")
                logger.info("features: merged external '%s' (%d cols)", src_name, len(ext_df.columns) - 1)
            except Exception as e:
                logger.warning("features: failed to merge '%s': %s", src_name, e)

    # Merge derivatives data (funding_rate, open_interest, taker_ratio, long_short_ratio)
    deriv_dir = Path("data/ts/derivatives/aligned/15m")
    if deriv_dir.exists():
        for deriv_file in sorted(deriv_dir.glob("*.parquet")):
            src_name = deriv_file.stem
            try:
                deriv_df = pd.read_parquet(deriv_file)
                deriv_df["ts"] = pd.to_datetime(deriv_df["ts"], utc=True)
                df = df.merge(deriv_df, on="ts", how="left")
                logger.info("features: merged derivatives '%s' (%d cols)", src_name, len(deriv_df.columns) - 1)
            except Exception as e:
                logger.warning("features: failed to merge derivatives '%s': %s", src_name, e)

    # Fill NaN in derivatives columns with forward fill then backward fill
    deriv_cols = ["funding_rate", "open_interest", "open_interest_value",
                  "taker_buy_vol", "taker_sell_vol", "taker_ratio",
                  "long_account", "short_account", "long_short_ratio"]
    for col in deriv_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    calc_time = time.time() - t0

    n = ts_write(asset_id, df, frequency="features")
    logger.info("features %s: wrote %d rows (%.1fM) in %.1fs",
                sym, n, n / 1_000_000, calc_time)
    return n


def build_all(rebuild: bool = False) -> None:
    """Build features for all symbols found in TS Store."""
    cat = get_catalog()
    if cat.empty:
        logger.info("No assets found in TS Store")
        return

    raw_assets = cat[cat["frequency"] == "raw"]["asset_id"].unique()
    symbols = [aid.split(":")[-1] for aid in raw_assets]
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
