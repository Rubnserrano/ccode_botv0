"""Universal TimeSeries Aligner — resample any asset to standard frequencies.

Each source type has a default aggregation method:
  - ``market``   → ohlc  (open, high, low, close, volume)
  - ``sentiment`` → avg   (mean of values per bucket)
  - ``macro``    → ffill (forward fill last value)
  - ``news``     → avg   (mean + count of events per bucket)
  - ``derivatives`` → avg (mean of values per bucket)
  - ``onchain``  → sum   (sum of values per bucket)

Usage:
    from src.ts_aligner import align

    align("market:binance:btcusdt", to_freq="15m")
    align("sentiment:alternative:fear_greed", to_freq="15m")
    align_all("15m")  # align all assets to 15m
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ts_store import read as ts_read, write as ts_write, _parse_asset_id

logger = logging.getLogger(__name__)

# Standard target frequencies and their pandas aliases
FREQ_MAP = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

# Default alignment method per source_type
METHOD_MAP = {
    "market": "ohlc",
    "sentiment": "avg",
    "macro": "ffill",
    "news": "avg",
    "derivatives": "avg",
    "onchain": "sum",
}


def align(
    asset_id: str,
    to_freq: str = "15m",
    method: Optional[str] = None,
    from_freq: str = "raw",
) -> int:
    """Resample an asset's raw data to a standard frequency.

    Args:
        asset_id: Asset identifier (e.g. ``market:binance:btcusdt``).
        to_freq: Target frequency (``15m``, ``1h``, ``1d``).
        method: Aggregation method. Auto-detected from source_type if None.
        from_freq: Source frequency to read.

    Returns:
        Number of rows written to aligned store.
    """
    freq_alias = FREQ_MAP.get(to_freq)
    if freq_alias is None:
        logger.warning("aligner: unknown target frequency '%s'", to_freq)
        return 0

    df = ts_read(asset_id, frequency=from_freq)
    if df.empty:
        logger.info("aligner: no raw data for '%s'", asset_id)
        return 0

    st, _src, _name = _parse_asset_id(asset_id)
    if method is None:
        method = METHOD_MAP.get(st, "ffill")

    df = df.set_index("ts").sort_index()

    if method == "ohlc":
        ohlc_cols = {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
        use_cols = {k: v for k, v in ohlc_cols.items() if k in df.columns}
        other_cols = {c: "mean" for c in df.columns
                      if c not in ohlc_cols and c != "ts"}
        agg_dict = {**use_cols, **other_cols}
        if not agg_dict:
            agg_dict = {"close": "last"}
        resampled = df.resample(freq_alias).agg(agg_dict).dropna(how="all")

    elif method == "avg":
        resampled = df.resample(freq_alias).mean()
        # Always include event count
        if not resampled.empty:
            count_col = f"{list(df.columns)[0]}_count" if len(df.columns) == 1 else "event_count"
            resampled[count_col] = df.resample(freq_alias).size()

    elif method == "sum":
        resampled = df.resample(freq_alias).sum()

    elif method == "ffill":
        resampled = df.resample(freq_alias).ffill()

    elif method == "last":
        resampled = df.resample(freq_alias).last()

    else:
        logger.warning("aligner: unknown method '%s', using ffill", method)
        resampled = df.resample(freq_alias).ffill()

    resampled = resampled.reset_index()
    resampled["ts"] = resampled["ts"].dt.tz_convert("UTC")

    n = ts_write(asset_id, resampled, frequency=to_freq)
    logger.info("aligner: '%s' %s → %s: %d rows (method=%s)", asset_id, from_freq, to_freq, n, method)
    return n


def align_all(to_freq: str = "15m", source_type: Optional[str] = None) -> int:
    """Align all assets (optionally filtered by source_type) to a target frequency.

    Returns:
        Total rows written across all assets.
    """
    from src.ts_store import catalog
    cat = catalog()
    if cat.empty:
        return 0

    if source_type:
        cat = cat[cat["source_type"] == source_type]

    total = 0
    for _, row in cat.iterrows():
        aid = row["asset_id"]
        freq = row["frequency"]
        if freq == to_freq:
            continue
        if freq == "raw":
            total += align(aid, to_freq=to_freq)

    return total
