"""Multi-asset query — unified DataFrame from multiple sources.

Resolves any asset_id, auto-aligns if needed, renames columns with aliases,
and optionally computes indicators for market assets.

Usage:
    from src.query import get

    df = get({
        "btc": "market:binance:btcusdt",
        "fear": "sentiment:alternative:fear_greed",
    }, frequency="15m", with_indicators=True)

    # Columns: ts, btc_close, btc_rsi_14, btc_regime, fear_fear_greed
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.ts_store import read as ts_read
from src.ts_aligner import align

logger = logging.getLogger(__name__)


def get(
    assets: dict[str, str],
    frequency: str = "15m",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    with_indicators: bool = False,
    columns: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Query multiple assets aligned to the same frequency grid.

    Args:
        assets: Dict mapping ``alias → asset_id``.
            Example: ``{"btc": "market:binance:btcusdt", "vix": "macro:fred:vix"}``
        frequency: Target frequency. Auto-aligns if aligned data missing.
        start: Inclusive start datetime.
        end: Inclusive end datetime.
        limit: Max rows per asset (after alignment).
        with_indicators: If True, compute indicators for ``market`` assets.
        columns: Per-asset column filter. ``{"btc": ["close", "volume"]}``.

    Returns:
        DataFrame with ``ts`` column + ``{alias}_{col}`` for each asset.
        All columns are aligned to the same timestamps (outer join on ts).
    """
    if not assets:
        return pd.DataFrame()

    frames = {}
    for alias, asset_id in assets.items():
        # Read generous data for indicator computation (no limit yet)
        buf = 500 if with_indicators else 0
        df = _read_aligned(asset_id, frequency, start, end, limit=(limit + buf) if limit else None)
        if df.empty:
            logger.warning("query: no data for '%s' (alias=%s)", asset_id, alias)
            continue

        # Compute indicators for market assets if requested
        if with_indicators and asset_id.startswith("market:"):
            try:
                from src.indicators.calculator import calc_all
                df = calc_all(df)
            except Exception as e:
                logger.warning("query: calc_all failed for '%s': %s", asset_id, e)

        # Filter columns per-asset
        col_filter = columns.get(alias) if columns else None
        if col_filter:
            keep = [c for c in col_filter if c in df.columns]
            if "ts" not in keep:
                keep = ["ts"] + keep
            df = df[keep]

        # Rename columns with alias prefix (except ts)
        df = df.rename(columns={c: f"{alias}_{c}" for c in df.columns if c != "ts"})

        frames[alias] = df

    if not frames:
        return pd.DataFrame()

    # Merge all on ts (outer join → all timestamps)
    result = None
    for alias, df in frames.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on="ts", how="outer", suffixes=("", f"_{alias}_dup"))

    result = result.sort_values("ts").reset_index(drop=True)
    result["ts"] = pd.to_datetime(result["ts"], utc=True)

    # Apply limit AFTER merge (so all sources contribute)
    if limit and len(result) > limit:
        result = result.tail(limit)

    return result


def get_regime(
    asset_id: str = "market:binance:btcusdt",
    frequency: str = "15m",
    lookback_bars: int = 96,
) -> dict:
    """Get market regime distribution for a given asset.

    Returns dict like: ``{"trending_up": 0.3, "ranging": 0.5, "trending_down": 0.2}``
    """
    df = _read_aligned(asset_id, frequency, limit=lookback_bars + 200)
    if df.empty or len(df) < 50:
        return {}

    try:
        from src.indicators.calculator import calc_all, detect_regime
        df = calc_all(df)
        df["regime"] = detect_regime(df)
        dist = df["regime"].tail(lookback_bars).value_counts(normalize=True)
        return dist.to_dict()
    except Exception as e:
        logger.warning("query: regime detection failed: %s", e)
        return {}


def list_available(source_type: Optional[str] = None) -> list[dict]:
    """Return all available assets with their metadata."""
    from src.ts_catalog import get_catalog
    cat = get_catalog()
    if cat.empty:
        return []

    if source_type:
        cat = cat[cat["source_type"] == source_type]

    rows = cat[cat["frequency"] == "raw"].to_dict("records") if "frequency" in cat.columns else []
    simplified = []
    for r in rows:
        simplified.append({
            "asset_id": r.get("asset_id", ""),
            "source_type": r.get("source_type", ""),
            "min_ts": str(r.get("min_ts", ""))[:10] if r.get("min_ts") else "",
            "max_ts": str(r.get("max_ts", ""))[:10] if r.get("max_ts") else "",
            "rows": int(r.get("rows", 0)),
            "columns": r.get("columns", []),
        })
    return simplified


def _read_aligned(
    asset_id: str,
    frequency: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read aligned data, auto-aligning if necessary."""
    df = ts_read(asset_id, frequency=frequency, start=start, end=end, limit=limit)

    if df.empty and frequency != "raw":
        logger.info("query: aligning '%s' to %s on-the-fly", asset_id, frequency)
        try:
            align(asset_id, to_freq=frequency)
        except Exception as e:
            logger.warning("query: auto-align failed for '%s': %s", asset_id, e)
            return pd.DataFrame()
        df = ts_read(asset_id, frequency=frequency, start=start, end=end, limit=limit)

    return df
