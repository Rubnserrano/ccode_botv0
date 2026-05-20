"""Data Catalog — metadata registry for all time-series assets.

Provides a fast, queryable view of what data exists without scanning Parquet files.
Powered by ``ts_store.catalog()`` with optional caching.

Usage:
    from src.ts_catalog import get_catalog, lookup_asset

    cat = get_catalog()
    # Filter by source type
    market_assets = cat[cat["source_type"] == "market"]
    # Find specific asset
    btc = lookup_asset("market:binance:btcusdt", "15m")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ts_store import catalog as _store_catalog

_CATALOG_DIR = Path(__file__).resolve().parents[1] / "data" / "catalog"
_CATALOG_PATH = _CATALOG_DIR / "catalog.parquet"


def get_catalog(force_refresh: bool = False) -> pd.DataFrame:
    """Return the asset catalog (cached as Parquet for fast loading).

    Args:
        force_refresh: If True, rescans all Parquet files instead of using cache.

    Returns:
        DataFrame with columns: asset_id, source_type, frequency,
        min_ts, max_ts, rows, columns.
    """
    if not force_refresh and _CATALOG_PATH.exists():
        try:
            return pd.read_parquet(_CATALOG_PATH)
        except Exception:
            pass

    df = _store_catalog()
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CATALOG_PATH, compression="snappy")
    return df


def refresh_catalog() -> pd.DataFrame:
    """Force a full rescan and return fresh catalog."""
    return get_catalog(force_refresh=True)


def lookup_asset(
    asset_id: str,
    frequency: str = "raw",
) -> dict | None:
    """Return metadata for a specific asset+freq, or None."""
    cat = get_catalog()
    mask = (cat["asset_id"] == asset_id) & (cat["frequency"] == frequency)
    matching = cat[mask]
    if matching.empty:
        return None
    return matching.iloc[0].to_dict()


def list_source_types() -> list[str]:
    """Return all source types with data."""
    cat = get_catalog()
    return sorted(cat["source_type"].unique().tolist()) if not cat.empty else []


def list_assets(source_type: str | None = None) -> list[str]:
    """Return all asset IDs, optionally filtered by source type."""
    cat = get_catalog()
    if source_type:
        cat = cat[cat["source_type"] == source_type]
    return sorted(cat["asset_id"].unique().tolist()) if not cat.empty else []
