"""Registry for external data sources — CRUD + lazy auto-discovery.

A registered source only costs ~1KB on disk. It is NOT fetched until:
  a) A strategy references its column name
  b) An agent calls POST /{name}/fetch explicitly

Auto-discovery hook for the Strategy Engine evaluator:
  When an indicator is not found in INDICATOR_REGISTRY,
  the evaluator calls try_auto_discover(name) which:
    1. Checks data/sources/{name}.json
    2. If exists and not fetched → triggers fetch + align
    3. Registers the column in INDICATOR_REGISTRY
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pathlib import Path

import pandas as pd

from src.intelligence.models import DataSource, ParseConfig, AlignConfig, RateLimit

logger = logging.getLogger(__name__)

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"


def _source_path(name: str) -> Path:
    return _SOURCES_DIR / f"{name}.json"


def save_source(ds: DataSource) -> Path:
    """Save a data source definition to disk."""
    _SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    path = _source_path(ds.name)
    data = ds.to_dict()
    path.write_text(json.dumps(data, indent=2))
    logger.info("source: saved '%s' (%s)", ds.name, path)
    return path


def load_source(name: str) -> DataSource | None:
    """Load a data source by name. Returns None if not found."""
    path = _source_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        p = data.get("parse", {})
        a = data.get("align", {})
        r = data.get("rate_limit", {})
        return DataSource(
            name=data["name"],
            url=data["url"],
            description=data.get("description", ""),
            params=data.get("params", {}),
            headers=data.get("headers", {}),
            schedule=data.get("schedule", "1h"),
            parse=ParseConfig(
                type=p.get("type", "json"),
                timestamp_field=p.get("timestamp_field", "ts"),
                value_field=p.get("value_field", "value"),
                value_transform=p.get("value_transform", "float"),
            ),
            columns=data.get("columns", {"value": "value"}),
            align=AlignConfig(
                method=a.get("method", "ffill"),
                decay_periods=a.get("decay_periods", 0),
                target_timeframes=a.get("target_timeframes", ["15m"]),
            ),
            rate_limit=RateLimit(
                max_calls_per_hour=r.get("max_calls_per_hour", 60),
                cooldown_seconds=r.get("cooldown_seconds", 60),
            ),
            storage_quota_mb=data.get("storage_quota_mb", 100),
            api_key=data.get("api_key", ""),
            enabled=data.get("enabled", True),
            last_fetch_at=data.get("last_fetch_at", ""),
            total_rows=data.get("total_rows", 0),
        )
    except Exception as e:
        logger.warning("source: failed to load '%s': %s", name, e)
        return None


def list_sources() -> list[dict]:
    """List all registered sources with their metadata."""
    if not _SOURCES_DIR.exists():
        return []
    sources = []
    for p in sorted(_SOURCES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            sources.append({
                "name": data.get("name", p.stem),
                "description": data.get("description", ""),
                "schedule": data.get("schedule", ""),
                "enabled": data.get("enabled", True),
                "last_fetch_at": data.get("last_fetch_at", ""),
                "total_rows": data.get("total_rows", 0),
                "columns": list(data.get("columns", {}).values()),
            })
        except Exception:
            pass
    return sources


def delete_source(name: str) -> None:
    """Delete a data source and all its data."""
    import shutil
    path = _source_path(name)
    if path.exists():
        path.unlink()
    # Remove external data
    for base in [
        Path("data/external") / name,
        Path("data/external_aligned") / "15m" / f"{name}.parquet",
        Path("data/external_aligned") / "1h" / f"{name}.parquet",
    ]:
        if base.exists():
            if base.is_dir():
                shutil.rmtree(base)
            else:
                base.unlink()
    logger.info("source: deleted '%s'", name)


def try_auto_discover(name: str) -> bool:
    """Lazy auto-discovery: find and fetch a source by column name.

    Called by the evaluator when an indicator is not in INDICATOR_REGISTRY.
    If the source exists and has data, it registers the column.

    Returns True if the source was found and data is now available.
    """
    ds = load_source(name)
    if ds is None:
        return False
    if not ds.enabled:
        return False

    # Check if aligned data already exists
    aligned_path = Path("data/external_aligned") / "15m" / f"{name}.parquet"
    if aligned_path.exists():
        # Already fetched — register in INDICATOR_REGISTRY
        _register_column(name)
        return True

    # Not fetched yet — trigger immediate fetch
    try:
        from src.intelligence.fetcher import fetch_source
        df = fetch_source(ds)
        if df.empty:
            logger.warning("source: auto-discover '%s' returned no data", name)
            return False

        from src.intelligence.aligner import align_source
        for tf in ds.align.target_timeframes:
            align_source(df, ds, tf)

        # Update metadata
        ds.total_rows = len(df)
        from datetime import datetime, timezone
        ds.last_fetch_at = datetime.now(timezone.utc).isoformat()
        save_source(ds)

        _register_column(name)
        logger.info("source: auto-discovered '%s' (%d rows)", name, len(df))
        return True
    except Exception as e:
        logger.warning("source: auto-discover '%s' failed: %s", name, e)
        return False


def _register_column(name: str) -> None:
    """Register a source column as an indicator that fetches from aligned data.

    If the column doesn't exist in the DataFrame (no feature rebuild yet),
    it loads from the aligned external Parquet.
    """
    from src.strategy_engine.registry import INDICATOR_REGISTRY
    if name in INDICATOR_REGISTRY:
        return

    def _indicator_fn(df, p, _n=name):
        if _n in df.columns:
            return df[_n]
        # Try loading from aligned external data
        aligned_path = Path("data/external_aligned") / "15m" / f"{_n}.parquet"
        if aligned_path.exists():
            ext = pd.read_parquet(aligned_path)
            ext["ts"] = pd.to_datetime(ext["ts"], utc=True)
            result = df.merge(ext[["ts", _n]], on="ts", how="left")[_n].ffill()
            return result
        return pd.Series(0.0, index=df.index)

    INDICATOR_REGISTRY[name] = _indicator_fn
    logger.info("source: registered column '%s' in INDICATOR_REGISTRY", name)
