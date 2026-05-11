"""Generic Fetcher — downloads external data via HTTP.

Supports JSON and CSV APIs. Attempts to fetch up to 1 year of history.
If the full response payload is < 1MB, fetches ALL available history.

Usage:
    from src.intelligence.fetcher import fetch_source
    df = await fetch_source(data_source)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.intelligence.models import DataSource

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "data" / "external"
MAX_PAYLOAD_BYTES = 1_000_000  # 1MB — if response < this, try to get all history


def _navigate_path(data: Any, path: str) -> Any:
    """Navigate a dotted path in nested dict/list structures."""
    parts = path.replace("[", ".").replace("]", "").split(".")
    current = data
    for part in parts:
        if not part:
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _parse_response(text: str, source: DataSource) -> pd.DataFrame:
    """Parse HTTP response body into a DataFrame with timestamp + value columns."""
    parse = source.parse
    rows = []

    if parse.type == "json":
        data = json.loads(text)
        # Navigate to the array of items
        items = _navigate_path(data, parse.timestamp_field.rsplit("[", 1)[0]) if "[" in parse.timestamp_field else data
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = [data]

        for item in items:
            ts_val = _navigate_path(item, parse.timestamp_field.split(".")[-1])
            val_val = _navigate_path(item, parse.value_field.split(".")[-1])
            if ts_val is None:
                continue
            try:
                ts = pd.to_datetime(ts_val, utc=True)
                val = float(val_val) if parse.value_transform == "float" else (
                    int(val_val) if parse.value_transform == "int" else str(val_val)
                )
                rows.append({"ts": ts, **{col: val for col in source.columns.values()}})
            except (ValueError, TypeError):
                continue

    elif parse.type == "csv":
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            ts_val = row.get(parse.timestamp_field)
            val_val = row.get(parse.value_field)
            if ts_val is None or val_val is None:
                continue
            try:
                ts = pd.to_datetime(ts_val, utc=True)
                val = float(val_val) if parse.value_transform == "float" else val_val
                rows.append({"ts": ts, **{col: val for col in source.columns.values()}})
            except (ValueError, TypeError):
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    return df


def fetch_source(source: DataSource) -> pd.DataFrame:
    """Fetch data from an external source.

    Returns a DataFrame with columns: ts + all source.columns values.
    Empty DataFrame if fetch failed.
    """
    # Build request
    params = dict(source.params)
    headers = dict(source.headers)
    if source.api_key:
        # Try common API key patterns
        if "{API_KEY}" in source.url:
            source.url = source.url.replace("{API_KEY}", source.api_key)
        elif "api_key" not in params:
            headers["Authorization"] = f"Bearer {source.api_key}"

    try:
        resp = httpx.get(source.url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("fetcher: HTTP error for '%s': %s", source.name, e)
        return pd.DataFrame()

    payload_size = len(resp.content)
    df = _parse_response(resp.text, source)

    if df.empty:
        logger.warning("fetcher: no data parsed for '%s'", source.name)
        return df

    # Trim to 1 year by default, or full if payload small
    if payload_size < MAX_PAYLOAD_BYTES:
        logger.info("fetcher: '%s' payload=%d bytes < 1MB — keeping all history", source.name, payload_size)
    else:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
        before = len(df)
        df = df[df["ts"] >= cutoff]
        logger.info("fetcher: '%s' trimmed %d→%d rows (payload=%d bytes > 1MB)", source.name, before, len(df), payload_size)

    # Save raw data
    _save_raw(source.name, df)
    return df


def _save_raw(name: str, df: pd.DataFrame) -> None:
    """Save raw fetched data to data/external/{name}/{YYYY-MM}.parquet."""
    if "ts" not in df.columns:
        return
    for month, group in df.groupby(df["ts"].dt.to_period("M")):
        out_dir = EXTERNAL_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{month}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            merged = pd.concat([existing, group], ignore_index=True)
            merged.drop_duplicates(subset="ts", keep="last", inplace=True)
            merged.sort_values("ts", inplace=True)
        else:
            merged = group.sort_values("ts")
        merged.to_parquet(path, compression="snappy")
