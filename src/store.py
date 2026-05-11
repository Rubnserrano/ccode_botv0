"""Parquet-backed time-series storage, partitioned by month.

Layout:  data/raw/{exchange}/{symbol}/{YYYY-MM}.parquet
Schema: ts (datetime64[ns, UTC]), open, high, low, close, volume (float64)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

_SCHEMA = pa.schema([
    pa.field("ts",     pa.timestamp("ns", tz="UTC")),
    pa.field("open",   pa.float64()),
    pa.field("high",   pa.float64()),
    pa.field("low",    pa.float64()),
    pa.field("close",  pa.float64()),
    pa.field("volume", pa.float64()),
])


def _partition_path(exchange: str, symbol: str, month: str) -> Path:
    return _ROOT / exchange.lower() / symbol.lower() / f"{month}.parquet"


def write(exchange: str, symbol: str, df: pd.DataFrame) -> int:
    """Write/merge OHLCV rows into monthly Parquet partitions.

    ``df`` must have columns: ts (datetime-like), open, high, low, close, volume.
    Returns total rows written across all partitions.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype({
        "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "volume": "float64",
    })
    written = 0
    for month, group in df.groupby(df["ts"].dt.tz_convert(None).dt.to_period("M")):
        path = _partition_path(exchange, symbol, str(month))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            merged = pd.concat([existing, group], ignore_index=True)
            merged.drop_duplicates(subset=["ts"], keep="last", inplace=True)
            merged.sort_values("ts", inplace=True)
        else:
            merged = group.sort_values("ts")
        table = pa.Table.from_pandas(merged, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        written += len(merged)
    return written


def read(
    exchange: str,
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Read OHLCV rows within a time range.

    start/end are inclusive, UTC-aware or naive. Returns DataFrame sorted by ts,
    or empty DataFrame if no data.
    """
    base = _ROOT / exchange.lower() / symbol.lower()
    if not base.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    parquet_files = sorted(base.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    if start or end:
        def _month_in_range(p: Path) -> bool:
            try:
                month = pd.Period(p.stem, freq="M")
                if start and month.end_time.tz_localize("UTC") < pd.Timestamp(start, tz="UTC"):
                    return False
                if end and month.start_time.tz_localize("UTC") > pd.Timestamp(end, tz="UTC"):
                    return False
            except Exception:
                pass
            return True
        parquet_files = [p for p in parquet_files if _month_in_range(p)]
    if not parquet_files:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    frames = [pq.read_table(p).to_pandas() for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    if start is not None:
        s = pd.Timestamp(start)
        if s.tz is None:
            s = s.tz_localize("UTC")
        df = df[df["ts"] >= s]
    if end is not None:
        e = pd.Timestamp(end)
        if e.tz is None:
            e = e.tz_localize("UTC")
        df = df[df["ts"] <= e]
    return df.reset_index(drop=True)


def available_range(exchange: str, symbol: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Return (earliest_ts, latest_ts) or (None, None) if no data."""
    df = read(exchange, symbol)
    if df.empty:
        return None, None
    return df["ts"].iloc[0].to_pydatetime(), df["ts"].iloc[-1].to_pydatetime()


def row_count(exchange: str, symbol: str) -> int:
    """Return total row count across all partitions without loading data."""
    base = _ROOT / exchange.lower() / symbol.lower()
    if not base.exists():
        return 0
    return sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in base.glob("*.parquet")
    )
