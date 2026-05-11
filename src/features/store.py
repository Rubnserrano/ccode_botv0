"""Feature Store — Parquet with precomputed indicators.

Layout:  data/features/{symbol}/{YYYY-MM}.parquet
Schema: ts + all columns from calc_all() (dynamic, inferred from DataFrame)

The store is write-once, read-many. Indicators are precomputed during
build and updated in real-time via main.py's on_candle_close.

Usage:
    from src.features.store import write, read, available_range
    df = read("btcusdt", start=..., end=...)  # OHLCV + indicators
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_FEATURES_ROOT = Path(__file__).resolve().parents[2] / "data" / "features"


def _partition_path(symbol: str, month: str) -> Path:
    return _FEATURES_ROOT / symbol.lower() / f"{month}.parquet"


def _schema_from_df(df: pd.DataFrame) -> pa.schema:
    """Infer pyarrow schema from DataFrame, ensuring ts is first."""
    fields = [pa.field("ts", pa.timestamp("ns", tz="UTC"))]
    for c in df.columns:
        if c == "ts":
            continue
        dtype = df[c].dtype
        if pd.api.types.is_float_dtype(dtype):
            fields.append(pa.field(c, pa.float64()))
        elif pd.api.types.is_integer_dtype(dtype):
            fields.append(pa.field(c, pa.int64()))
        else:
            fields.append(pa.field(c, pa.string()))
    return pa.schema(fields)


def write(symbol: str, df: pd.DataFrame) -> int:
    """Write/merge features into monthly Parquet partitions.

    ``df`` must have a ``ts`` column and at least the OHLCV columns.
    Returns total rows written.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    written = 0
    for month, group in df.groupby(df["ts"].dt.tz_convert(None).dt.to_period("M")):
        path = _partition_path(symbol, str(month))
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pq.read_table(path).to_pandas()
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            merged = pd.concat([existing, group], ignore_index=True)
            merged.drop_duplicates(subset=["ts"], keep="last", inplace=True)
            merged.sort_values("ts", inplace=True)
        else:
            merged = group.sort_values("ts")

        schema = _schema_from_df(merged)
        table = pa.Table.from_pandas(merged, schema=schema, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        written += len(merged)

    return written


def read(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Read features within a time range.

    Returns empty DataFrame if no features exist for the symbol.
    """
    base = _FEATURES_ROOT / symbol.lower()
    if not base.exists():
        return pd.DataFrame()

    files = sorted(base.glob("*.parquet"))
    if start or end:
        def _in_range(p: Path) -> bool:
            try:
                month = pd.Period(p.stem, freq="M")
                if start and month.end_time.tz_localize("UTC") < pd.Timestamp(start, tz="UTC"):
                    return False
                if end and month.start_time.tz_localize("UTC") > pd.Timestamp(end, tz="UTC"):
                    return False
            except Exception:
                pass
            return True
        files = [p for p in files if _in_range(p)]

    if not files:
        return pd.DataFrame()

    frames = [pq.read_table(p).to_pandas() for p in files]
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


def available_range(symbol: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Return (earliest_ts, latest_ts) or (None, None)."""
    df = read(symbol)
    if df.empty:
        return None, None
    return df["ts"].iloc[0].to_pydatetime(), df["ts"].iloc[-1].to_pydatetime()


def row_count(symbol: str) -> int:
    """Return total row count without loading data."""
    base = _FEATURES_ROOT / symbol.lower()
    if not base.exists():
        return 0
    return sum(pq.ParquetFile(p).metadata.num_rows for p in base.glob("*.parquet"))
