"""Universal TimeSeries Store — write/read any time-series data.

Asset ID format: ``source_type:source:name``
  - ``market:binance:btcusdt``     → OHLCV from Binance
  - ``sentiment:alternative:fear_greed`` → Fear & Greed Index
  - ``macro:fred:vix``             → VIX from FRED
  - ``derivatives:binance:funding`` → Binance funding rates

Storage layout:
  data/ts/{source_type}/{source}_{name}/{frequency}/{YYYY-MM}.parquet

Frequencies:
  - ``raw``       → native frequency (what the source provides)
  - ``15m``, ``1h``, ``1d`` → aligned to standard intervals
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_TS_ROOT = Path(__file__).resolve().parents[1] / "data" / "ts"


def _parse_asset_id(asset_id: str) -> tuple[str, str, str]:
    """Parse asset_id into (source_type, source, name).

    ``market:binance:btcusdt`` → (``market``, ``binance``, ``btcusdt``)
    ``sentiment:fear_greed``   → (``sentiment``, ``fear_greed``, ``fear_greed``)
    ``btcusdt``                → (``other``, ``btcusdt``, ``btcusdt``)
    """
    parts = asset_id.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], ":".join(parts[2:])
    elif len(parts) == 2:
        return parts[0], parts[1], parts[1]
    else:
        return "other", asset_id, asset_id


def _asset_dir(asset_id: str) -> Path:
    """Return the base directory for an asset (no frequency).

    Uses the raw asset_id (with colons) as the directory name on disk.
    On Linux/macOS colons in filenames work fine.
    """
    st, _src, _name = _parse_asset_id(asset_id)
    return _TS_ROOT / st / asset_id


def _freq_dir(asset_id: str, frequency: str = "raw") -> Path:
    """Return the directory for an asset at a given frequency."""
    return _asset_dir(asset_id) / frequency


def write(asset_id: str, df: pd.DataFrame, frequency: str = "raw") -> int:
    """Write time-series data to monthly Parquet partitions.

    Args:
        asset_id: Unique asset identifier (e.g. ``market:binance:btcusdt``).
        df: DataFrame with at least a ``ts`` column (datetime UTC).
        frequency: ``raw``, ``15m``, ``1h``, ``1d``, etc.

    Returns:
        Number of rows written.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    written = 0
    for month, group in df.groupby(df["ts"].dt.tz_convert(None).dt.to_period("M")):
        out_dir = _freq_dir(asset_id, frequency)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{month}.parquet"

        group = group.sort_values("ts")

        if path.exists():
            existing = _read_parquet_safe(path)
            if not existing.empty:
                merged = pd.concat([existing, group], ignore_index=True)
                merged.drop_duplicates(subset=["ts"], keep="last", inplace=True)
                merged.sort_values("ts", inplace=True)
            else:
                merged = group
        else:
            merged = group

        _write_parquet_safe(path, merged)
        written += len(merged)

    return written


def read(
    asset_id: str,
    frequency: str = "raw",
    columns: Optional[list[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read time-series data for an asset.

    Args:
        asset_id: Unique asset identifier.
        frequency: Data frequency.
        columns: Subset of columns (None = all).
        start: Inclusive start timestamp.
        end: Inclusive end timestamp.
        limit: Max rows (applied after filters, from end).

    Returns:
        DataFrame sorted by ``ts``, or empty if no data.
    """
    base = _freq_dir(asset_id, frequency)
    if not base.exists():
        return pd.DataFrame()

    files = sorted(base.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    if start or end:
        def _in_range(p: Path) -> bool:
            try:
                month = pd.Period(p.stem, freq="M")
                if start and month.end_time.tz_localize("UTC") < _ts(start):
                    return False
                if end and month.start_time.tz_localize("UTC") > _ts(end):
                    return False
            except Exception:
                pass
            return True
        files = [p for p in files if _in_range(p)]

    if not files:
        return pd.DataFrame()

    frames = []
    for p in files:
        try:
            tbl = pq.read_table(p)
            if columns:
                keep = [c for c in columns if c in tbl.schema.names]
                if "ts" not in keep:
                    keep = ["ts"] + keep
                tbl = tbl.select(keep)
            frames.append(tbl.to_pandas())
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)

    if start is not None:
        df = df[df["ts"] >= _ts(start)]
    if end is not None:
        df = df[df["ts"] <= _ts(end)]
    if limit is not None and limit < len(df):
        df = df.tail(limit)

    return df.reset_index(drop=True)


def delete(asset_id: str) -> None:
    """Delete all data for an asset."""
    base = _asset_dir(asset_id)
    if base.exists():
        import shutil
        shutil.rmtree(base)


def catalog() -> pd.DataFrame:
    """Return a DataFrame with metadata for all stored assets.

    Columns: asset_id, source_type, frequency, min_ts, max_ts, rows, columns.
    """
    if not _TS_ROOT.exists():
        return pd.DataFrame(columns=["asset_id", "source_type", "frequency",
                                      "min_ts", "max_ts", "rows", "columns"])
    rows = []
    for st_dir in sorted(_TS_ROOT.iterdir()):
        if not st_dir.is_dir():
            continue
        source_type = st_dir.name
        for sub_dir in sorted(st_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            # The directory name IS the asset_id (e.g. "market:binance:btcusdt")
            asset_id = sub_dir.name

            for freq_dir in sorted(sub_dir.iterdir()):
                if not freq_dir.is_dir():
                    continue
                freq = freq_dir.name
                parquets = list(freq_dir.glob("*.parquet"))
                if not parquets:
                    continue

                try:
                    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
                except Exception:
                    total_rows = 0

                try:
                    first = _read_sample(parquets[0])
                    last = _read_sample(parquets[-1], last=True)
                    min_ts = str(first.get("ts", "")) if first else ""
                    max_ts = str(last.get("ts", "")) if last else ""
                    cols = list(first.get("columns", [])) if first else []
                except Exception:
                    min_ts = max_ts = ""
                    cols = []

                rows.append({
                    "asset_id": asset_id,
                    "source_type": source_type,
                    "frequency": freq,
                    "min_ts": min_ts,
                    "max_ts": max_ts,
                    "rows": total_rows,
                    "columns": cols,
                })

    return pd.DataFrame(rows)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _ts(dt: datetime) -> pd.Timestamp:
    t = pd.Timestamp(dt)
    return t.tz_convert("UTC") if t.tz else t.tz_localize("UTC")


def _read_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        tbl = pq.read_table(path)
        df = tbl.to_pandas()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def _write_parquet_safe(path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame as Parquet, inferring schema."""
    fields = [pa.field("ts", pa.timestamp("ns", tz="UTC"))]
    for c in df.columns:
        if c == "ts":
            continue
        dtype = df[c].dtype
        if pd.api.types.is_float_dtype(dtype):
            fields.append(pa.field(c, pa.float64()))
        elif pd.api.types.is_integer_dtype(dtype):
            fields.append(pa.field(c, pa.int64()))
        elif pd.api.types.is_bool_dtype(dtype):
            fields.append(pa.field(c, pa.bool_()))
        else:
            fields.append(pa.field(c, pa.string()))
    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _read_sample(path: Path, last: bool = False) -> dict:
    """Read first or last row from a Parquet file for metadata."""
    try:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        if n == 0:
            return {}
        # Read a small batch
        batch_size = min(1000, n)
        offset = n - batch_size if last else 0
        tbl = pf.read_row_groups([0])
        if tbl.num_rows == 0:
            return {}
        return {
            "ts": str(tbl.column("ts")[-1 if last else 0].as_py()),
            "columns": tbl.schema.names,
        }
    except Exception:
        return {}
