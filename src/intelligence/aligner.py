"""Aligner — resamples external data to OHLCV timeframes.

External data comes at arbitrary timestamps (daily VIX, irregular news).
The aligner resamples it to regular intervals (15m, 1h) using ffill/interpolate.

Output: data/external_aligned/{timeframe}/{source_name}.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.intelligence.models import DataSource

logger = logging.getLogger(__name__)

ALIGNED_DIR = Path(__file__).resolve().parents[2] / "data" / "external_aligned"

# Map user-friendly names to pandas offset aliases
_OFFSET_MAP = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def align_source(df: pd.DataFrame, source: DataSource, timeframe: str) -> pd.DataFrame:
    """Resample external data to a target timeframe.

    The resulting DataFrame has a regular time index (every `timeframe`)
    with values propagated/aligned according to source.align.method.

    Writes to: data/external_aligned/{timeframe}/{source.name}.parquet
    """
    if df.empty or "ts" not in df.columns:
        return df

    freq = _OFFSET_MAP.get(timeframe)
    if freq is None:
        logger.warning("aligner: unknown timeframe '%s'", timeframe)
        return df

    method = source.align.method
    value_columns = list(source.columns.values())
    cols_to_align = [c for c in value_columns if c in df.columns]

    if not cols_to_align:
        return df

    # Create regular time index
    full_idx = pd.date_range(
        start=df["ts"].min().floor(freq),
        end=df["ts"].max().ceil(freq),
        freq=freq,
        tz="UTC",
        name="ts",
    )
    aligned = pd.DataFrame({"ts": full_idx})
    aligned = aligned.set_index("ts")

    # Set source data to same index for merge
    src = df[["ts"] + cols_to_align].set_index("ts").sort_index()

    # Remove duplicates by averaging
    if src.index.duplicated().any():
        src = src.groupby(src.index).mean()

    # Merge and align
    for col in cols_to_align:
        if method == "ffill":
            aligned[col] = src[col].reindex(aligned.index, method="ffill")
            if source.align.decay_periods > 0:
                # Linear decay: value decreases over N periods
                last_val = src[col].reindex(aligned.index, method="ffill")
                decay = pd.Series(1.0, index=aligned.index)
                for i in range(1, source.align.decay_periods + 1):
                    mask = aligned.index.isin(src.index).shift(i, fill_value=False)
                    decay[mask] = 1.0 - (i / source.align.decay_periods)
                aligned[col] = last_val * decay.clip(0, 1)

        elif method == "interpolate":
            aligned[col] = src[col].reindex(aligned.index).interpolate(method="time")

        elif method == "sum":
            aligned[col] = src[col].reindex(aligned.index, method=None).fillna(0).rolling(2).sum()

        elif method == "avg":
            aligned[col] = src[col].reindex(aligned.index, method=None).rolling(2).mean()

    aligned = aligned.reset_index()

    # Save
    out_dir = ALIGNED_DIR / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.name}.parquet"
    aligned.to_parquet(out_path, compression="snappy")
    logger.info("aligner: wrote %d rows to %s", len(aligned), out_path)

    return aligned
