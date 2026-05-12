"""Leaderboard — progressive save of research results.

Primary storage is **JSONL** (append-only, safe for concurrent writes).
Parquet is a secondary compacted format rebuilt on demand.

Each strategy result is written to JSONL immediately.
If the file was being written during a crash, the JSONL survives.
The Parquet can always be rebuilt: `load_leaderboard(force_rebuild=True)`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_LEADERBOARD_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet" / "research"
_JSONL_PATH = _LEADERBOARD_DIR / "leaderboard.jsonl"
_PARQUET_PATH = _LEADERBOARD_DIR / "leaderboard.parquet"

_SCHEMA_COLS = [
    "run_id", "generation",
    "sharpe", "win_rate", "profit_factor",
    "total_pnl", "max_dd", "n_trades", "passes_gates",
    "rules_json", "config_json",
    "elapsed_bt",
    "llm_model", "llm_explanation", "llm_suggestions", "llm_confidence",
]


def init_leaderboard() -> None:
    """Ensure the leaderboard directory exists."""
    _LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    # Touch JSONL if it doesn't exist
    if not _JSONL_PATH.exists():
        _JSONL_PATH.write_text("")


def append_result(row: dict, strategy_dict: dict | None = None) -> None:
    """Append a single result row to the leaderboard (JSONL, then Parquet).

    JSONL is append-safe — concurrent writes don't corrupt it.
    Parquet is rebuilt every 50 rows for performance.

    Also saves the strategy JSON to data/strategies/ if promising.
    """
    # 1) Append to JSONL (always safe)
    with open(_JSONL_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")

    # 2) Every 50 rows, rebuild Parquet from JSONL
    n_entries = sum(1 for _ in _JSONL_PATH.open())
    if n_entries % 50 == 0:
        _rebuild_parquet()
    else:
        # Quick Parquet append (best-effort, no crash if corrupted)
        try:
            df_new = pd.DataFrame([row])
            if _PARQUET_PATH.exists():
                try:
                    df_old = pd.read_parquet(_PARQUET_PATH)
                    for col in df_new.columns:
                        if col not in df_old.columns:
                            df_old[col] = None
                    for col in df_old.columns:
                        if col not in df_new.columns:
                            df_new[col] = None
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                except Exception:
                    df_all = df_new
            else:
                df_all = df_new
            df_all.to_parquet(_PARQUET_PATH, compression="snappy")
        except Exception:
            pass  # Parquet failed, but JSONL is safe

    # 3) Save strategy JSON if promising
    if strategy_dict and row.get("sharpe", -999) > 0 and row.get("n_trades", 0) >= 30:
        strat_dir = Path(__file__).resolve().parents[2] / "data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        path = strat_dir / f"research_{row['run_id']}.json"
        path.write_text(json.dumps(strategy_dict, indent=2))


def load_leaderboard(force_rebuild: bool = False) -> pd.DataFrame:
    """Load all research results.

    If the Parquet is corrupted or missing, rebuilds from JSONL.
    Set ``force_rebuild=True`` to force a fresh Parquet from JSONL.
    """
    # Rebuild from JSONL if Parquet missing, corrupted, or forced
    if not _PARQUET_PATH.exists() or force_rebuild or _is_parquet_corrupted():
        _rebuild_parquet()

    try:
        return pd.read_parquet(_PARQUET_PATH)
    except Exception:
        _rebuild_parquet()
        try:
            return pd.read_parquet(_PARQUET_PATH)
        except Exception:
            return _load_from_jsonl()


def _is_parquet_corrupted() -> bool:
    """Quick check if the Parquet file is valid."""
    try:
        import pyarrow.parquet as pq
        pq.read_schema(_PARQUET_PATH)
        return False
    except Exception:
        return True


def _rebuild_parquet() -> None:
    """Rebuild the Parquet file from the JSONL log."""
    try:
        rows = []
        with open(_JSONL_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(_PARQUET_PATH, compression="snappy")
        else:
            _PARQUET_PATH.write_text("")
    except Exception as e:
        logger.warning("leaderboard: parquet rebuild failed: %s", e)


def _load_from_jsonl() -> pd.DataFrame:
    """Load all rows from JSONL directly (fallback)."""
    rows = []
    with open(_JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=_SCHEMA_COLS)


import logging
logger = logging.getLogger(__name__)
