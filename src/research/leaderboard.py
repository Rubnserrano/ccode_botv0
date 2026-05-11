"""Leaderboard — progressive save of research results.

Each strategy result is appended to a Parquet file immediately,
so partial data survives timeout/crash.

The schema includes LLM analysis columns for future model training.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_LEADERBOARD_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet" / "research"
_LEADERBOARD_PATH = _LEADERBOARD_DIR / "leaderboard.parquet"

_SCHEMA_COLS = [
    # Strategy id
    "run_id", "generation",
    # Backtest metrics
    "sharpe", "win_rate", "profit_factor",
    "total_pnl", "max_dd", "n_trades", "passes_gates",
    # Strategy definition + config
    "rules_json", "config_json",
    # Timing
    "elapsed_bt",
    # LLM analysis (for future training)
    "llm_model", "llm_explanation", "llm_suggestions", "llm_confidence",
]


def init_leaderboard() -> None:
    """Ensure the leaderboard directory exists."""
    _LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    if not _LEADERBOARD_PATH.exists():
        df = pd.DataFrame(columns=_SCHEMA_COLS)
        df.to_parquet(_LEADERBOARD_PATH, compression="snappy")


def append_result(row: dict, strategy_dict: dict | None = None) -> None:
    """Append a single result row to the leaderboard Parquet.

    Also saves the strategy JSON to data/strategies/ if promising.
    """
    df_new = pd.DataFrame([row])
    if _LEADERBOARD_PATH.exists():
        df_old = pd.read_parquet(_LEADERBOARD_PATH)
        # Reconcile columns — new rows may have cols the schema doesn't yet
        for col in df_new.columns:
            if col not in df_old.columns:
                df_old[col] = None
        for col in df_old.columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(_LEADERBOARD_PATH, compression="snappy")

    # Save strategy JSON if promising
    if strategy_dict and row.get("sharpe", -999) > 0 and row.get("n_trades", 0) >= 30:
        strat_dir = Path(__file__).resolve().parents[2] / "data" / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        path = strat_dir / f"research_{row['run_id']}.json"
        path.write_text(json.dumps(strategy_dict, indent=2))


def load_leaderboard() -> pd.DataFrame:
    """Load all research results."""
    if _LEADERBOARD_PATH.exists():
        return pd.read_parquet(_LEADERBOARD_PATH)
    return pd.DataFrame(columns=_SCHEMA_COLS)
