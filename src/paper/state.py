"""State persistence for paper trading — writes state.json + equity.parquet.

Decoupled: no imports from src/paper or src/backtesting.
Just reads/writes files for the dashboard to consume.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _state_path() -> Path:
    return _DATA_DIR / "paper_state.json"


def _equity_path() -> Path:
    return _DATA_DIR / "parquet" / "paper_equity.parquet"


def write_state(account_dict: dict) -> None:
    state = dict(account_dict)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _state_path().write_text(json.dumps(state, indent=2))


def append_equity(account_dict: dict) -> None:
    row = {
        "ts": datetime.now(timezone.utc),
        "capital": account_dict.get("capital", 0),
        "equity": account_dict.get("equity", 0),
        "total_pnl": account_dict.get("total_pnl", 0),
        "n_trades": account_dict.get("n_trades", 0),
    }
    df_new = pd.DataFrame([row])
    path = _equity_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_old = pd.read_parquet(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(path, compression="snappy")
