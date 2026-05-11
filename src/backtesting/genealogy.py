"""Genealogy — track strategy lineage across generations.

Each evolved strategy records its parent(s) and fitness.
Data is persisted as Parquet for later analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class GenealogyNode:
    strategy_id: str
    base_name: str
    params: dict
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    fitness: float = -999.0
    sharpe: float = 0.0
    n_trades: int = 0
    creator: str = "evolution"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_PARQUET_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet"
_NODES_PATH = _PARQUET_DIR / "genealogy_nodes.parquet"


def add_node(node: GenealogyNode) -> None:
    """Append a genealogy node to the Parquet store."""
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy_id": node.strategy_id,
        "base_name": node.base_name,
        "params": str(node.params),
        "parent_ids": ",".join(node.parent_ids),
        "generation": node.generation,
        "fitness": node.fitness,
        "sharpe": node.sharpe,
        "n_trades": node.n_trades,
        "creator": node.creator,
        "created_at": node.created_at,
    }
    df_new = pd.DataFrame([row])

    if _NODES_PATH.exists():
        df_old = pd.read_parquet(_NODES_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(_NODES_PATH, compression="snappy")


def get_lineage(strategy_id: str) -> list[dict]:
    """Return the lineage chain for a strategy (parent → child → ...)."""
    if not _NODES_PATH.exists():
        return []
    df = pd.read_parquet(_NODES_PATH)
    chain = []
    current_id = strategy_id
    for _ in range(100):
        row = df[df["strategy_id"] == current_id]
        if row.empty:
            break
        r = row.iloc[0]
        chain.append(dict(r))
        parents = r["parent_ids"]
        if parents:
            current_id = parents.split(",")[0]
        else:
            break
    return chain


def get_all_nodes() -> list[dict]:
    """Return all genealogy nodes."""
    if not _NODES_PATH.exists():
        return []
    return pd.read_parquet(_NODES_PATH).to_dict(orient="records")
