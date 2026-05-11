"""Strategy store — persists strategy definitions as JSON.

Layout: data/strategies/{name}.json

Decoupled from the engine — strategies are just data files.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.strategy_engine.schema import StrategyDef, validate_strategy

_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "data" / "strategies"


def save(strategy_def: StrategyDef) -> Path:
    """Save a strategy definition to a JSON file."""
    _STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    path = _STRATEGIES_DIR / f"{strategy_def.name}.json"
    data = {
        "name": strategy_def.name,
        "entry_operator": strategy_def.entry_operator,
        "entry_conditions": [
            {
                "indicator": c.indicator,
                "op": c.op,
                "value": c.value,
                "params": c.params,
                "rolling": c.rolling,
                "period": c.period,
            }
            for c in strategy_def.entry_conditions
        ],
        "exit": {
            "tp_pct": strategy_def.exit.tp_pct,
            "sl_pct": strategy_def.exit.sl_pct,
            "horizon_bars": strategy_def.exit.horizon_bars,
        },
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load(name: str) -> StrategyDef:
    """Load a strategy definition from a JSON file."""
    path = _STRATEGIES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Strategy '{name}' not found at {path}")
    data = json.loads(path.read_text())
    return validate_strategy(data)


def list_strategies() -> list[str]:
    """List all saved strategy names."""
    if not _STRATEGIES_DIR.exists():
        return []
    return sorted(p.stem for p in _STRATEGIES_DIR.glob("*.json"))


def delete(name: str) -> None:
    """Delete a saved strategy."""
    path = _STRATEGIES_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
