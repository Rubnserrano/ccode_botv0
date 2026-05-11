"""Strategy dataclass — wraps a signal function with metadata.

Prepares the terrain for a future DSL by making every strategy
serializable to dict/JSON today.

Usage:
    strat = Strategy(
        name="rsi_mean_reversion",
        params={"rsi_period": 14, "oversold": 30, "overbought": 70},
        fn=rsi_mean_reversion,
    )
    signals = strat.generate(df)
    print(strat.to_dict())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Callable

import pandas as pd


SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Strategy:
    """An immutable strategy specification.

    ``fn`` is a callable ``(df: pd.DataFrame) -> pd.Series`` where
    the series values are 1 (BUY), -1 (SELL), 0 (HOLD).

    For future DSL migration, all strategy logic lives in ``params``
    while ``fn`` is the interpreter that executes those params.
    """
    name: str
    params: dict = field(default_factory=dict)
    description: str = ""
    fn: SignalFn | None = None

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Compute signals for the given OHLCV+indicators DataFrame."""
        if self.fn is None:
            return pd.Series(0, index=df.index)
        return self.fn(df)

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-compatible). Excludes ``fn``."""
        d = {"name": self.name, "params": dict(self.params), "description": self.description}
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict, fn_map: dict[str, SignalFn] | None = None) -> Strategy:
        """Deserialize from dict.

        If ``fn_map`` is provided, the ``fn`` is resolved from ``d["name"]``.
        """
        fn = None
        if fn_map and d.get("name") in fn_map:
            fn = fn_map[d["name"]]
        return cls(
            name=d["name"],
            params=d.get("params", {}),
            description=d.get("description", ""),
            fn=fn,
        )
