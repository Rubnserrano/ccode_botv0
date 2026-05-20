"""Strategy schema — validates JSON strategy definitions.

A strategy is a JSON document describing entry/exit rules.
Conditions are combined with AND logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

VALID_OPS = {"lt", "gt", "lte", "gte", "eq", "ne", "in",
             "cross_above", "cross_below",
             "gt_rolling", "lt_rolling",
             "streak_gte", "streak_lte"}

VALID_ROLLING = {"sma", "ema"}  # sma=mean, ema=exponential


@dataclass
class Condition:
    indicator: str          # "rsi", "close", "volume", "regime", etc
    op: str                 # "lt", "gt", "cross_above", "gt_rolling", etc
    value: Any = None       # umbral fijo (30, 0.5) o dict para rolling
    params: dict = field(default_factory=dict)   # {"period": 14} para el indicador
    rolling: str | None = None   # "sma", "ema" — solo para op=gt_rolling/lt_rolling
    period: int | None = None    # periodo del rolling


@dataclass
class ExitRules:
    tp_pct: float = 0.05
    sl_pct: float = 0.02
    horizon_bars: int = 48


@dataclass
class StrategyDef:
    """A strategy defined as data (not Python code)."""
    name: str
    entry_conditions: list[Condition]
    exit: ExitRules = field(default_factory=ExitRules)
    entry_operator: str = "all"  # only "all" (AND) for now
    direction: str = "long"  # "long" or "short"


def validate_condition(c: dict) -> Condition:
    """Validate and return a Condition from a raw dict."""
    if not isinstance(c, dict):
        raise ValueError(f"Condition must be a dict, got {type(c)}")
    if "indicator" not in c:
        raise ValueError("Condition missing 'indicator'")
    if "op" not in c:
        raise ValueError("Condition missing 'op'")
    op = c["op"]
    if op not in VALID_OPS:
        raise ValueError(f"Invalid op '{op}'. Valid: {VALID_OPS}")
    if op in ("gt_rolling", "lt_rolling"):
        if "rolling" not in c:
            raise ValueError(f"op '{op}' requires 'rolling' field")
        if c["rolling"] not in VALID_ROLLING:
            raise ValueError(f"Invalid rolling '{c['rolling']}'. Valid: {VALID_ROLLING}")
        if "period" not in c:
            raise ValueError(f"op '{op}' requires 'period' field")
    value = c.get("value")
    if value is not None and not isinstance(value, (int, float)):
        logger.warning("Non-numeric value '%s' for indicator '%s', coercing", value, c.get("indicator"))
    return Condition(
        indicator=c["indicator"],
        op=op,
        value=c.get("value"),
        params=c.get("params", {}),
        rolling=c.get("rolling"),
        period=c.get("period"),
    )


def validate_strategy(sd: dict) -> StrategyDef:
    """Validate a full strategy dict, return StrategyDef."""
    if "name" not in sd:
        raise ValueError("Strategy missing 'name'")
    if "entry_conditions" not in sd:
        raise ValueError("Strategy missing 'entry_conditions'")
    conditions = [validate_condition(c) for c in sd["entry_conditions"]]
    exit_rules = ExitRules(**sd.get("exit", {}))
    direction = sd.get("direction", "long")
    if direction not in ("long", "short"):
        logger.warning("Invalid direction '%s', defaulting to 'long'", direction)
        direction = "long"
    return StrategyDef(
        name=sd["name"],
        entry_conditions=conditions,
        exit=exit_rules,
        entry_operator=sd.get("entry_operator", "all"),
        direction=direction,
    )
