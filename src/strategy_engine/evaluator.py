"""Evaluator — takes a JSON strategy definition + DataFrame → signals.

Produces a pd.Series with values 1 (BUY), -1 (SELL), 0 (HOLD).
Conditions are combined with AND logic (all must be true).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategy_engine.registry import calc_indicator, calc_rolling
from src.strategy_engine.schema import StrategyDef, Condition, validate_strategy

logger = logging.getLogger(__name__)

# Cache for auto-discovered sources (avoid repeated lookups)
_auto_discovery_cache: set = set()


def evaluate(df: pd.DataFrame, strategy_def: StrategyDef) -> pd.Series:
    """Evaluate strategy rules against a DataFrame.

    Results are cached per (indicator, frozenset(params)) to avoid
    recomputing the same indicator for every condition on every bar.

    Returns
    -------
    pd.Series with values 1 (BUY), -1 (SELL), 0 (HOLD).
    """
    signals = pd.Series(0, index=df.index, dtype=int)
    n = len(df)

    # Cache: indicator_name+frozenset(params) → pd.Series
    _cache: dict = {}

    def _cached_indicator(name: str, params: dict) -> pd.Series:
        key = (name, frozenset((k, v) for k, v in sorted(params.items())))
        if key not in _cache:
            try:
                _cache[key] = calc_indicator(name, df, params)
            except (ValueError, KeyError):
                # Indicator not found — try auto-discovery from external sources
                if name not in _auto_discovery_cache:
                    _auto_discovery_cache.add(name)
                    from src.intelligence.registry import try_auto_discover
                    if try_auto_discover(name):
                        _cache[key] = calc_indicator(name, df, params)
                        return _cache[key]
                # Still not found — return zeros
                logger.warning("evaluator: indicator '%s' not found (returning zeros)", name)
                return pd.Series(0.0, index=df.index)
        return _cache[key]

    def _eval(cond: Condition, i: int) -> bool:
        series = _cached_indicator(cond.indicator, cond.params)
        if series.empty:
            return False
        value = series.iloc[i]
        if pd.isna(value):
            return False

        op = cond.op
        cmp_val = cond.value

        # Auto-resolve: if cmp_val is a string that names another column/indicator,
        # compare the two indicators instead of using a fixed value.
        # This allows natural expressions like {"indicator": "close", "op": "gt", "value": "vwap"}
        if isinstance(cmp_val, str) and cmp_val != "":
            try:
                cmp_series = _cached_indicator(cmp_val, {})
                if not cmp_series.empty:
                    other_val = cmp_series.iloc[i]
                    if not pd.isna(other_val):
                        cmp_val = float(other_val)
            except Exception:
                pass

        # Rolling, cross, and streak operators don't need cmp_val — handle them first
        if op in ("gt_rolling", "lt_rolling"):
            roll = calc_rolling(series, cond.rolling, cond.period)
            roll_val = roll.iloc[i]
            if pd.isna(roll_val):
                return False
            return value > roll_val if op == "gt_rolling" else value < roll_val

        if op in ("cross_above", "cross_below"):
            if i == 0:
                return False
            prev = series.iloc[i - 1]
            if pd.isna(prev) or pd.isna(value):
                return False
            if op == "cross_above":
                return prev <= cmp_val and value > cmp_val
            else:
                return prev >= cmp_val and value < cmp_val

        if op in ("streak_gte", "streak_lte"):
            count = 0
            direction = 1 if op == "streak_gte" else -1
            for j in range(i, max(-1, i - (cond.period or 5)), -1):
                v = series.iloc[j]
                if pd.notna(v) and (v > 0 if direction == 1 else v < 0):
                    count += 1
                else:
                    break
            return count >= (cond.value or 3)

        # For all other ops: safely coerce value to numeric
        if not isinstance(cmp_val, (int, float)):
            try:
                cmp_val = float(cmp_val)
            except (ValueError, TypeError):
                return False

        if op in ("lt", "gt", "lte", "gte", "eq", "ne", "in"):
            if op == "lt":   return value < cmp_val
            if op == "gt":   return value > cmp_val
            if op == "lte":  return value <= cmp_val
            if op == "gte":  return value >= cmp_val
            if op == "eq":   return value == cmp_val
            if op == "ne":   return value != cmp_val
            if op == "in":
                if isinstance(cmp_val, (list, tuple)):
                    return value in cmp_val
                return False

        return False

    for i in range(min(50, n), n):
        passed = all(_eval(cond, i) for cond in strategy_def.entry_conditions)
        if passed:
            signals.iloc[i] = 1

    return signals
