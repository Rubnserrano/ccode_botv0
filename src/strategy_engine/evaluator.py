"""Evaluator — takes a JSON strategy definition + DataFrame → signals.

Produces a pd.Series with values 1 (BUY), -1 (SELL), 0 (HOLD).
Conditions are combined with AND logic (all must be true).

This version uses VECTORIZED evaluation (numpy/pandas ops on entire Series)
instead of a bar-by-bar Python loop. This makes it ~50-100x faster.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategy_engine.registry import calc_indicator, calc_rolling
from src.strategy_engine.schema import StrategyDef, Condition, validate_strategy

logger = logging.getLogger(__name__)

_auto_discovery_cache: set = set()


def evaluate(df: pd.DataFrame, strategy_def: StrategyDef) -> pd.Series:
    """Evaluate strategy rules against a DataFrame using vectorized operations.

    Returns
    -------
    pd.Series with values 1 (BUY), -1 (SELL), 0 (HOLD).
    """
    n = len(df)
    if n == 0:
        return pd.Series(0, index=df.index, dtype=int)

    _cache: dict = {}

    def _cached_indicator(name: str, params: dict) -> pd.Series:
        key = (name, frozenset((k, v) for k, v in sorted(params.items())))
        if key not in _cache:
            try:
                _cache[key] = calc_indicator(name, df, params)
            except (ValueError, KeyError):
                if name not in _auto_discovery_cache:
                    _auto_discovery_cache.add(name)
                    from src.intelligence.registry import try_auto_discover
                    if try_auto_discover(name):
                        _cache[key] = calc_indicator(name, df, params)
                        return _cache[key]
                logger.warning("evaluator: indicator '%s' not found (returning zeros)", name)
                return pd.Series(0.0, index=df.index)
        return _cache[key]

    # Evaluate each condition as a boolean Series, then AND them all
    warmup = 50
    condition_masks = []

    for cond in strategy_def.entry_conditions:
        mask = _eval_condition_vectorized(cond, df, _cached_indicator, warmup, n)
        condition_masks.append(mask)

    if not condition_masks:
        return pd.Series(0, index=df.index, dtype=int)

    # AND all conditions
    combined = condition_masks[0]
    for mask in condition_masks[1:]:
        combined = combined & mask

    signals = pd.Series(0, index=df.index, dtype=int)
    direction_val = -1 if strategy_def.direction == "short" else 1
    signals.iloc[warmup:] = combined.iloc[warmup:].astype(int) * direction_val
    return signals


def _eval_condition_vectorized(
    cond: Condition,
    df: pd.DataFrame,
    cached_indicator,
    warmup: int,
    n: int,
) -> pd.Series:
    """Evaluate a single condition as a vectorized boolean Series.

    Returns a Series of bool values, one per row.
    """
    series = cached_indicator(cond.indicator, cond.params)
    if series.empty or len(series) != n:
        return pd.Series(False, index=df.index)

    op = cond.op
    cmp_val = cond.value

    # Auto-resolve string values to indicator comparison
    if isinstance(cmp_val, str) and cmp_val != "":
        try:
            cmp_series = cached_indicator(cmp_val, {})
            if not cmp_series.empty and len(cmp_series) == n:
                cmp_val = cmp_series
        except Exception:
            pass

    # ── Rolling comparisons ──────────────────────────────────────────
    if op in ("gt_rolling", "lt_rolling"):
        roll = calc_rolling(series, cond.rolling, cond.period)
        if op == "gt_rolling":
            result = series > roll
        else:
            result = series < roll
        return result.fillna(False)

    # ── Crossover ────────────────────────────────────────────────────
    if op in ("cross_above", "cross_below"):
        prev = series.shift(1)
        if isinstance(cmp_val, pd.Series):
            prev_cmp = cmp_val.shift(1)
            curr_cmp = cmp_val
        else:
            prev_cmp = None
            curr_cmp = cmp_val

        if op == "cross_above":
            if prev_cmp is not None:
                result = (prev <= prev_cmp) & (series > curr_cmp)
            else:
                result = (prev <= cmp_val) & (series > cmp_val)
        else:
            if prev_cmp is not None:
                result = (prev >= prev_cmp) & (series < curr_cmp)
            else:
                result = (prev >= cmp_val) & (series < cmp_val)
        return result.fillna(False)

    # ── Streak ───────────────────────────────────────────────────────
    if op in ("streak_gte", "streak_lte"):
        direction = 1 if op == "streak_gte" else -1
        threshold = cond.value or 3
        if direction == 1:
            positive = (series > 0).astype(int)
        else:
            positive = (series < 0).astype(int)

        # Rolling count of consecutive positive/negative values
        groups = (positive != positive.shift(1)).cumsum()
        streak = positive.groupby(groups).cumsum()
        result = streak >= threshold
        return result.fillna(False)

    # ── Simple comparisons (vectorized) ─────────────────────────────
    if isinstance(cmp_val, pd.Series):
        right = cmp_val
    else:
        try:
            right = float(cmp_val)
        except (ValueError, TypeError):
            return pd.Series(False, index=df.index)

    if op == "lt":
        result = series < right
    elif op == "gt":
        result = series > right
    elif op == "lte":
        result = series <= right
    elif op == "gte":
        result = series >= right
    elif op == "eq":
        result = series == right
    elif op == "ne":
        result = series != right
    elif op == "in":
        if isinstance(cmp_val, (list, tuple)):
            result = series.isin(cmp_val)
        else:
            result = pd.Series(False, index=df.index)
    else:
        result = pd.Series(False, index=df.index)

    return result.fillna(False)