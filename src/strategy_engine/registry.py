"""Indicator registry — maps names to calculator functions.

On-the-fly calculation with optional caching.

EXTENDING FOR AGENTS:
  To register a new indicator at runtime (no code change):
    from src.strategy_engine.generic_calculator import register_dynamic_indicator
    register_dynamic_indicator("momentum_12", "close - close.shift(12)")

  To register via API:
    POST /api/indicators  {"name": "momentum_12", "formula": "close - close.shift(12)"}

  After registration, any strategy JSON can use the indicator:
    {"indicator": "momentum_12", "op": "gt", "value": 0}
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.indicators.calculator import (
    calc_rsi, calc_macd, calc_ema, calc_vwap,
    calc_obi, calc_atr, calc_adx, detect_regime,
)

# Rolling helpers
def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period).mean()


INDICATOR_REGISTRY: dict[str, callable] = {
    # Raw columns (passthrough)
    "close":  lambda df, p: df["close"],
    "high":   lambda df, p: df["high"],
    "low":    lambda df, p: df["low"],
    "open":   lambda df, p: df["open"],
    "volume": lambda df, p: df["volume"],

    # Computed indicators
    "rsi":      lambda df, p: calc_rsi(df["close"], p.get("period", 14)),
    "ema":      lambda df, p: calc_ema(df["close"], p.get("period", 20)),
    "macd":     lambda df, p: calc_macd(df["close"], p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))["macd"],
    "macd_signal": lambda df, p: calc_macd(df["close"], p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))["macd_signal"],
    "macd_hist": lambda df, p: calc_macd(df["close"], p.get("fast", 12), p.get("slow", 26), p.get("signal", 9))["macd_hist"],
    "vwap":     lambda df, p: calc_vwap(df),
    "obi":      lambda df, p: calc_obi(df["close"], df["volume"]),
    "atr":      lambda df, p: calc_atr(df, p.get("period", 14)),
    "adx":      lambda df, p: calc_adx(df, p.get("period", 14)),
    "regime":   lambda df, p: detect_regime(df),

    # Heikin-Ashi individual columns
    "ha_close": lambda df, p: (df["open"] + df["high"] + df["low"] + df["close"]) / 4,
    "ha_open":  lambda df, p: _ha_open(df),
}

ROLLING_REGISTRY: dict[str, callable] = {
    "sma": _sma,
    "ema": _ema,
}


def _ha_open(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi open (sequential, can't vectorize easily)."""
    ha = df["open"].copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    for i in range(1, len(df)):
        ha.iloc[i] = (ha.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    return ha


def calc_indicator(name: str, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Compute a single indicator on-the-fly.

    Uses function caching to avoid recomputing the same indicator
    with the same params within a single evaluation session.
    """
    p = params or {}
    fn = INDICATOR_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown indicator: '{name}'. Available: {list(INDICATOR_REGISTRY)}")
    return fn(df, p)


def calc_rolling(series: pd.Series, method: str, period: int) -> pd.Series:
    """Compute a rolling statistic."""
    fn = ROLLING_REGISTRY.get(method)
    if fn is None:
        raise ValueError(f"Unknown rolling method: '{method}'. Available: {list(ROLLING_REGISTRY)}")
    return fn(series, period)
