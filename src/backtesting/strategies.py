"""Strategy functions for backtesting.

Each strategy is a function:
    (df: pd.DataFrame) -> pd.Series

Where the return series has values:
    1  = BULLISH (enter LONG)
    -1 = BEARISH (enter SHORT)
    0  = HOLD / no signal

All functions are pure — they only read the DataFrame, never mutate it.

The module also exports ``STRATEGY_REGISTRY`` as a ``dict[str, Strategy]``
and ``FN_MAP`` for use with ``Strategy.from_dict()``.
"""
from __future__ import annotations

from functools import partial

import pandas as pd

from src.backtesting.strategy import Strategy


def rsi_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """BUY if RSI < 30 in RANGING regime, SELL if RSI > 70 in RANGING regime."""
    sig = pd.Series(0, index=df.index)
    ranging = df.get("regime", pd.Series(0, index=df.index)) == 0
    sig[(df["rsi_14"] < 30) & ranging] = 1
    sig[(df["rsi_14"] > 70) & ranging] = -1
    return sig


def macd_crossover(df: pd.DataFrame) -> pd.Series:
    """BUY if MACD crosses above signal line, SELL if crosses below."""
    sig = pd.Series(0, index=df.index)
    prev_hist = df["macd_hist"].shift(1)
    sig[(df["macd_hist"] > 0) & (prev_hist <= 0)] = 1
    sig[(df["macd_hist"] < 0) & (prev_hist >= 0)] = -1
    return sig


def ema_trend(df: pd.DataFrame) -> pd.Series:
    """BUY if close > EMA(50) and EMA(9) > EMA(21). SELL if opposite."""
    sig = pd.Series(0, index=df.index)
    bull = (df["close"] > df["ema_50"]) & (df["ema_9"] > df["ema_21"])
    bear = (df["close"] < df["ema_50"]) & (df["ema_9"] < df["ema_21"])
    sig[bull] = 1
    sig[bear] = -1
    return sig


def vwap_bounce(df: pd.DataFrame) -> pd.Series:
    """BUY if price pulls back below VWAP by >1%. SELL if >1% above VWAP."""
    sig = pd.Series(0, index=df.index)
    pct_above = (df["close"] - df["vwap"]) / df["vwap"]
    sig[pct_above < -0.01] = 1
    sig[pct_above > 0.01] = -1
    return sig


def heikin_ashi_streak(df: pd.DataFrame) -> pd.Series:
    """BUY if 3+ consecutive green HA candles, SELL if 3+ consecutive red."""
    green = df["ha_close"] > df["ha_open"]
    streak = green.astype(int).groupby((green != green.shift()).cumsum()).cumsum()
    red_streak = (~green).astype(int).groupby((green == green.shift()).cumsum()).cumsum()
    sig = pd.Series(0, index=df.index)
    sig[streak >= 3] = 1
    sig[red_streak >= 3] = -1
    return sig


FN_MAP = {
    "rsi_mean_reversion": rsi_mean_reversion,
    "macd_crossover": macd_crossover,
    "ema_trend": ema_trend,
    "vwap_bounce": vwap_bounce,
    "heikin_ashi_streak": heikin_ashi_streak,
}

STRATEGY_REGISTRY: dict[str, Strategy] = {
    "rsi_mean_reversion": Strategy(
        name="rsi_mean_reversion",
        params={"rsi_period": 14, "oversold": 30, "overbought": 70},
        description="BUY if RSI < 30 in RANGING regime, SELL if RSI > 70 in RANGING regime",
        fn=rsi_mean_reversion,
    ),
    "macd_crossover": Strategy(
        name="macd_crossover",
        params={"fast": 12, "slow": 26, "signal": 9},
        description="BUY on MACD bullish cross, SELL on bearish cross",
        fn=macd_crossover,
    ),
    "ema_trend": Strategy(
        name="ema_trend",
        params={"fast": 9, "medium": 21, "slow": 50},
        description="BUY if close > EMA50 and EMA9 > EMA21, SELL opposite",
        fn=ema_trend,
    ),
    "vwap_bounce": Strategy(
        name="vwap_bounce",
        params={"deviation_pct": 0.01},
        description="BUY if price < VWAP - 1%, SELL if price > VWAP + 1%",
        fn=vwap_bounce,
    ),
    "heikin_ashi_streak": Strategy(
        name="heikin_ashi_streak",
        params={"min_streak": 3},
        description="BUY on 3+ green HA candles, SELL on 3+ red HA candles",
        fn=heikin_ashi_streak,
    ),
}
