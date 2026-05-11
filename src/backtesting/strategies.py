"""Strategy functions for backtesting.

Each strategy is a function:
    (df: pd.DataFrame) -> pd.Series

Where the return series has values:
    1  = BULLISH (enter LONG)
    -1 = BEARISH (enter SHORT)
    0  = HOLD / no signal

All functions are pure — they only read the DataFrame, never mutate it.
"""
from __future__ import annotations

import pandas as pd


def rsi_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """BUY if RSI < 30 (oversold), SELL if RSI > 70 (overbought)."""
    sig = pd.Series(0, index=df.index)
    sig[df["rsi_14"] < 30] = 1
    sig[df["rsi_14"] > 70] = -1
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


STRATEGY_REGISTRY = {
    "rsi_mean_reversion": rsi_mean_reversion,
    "macd_crossover": macd_crossover,
    "ema_trend": ema_trend,
    "vwap_bounce": vwap_bounce,
    "heikin_ashi_streak": heikin_ashi_streak,
}
