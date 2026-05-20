"""Pure OHLCV indicator calculator.

All functions are stateless — they take pandas Series/DataFrame
and return computed values. No I/O, no side effects.

Usage:
    from src.indicators.calculator import calc_all
    df = calc_all(df)  # adds rsi_14, macd, ema_*, vwap, obi, ha_*
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── RSI ─────────────────────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI. Returns 0-100, NaN until period+1 bars."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ─── MACD ────────────────────────────────────────────────────────────────────

def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


# ─── EMA ─────────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, min_periods=period).mean()


# ─── VWAP ────────────────────────────────────────────────────────────────────

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (rolling from start of DataFrame).

    Uses high+low+close as typical price: (h+l+c)/3.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


# ─── OBI (On-Balance Volume) ─────────────────────────────────────────────────

def calc_obi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Accumulates volume based on close direction."""
    direction = np.sign(close.diff())
    obi = (direction * volume).fillna(0).cumsum()
    return obi


# ─── Heikin-Ashi ─────────────────────────────────────────────────────────────

def calc_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles: ha_open, ha_high, ha_low, ha_close."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = df["open"].copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = df[["high", "low", "close"]].max(axis=1).combine(
        ha_open, np.maximum
    )
    ha_low = df[["high", "low", "close"]].min(axis=1).combine(
        ha_open, np.minimum
    )
    return pd.DataFrame({
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low": ha_low,
        "ha_close": ha_close,
    })


# ─── ATR (Average True Range) ────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility indicator.
    
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = SMA of True Range over period.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ─── ADX (Average Directional Index) ───────────────────────────────────────────

def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0-100).

    ADX < 20: weak trend / ranging
    ADX > 25: strong trend
    """
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Market regime detector — no training required.

    0 = RANGING       lateral, sin tendencia fuerte
    1 = TRENDING_UP   tendencia alcista
    2 = TRENDING_DOWN tendencia bajista
    3 = VOLATILE      volatilidad anómala
    """
    adx = calc_adx(df, 14)
    atr_mean = df["atr_14"].rolling(100).mean()
    vol_spike = df["atr_14"] > atr_mean * 1.5
    above_ema = (df["close"] > df["ema_50"]).rolling(50).mean()

    regime = pd.Series(0, index=df.index, dtype=int)
    regime[vol_spike] = 3
    trending = adx >= 20
    regime[trending & (above_ema > 0.6)] = 1
    regime[trending & (above_ema < 0.4)] = 2
    return regime


# ─── Calc All ────────────────────────────────────────────────────────────────

def calc_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to the OHLCV DataFrame.

    Returns a copy with indicator columns appended.
    Does not mutate the input.
    """
    result = df.copy()
    close = result["close"]

    result["rsi_14"] = calc_rsi(close, 14)

    macd_df = calc_macd(close, 12, 26, 9)
    for col in macd_df.columns:
        result[col] = macd_df[col]

    for period in (9, 21, 50):
        result[f"ema_{period}"] = calc_ema(close, period)

    result["vwap"] = calc_vwap(result)
    result["obi"] = calc_obi(close, result["volume"])

    ha = calc_heikin_ashi(result)
    for col in ha.columns:
        result[col] = ha[col]

    result["atr_14"] = calc_atr(result, 14)
    result["adx_14"] = calc_adx(result, 14)
    result["regime"] = detect_regime(result)

    # Derivatives indicators (computed from external columns if present)
    result = calc_derivatives_indicators(result)

    return result


def calc_derivatives_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add derivative indicators when derivatives data columns exist.

    Columns added:
    - funding_rate_8h: raw funding rate (already ffill'd)
    - funding_rate_ma: rolling mean of funding rate
    - funding_rate_zscore: z-score of funding rate
    - oi_growth: % change in open interest (24 bars)
    - oi_growth_ma: rolling mean of OI growth
    - ls_ratio_ma: rolling mean of long/short ratio
    - crowded_longs: binary flag (long_short_ratio > threshold)
    - crowded_shorts: binary flag (long_short_ratio < threshold)
    - abs_funding_rate: absolute value of funding rate
    - taker_ratio_ma: rolling mean of taker ratio
    """
    result = df.copy()

    if "funding_rate" in result.columns:
        fr = result["funding_rate"].ffill().bfill()
        result["funding_rate"] = fr
        result["funding_rate_ma"] = fr.rolling(21, min_periods=3).mean()
        fr_std = fr.rolling(21, min_periods=3).std()
        result["funding_rate_zscore"] = (fr - result["funding_rate_ma"]) / fr_std.replace(0, np.nan)
        result["abs_funding_rate"] = fr.abs()

    if "open_interest" in result.columns:
        oi = result["open_interest"].ffill().bfill()
        result["open_interest"] = oi
        result["oi_growth"] = oi.pct_change(24) * 100
        result["oi_growth_ma"] = result["oi_growth"].rolling(12, min_periods=3).mean()

    if "long_short_ratio" in result.columns:
        ls = result["long_short_ratio"].ffill().bfill()
        result["long_short_ratio"] = ls
        result["ls_ratio_ma"] = ls.rolling(21, min_periods=3).mean()
        result["crowded_longs"] = (ls > 1.3).astype(float)
        result["crowded_shorts"] = (ls < 0.7).astype(float)

    if "taker_ratio" in result.columns:
        tr = result["taker_ratio"].ffill().bfill()
        result["taker_ratio"] = tr
        result["taker_ratio_ma"] = tr.rolling(12, min_periods=3).mean()

    return result
