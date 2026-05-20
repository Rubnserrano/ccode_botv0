"""Hypothesis families — semantic categories for strategy exploration.

Each family defines a template of market behavior to exploit.
The LLM chooses a family + params; the search engine generates concrete implementations.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HypothesisFamily(Enum):
    MOMENTUM_CONTINUATION = "momentum_continuation"
    VOLATILITY_EXPANSION = "volatility_expansion"
    MEAN_REVERSION = "mean_reversion"
    REGIME_TRANSITION = "regime_transition"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    BREAKOUT_STRUCTURE = "breakout_structure"
    VOLATILITY_COMPRESSION_BREAKOUT = "volatility_compression_breakout"
    FUNDING_REVERSAL = "funding_reversal"
    CROWDED_POSITIONING = "crowded_positioning"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    SHORT_TREND_EXHAUSTION = "short_trend_exhaustion"
    SHORT_OVERBOUGHT_REVERSAL = "short_overbought_reversal"
    SHORT_BREAKDOWN = "short_breakdown"
    SHORT_SENTIMENT_EXTREME = "short_sentiment_extreme"
    SHORT_VOLATILITY_CRUSH = "short_volatility_crush"


FAMILY_DESCRIPTIONS = {
    HypothesisFamily.MOMENTUM_CONTINUATION: (
        "Capturar continuación de tendencia tras pausas o retrocesos. "
        "Funciona en regímenes trending (ADX>25). Usa cruces de EMA, "
        "pullbacks a medias móviles, o rupturas de rango con volumen."
    ),
    HypothesisFamily.VOLATILITY_EXPANSION: (
        "Detectar compresión de volatilidad seguida de expansión direccional. "
        "Funciona cuando ATR está comprimido vs su media histórica. "
        "Usa Bandas de Bollinger estrechas, ATR bajo, o rangos estrechos."
    ),
    HypothesisFamily.MEAN_REVERSION: (
        "Apostar a que el precio volverá a su media tras movimientos extremos. "
        "Funciona en regímenes ranging (ADX<25). Usa RSI extremo, "
        "desviaciones de VWAP, o Bandas de Bollinger."
    ),
    HypothesisFamily.REGIME_TRANSITION: (
        "Detectar transiciones entre regímenes de mercado. "
        "Por ejemplo: ranging → trending, o trending_up → trending_down. "
        "Usa cambios de ADX, cruces de medias de régimen, o divergencias."
    ),
    HypothesisFamily.SENTIMENT_DIVERGENCE: (
        "Explotar divergencias entre precio y sentimiento externo. "
        "Usa fear_greed index, volumen social, o funding rates "
        "cuando divergen de la acción del precio."
    ),
    HypothesisFamily.BREAKOUT_STRUCTURE: (
        "Detectar rupturas de niveles clave (soporte/resistencia) con confirmación. "
        "Crucial en crypto por acumulación/distribución. "
        "Usa ruptura de rango diario/semanal, o de promedios móviles clave."
    ),
    HypothesisFamily.VOLATILITY_COMPRESSION_BREAKOUT: (
        "Comprimir (rango estrecho + ATR bajo) y estallar con dirección. "
        "Ideal para capturar el inicio de movimientos grandes. "
        "Usa periodos de consolidación seguidos de expansión de rango."
    ),
    HypothesisFamily.FUNDING_REVERSAL: (
        "Explotar reversiones tras tasas de funding extremas. "
        "Funding positivo alto → overrecrowded longs → probable corrección. "
        "Funding negativo alto → overrecrowded shorts → probable rebote. "
        "Usa funding_rate con umbral y combinación con precio/RSI."
    ),
    HypothesisFamily.CROWDED_POSITIONING: (
        "Detectar posicionamiento overrecrowded usando OI + long/short ratio. "
        "OI creciendo + ratio long/short > 1.5 → mercado overrecrowded alcista. "
        "OI creciendo + ratio < 0.6 → mercado overrecrowded bajista. "
        "El unrecrowding produce movimientos acelerados."
    ),
    HypothesisFamily.LIQUIDATION_CASCADE: (
        "Capturar aceleraciones de precio por liquidaciones en cascada. "
        "Señal: OI alta + funding extremo + cambio súbito en taker_ratio. "
        "Las liquidaciones masivas amplifican movimientos y crean reversiones. "
        "Usa combinación de open_interest, funding_rate y taker_ratio."
    ),
    HypothesisFamily.SHORT_TREND_EXHAUSTION: (
        "Apostar CONTRA la tendencia cuando muestra señales de agotamiento. "
        "Funciona en regímenes trending cuando el momentum se desacelera. "
        "Señal: ADX cayendo tras momentum alto, volumen decreciente en dirección del trend. "
        "Dirección: SHORT — vende cuando el trend alcista se agota."
    ),
    HypothesisFamily.SHORT_OVERBOUGHT_REVERSAL: (
        "Apostar a reversión bajista cuando el activo está sobrecomprado. "
        "Funciona cuando RSI > 70, precio muy por encima de EMA, volumen seco. "
        "Dirección: SHORT — vende cuando detecta sobrecompra."
    ),
    HypothesisFamily.SHORT_BREAKDOWN: (
        "Vender en ruptura de soportes con confirmación de volumen. "
        "El precio rompe un suelo clave con volumen y momentum negativo. "
        "Usa ruptura de mínimos anteriores, EMA cruzando a la baja, volumen alto. "
        "Dirección: SHORT — vende en breakdown."
    ),
    HypothesisFamily.SHORT_SENTIMENT_EXTREME: (
        "Vender cuando el sentimiento es exuberantemente alcista (complacencia). "
        "Fear & Greed > 70 (greed extremo), funding positivo alto, OI creciendo. "
        "La complacencia precede a las correcciones. "
        "Dirección: SHORT — vende cuando el mercado está eufórico."
    ),
    HypothesisFamily.SHORT_VOLATILITY_CRUSH: (
        "Vender cuando la volatilidad se comprime y el precio está en tope. "
        "ATR bajo + precio cerca del máximo reciente + volumen decreciente. "
        "La compresión de volatilidad en topos señala distribución. "
        "Dirección: SHORT — vende antes de la expansión bajista."
    ),
}


# Template: defines parameter space for each family
# Each template has default params + ranges for search engine
FAMILY_TEMPLATES: dict[HypothesisFamily, dict[str, Any]] = {
    HypothesisFamily.MOMENTUM_CONTINUATION: {
        "indicators": ["ema", "macd", "adx", "close"],
        "params": {
            "adx_min": {"default": 25, "range": [20, 40], "type": int},
            "ema_period": {"default": 20, "range": [10, 50], "type": int},
            "tp_pct": {"default": 0.04, "range": [0.02, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "conditions_template": [
            {"indicator": "adx", "op": "gt", "value": "{adx_min}"},
            {"indicator": "close", "op": "gt_rolling", "rolling": "ema", "period": "{ema_period}"},
        ],
    },
    HypothesisFamily.VOLATILITY_EXPANSION: {
        "indicators": ["atr", "close", "volume"],
        "params": {
            "atr_period": {"default": 14, "range": [7, 21], "type": int},
            "compression_ratio": {"default": 0.5, "range": [0.3, 0.8], "type": float},
            "tp_pct": {"default": 0.05, "range": [0.03, 0.10], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "conditions_template": [
            {"indicator": "atr", "op": "lt_rolling", "rolling": "sma", "period": "{atr_period}"},
        ],
    },
    HypothesisFamily.MEAN_REVERSION: {
        "indicators": ["rsi", "vwap", "close", "volume"],
        "params": {
            "rsi_period": {"default": 14, "range": [7, 21], "type": int},
            "rsi_low": {"default": 30, "range": [20, 40], "type": int},
            "vol_period": {"default": 20, "range": [10, 50], "type": int},
            "tp_pct": {"default": 0.03, "range": [0.015, 0.06], "type": float},
            "sl_pct": {"default": 0.015, "range": [0.01, 0.03], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "conditions_template": [
            {"indicator": "rsi", "op": "lt", "value": "{rsi_low}"},
            {"indicator": "volume", "op": "gt_rolling", "rolling": "sma", "period": "{vol_period}"},
        ],
    },
    HypothesisFamily.REGIME_TRANSITION: {
        "indicators": ["adx", "regime", "ema"],
        "params": {
            "adx_period": {"default": 14, "range": [7, 21], "type": int},
            "adx_change": {"default": 10, "range": [5, 20], "type": int},
            "tp_pct": {"default": 0.05, "range": [0.03, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 48, "range": [24, 96], "type": int},
        },
        "conditions_template": [
            {"indicator": "adx", "op": "gt", "value": "{adx_change}"},
            {"indicator": "regime", "op": "in", "value": [1, 2]},
        ],
    },
    HypothesisFamily.SENTIMENT_DIVERGENCE: {
        "indicators": ["fear_greed", "close", "rsi"],
        "params": {
            "fear_threshold": {"default": 25, "range": [10, 40], "type": int},
            "tp_pct": {"default": 0.04, "range": [0.02, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "conditions_template": [
            {"indicator": "fear_greed", "op": "lt", "value": "{fear_threshold}"},
        ],
    },
    HypothesisFamily.BREAKOUT_STRUCTURE: {
        "indicators": ["close", "volume", "atr"],
        "params": {
            "range_period": {"default": 20, "range": [10, 50], "type": int},
            "breakout_pct": {"default": 0.02, "range": [0.01, 0.05], "type": float},
            "tp_pct": {"default": 0.05, "range": [0.03, 0.10], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "conditions_template": [
            {"indicator": "breakout_pct_above_high", "op": "gt", "value": "{breakout_pct}"},
            {"indicator": "volume", "op": "gt_rolling", "rolling": "sma", "period": 20},
        ],
        "feature_requirements": [
            {"name": "breakout_pct_above_high", "formula": "(close - max_roll(high.shift(1), {range_period})) / max_roll(high.shift(1), {range_period})", "description": "Percentage above previous rolling high (shifted to avoid lookahead)"},
        ],
    },
    HypothesisFamily.VOLATILITY_COMPRESSION_BREAKOUT: {
        "indicators": ["atr", "close", "volume"],
        "params": {
            "atr_period": {"default": 14, "range": [7, 21], "type": int},
            "compression_lookback": {"default": 20, "range": [10, 40], "type": int},
            "compression_threshold": {"default": 0.8, "range": [0.6, 0.9], "type": float},
            "tp_pct": {"default": 0.06, "range": [0.03, 0.12], "type": float},
            "sl_pct": {"default": 0.025, "range": [0.015, 0.05], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "conditions_template": [
            {"indicator": "atr_compression_ratio", "op": "lt", "value": "{compression_threshold}"},
            {"indicator": "atr", "op": "gt_rolling", "rolling": "sma", "period": 3},
            {"indicator": "volume", "op": "gt_rolling", "rolling": "sma", "period": 20},
        ],
        "feature_requirements": [
            {"name": "atr_compression_ratio", "formula": "atr_14 / sma(atr_14, {compression_lookback})", "description": "ATR compression ratio (current vs historical)"},
        ],
    },
    HypothesisFamily.FUNDING_REVERSAL: {
        "indicators": ["funding_rate", "close", "rsi", "volume"],
        "params": {
            "funding_threshold": {"default": 0.0005, "range": [0.0002, 0.002], "type": float},
            "rsi_threshold": {"default": 30, "range": [20, 45], "type": int},
            "tp_pct": {"default": 0.04, "range": [0.02, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "conditions_template": [
            {"indicator": "funding_rate", "op": "gt", "value": "{funding_threshold}"},
            {"indicator": "rsi", "op": "gt", "value": 50},
        ],
        "feature_requirements": [
            {"name": "funding_rate", "formula": "funding_rate", "description": "Perpetual futures funding rate (8h, ffill to 15m)"},
        ],
    },
    HypothesisFamily.CROWDED_POSITIONING: {
        "indicators": ["open_interest", "long_short_ratio", "close", "volume"],
        "params": {
            "ls_threshold": {"default": 1.5, "range": [1.2, 2.5], "type": float},
            "oi_lookback": {"default": 24, "range": [12, 48], "type": int},
            "tp_pct": {"default": 0.05, "range": [0.02, 0.10], "type": float},
            "sl_pct": {"default": 0.025, "range": [0.01, 0.05], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "conditions_template": [
            {"indicator": "long_short_ratio", "op": "gt", "value": "{ls_threshold}"},
            {"indicator": "oi_growth", "op": "gt", "value": 0},
        ],
        "feature_requirements": [
            {"name": "oi_growth", "formula": "pct_change(open_interest, 24)", "description": "OI growth rate (24 bars)"},
        ],
    },
    HypothesisFamily.LIQUIDATION_CASCADE: {
        "indicators": ["funding_rate", "open_interest", "taker_ratio", "atr", "close"],
        "params": {
            "funding_extreme": {"default": 0.001, "range": [0.0005, 0.003], "type": float},
            "taker_threshold": {"default": 1.5, "range": [1.2, 2.0], "type": float},
            "tp_pct": {"default": 0.06, "range": [0.03, 0.12], "type": float},
            "sl_pct": {"default": 0.03, "range": [0.015, 0.06], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "abs_funding_rate", "op": "gt", "value": "{funding_extreme}"},
            {"indicator": "taker_ratio", "op": "lt", "value": 0.7},
            {"indicator": "atr", "op": "gt_rolling", "rolling": "sma", "period": 20},
        ],
        "feature_requirements": [
            {"name": "abs_funding_rate", "formula": "abs(funding_rate)", "description": "Absolute funding rate (extreme in either direction)"},
        ],
    },
    HypothesisFamily.SHORT_TREND_EXHAUSTION: {
        "indicators": ["adx", "ema", "close", "volume"],
        "params": {
            "adx_peak": {"default": 35, "range": [25, 50], "type": int},
            "ema_period": {"default": 21, "range": [10, 50], "type": int},
            "tp_pct": {"default": 0.04, "range": [0.02, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "adx", "op": "lt", "value": "{adx_peak}"},
            {"indicator": "adx", "op": "cross_below", "value": 25},
            {"indicator": "close", "op": "gt_rolling", "rolling": "ema", "period": "{ema_period}"},
        ],
        "description_for_llm": "SHORT: sell when trend is exhausted — ADX falling below threshold after being high, while price still above EMA (overextended).",
    },
    HypothesisFamily.SHORT_OVERBOUGHT_REVERSAL: {
        "indicators": ["rsi", "close", "ema", "volume"],
        "params": {
            "rsi_high": {"default": 70, "range": [60, 85], "type": int},
            "ema_period": {"default": 21, "range": [10, 50], "type": int},
            "tp_pct": {"default": 0.03, "range": [0.015, 0.06], "type": float},
            "sl_pct": {"default": 0.015, "range": [0.01, 0.03], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "rsi", "op": "gt", "value": "{rsi_high}"},
            {"indicator": "close", "op": "gt_rolling", "rolling": "ema", "period": "{ema_period}"},
        ],
        "description_for_llm": "SHORT: sell overbought conditions — RSI above threshold + price above EMA (extended).",
    },
    HypothesisFamily.SHORT_BREAKDOWN: {
        "indicators": ["close", "volume", "atr", "ema"],
        "params": {
            "range_period": {"default": 20, "range": [10, 50], "type": int},
            "breakdown_pct": {"default": -0.02, "range": [-0.05, -0.01], "type": float},
            "tp_pct": {"default": 0.05, "range": [0.03, 0.10], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 24, "range": [12, 48], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "breakdown_pct_below_low", "op": "gt", "value": "{breakdown_pct}"},
            {"indicator": "volume", "op": "gt_rolling", "rolling": "sma", "period": 20},
        ],
        "feature_requirements": [
            {"name": "breakdown_pct_below_low", "formula": "(close - min_roll(low.shift(1), {range_period})) / min_roll(low.shift(1), {range_period})", "description": "Percentage below previous rolling low (shifted to avoid lookahead). Negative values mean below the low."},
        ],
    },
    HypothesisFamily.SHORT_SENTIMENT_EXTREME: {
        "indicators": ["close", "ema", "volume"],
        "params": {
            "ema_period": {"default": 21, "range": [10, 50], "type": int},
            "tp_pct": {"default": 0.04, "range": [0.02, 0.08], "type": float},
            "sl_pct": {"default": 0.02, "range": [0.01, 0.04], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "close", "op": "gt_rolling", "rolling": "ema", "period": "{ema_period}"},
            {"indicator": "volume", "op": "lt_rolling", "rolling": "sma", "period": 20},
        ],
        "description_for_llm": "SHORT: sell when price is above EMA but volume is drying up — distribution pattern.",
    },
    HypothesisFamily.SHORT_VOLATILITY_CRUSH: {
        "indicators": ["atr", "close", "volume", "rsi"],
        "params": {
            "atr_period": {"default": 14, "range": [7, 21], "type": int},
            "compression_threshold": {"default": 0.8, "range": [0.6, 0.9], "type": float},
            "rsi_high": {"default": 60, "range": [50, 75], "type": int},
            "tp_pct": {"default": 0.05, "range": [0.03, 0.10], "type": float},
            "sl_pct": {"default": 0.025, "range": [0.015, 0.05], "type": float},
            "horizon_bars": {"default": 36, "range": [12, 72], "type": int},
        },
        "direction": "short",
        "conditions_template": [
            {"indicator": "atr", "op": "lt_rolling", "rolling": "sma", "period": "{atr_period}"},
            {"indicator": "rsi", "op": "gt", "value": "{rsi_high}"},
            {"indicator": "close", "op": "gt_rolling", "rolling": "sma", "period": 50},
        ],
        "description_for_llm": "SHORT: sell when volatility is compressed, RSI is high, and price is above long-term SMA — compression at the top signals distribution.",
    },
}


def list_families() -> list[str]:
    return [f.value for f in HypothesisFamily]


def get_family(name: str) -> HypothesisFamily | None:
    for f in HypothesisFamily:
        if f.value == name:
            return f
    return None


def describe_family(family: HypothesisFamily) -> str:
    return FAMILY_DESCRIPTIONS.get(family, "")


def get_template(family: HypothesisFamily) -> dict:
    return FAMILY_TEMPLATES.get(family, {})


def family_description_for_prompt() -> str:
    lines = []
    for f in HypothesisFamily:
        desc = FAMILY_DESCRIPTIONS.get(f, "")
        lines.append(f"  • {f.value}: {desc}")
    return "\n".join(lines)
