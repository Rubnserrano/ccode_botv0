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
            {"name": "breakout_pct_above_high", "formula": "(close - max_roll(high, {range_period})) / max_roll(high, {range_period})", "description": "Percentage above rolling high"},
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
