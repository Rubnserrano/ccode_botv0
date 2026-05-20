"""Critic — evaluates strategy quality before backtesting.

Pipeline:
  1. Heuristic critic (barato): genericidad, formato, redundancia básica
  2. Embedding-based novelty (Fase 2)
  3. LLM critic (Fase 3, solo para top-K)

Fase 1: solo critic heurístico.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns that are too generic / well-known
# Format: (indicator, op, value_check) — reject if ALL conditions match
_GENERIC_PATTERNS: list[tuple[str, str, callable]] = [
    # RSI < 30 sin filtro adicional (oversold genérico)
    ("rsi", "lt", lambda v: isinstance(v, (int, float)) and v <= 35),
    # RSI > 70 sin filtro adicional (overbought genérico)
    ("rsi", "gt", lambda v: isinstance(v, (int, float)) and v >= 65),
    # MACD cruce por cero
    ("macd", "cross_above", lambda v: v == 0),
    ("macd", "cross_below", lambda v: v == 0),
    # close vs SMA(20) básico (no ema, no periodos largos)
    ("close", "gt_rolling", lambda v: v is None),
    ("close", "lt_rolling", lambda v: v is None),
]

# Indicators that are too commonly used alone without additional filter
_GENERIC_INDICATORS = {"rsi", "macd", "vwap", "obi"}


def heuristic_critique(
    strategy_dict: dict,
    past_strategies: list[dict] | None = None,
) -> dict:
    """Heuristic evaluation of a strategy before backtesting.

    Returns
    -------
    dict with keys:
      - passes: bool (True if strategy should be tested)
      - reasons: list[str] (why it failed)
      - novelty_score: float (0-1, 1 = completely novel)
      - archetype: str
    """
    reasons = []
    conditions = strategy_dict.get("entry_conditions", [])
    name = strategy_dict.get("name", "?")

    # 1. Check for generic patterns
    for pat_indicator, pat_op, pat_value_check in _GENERIC_PATTERNS:
        for c in conditions:
            if c.get("indicator") != pat_indicator or c.get("op") != pat_op:
                continue
            # For rolling operators, check rolling method
            if pat_op in ("gt_rolling", "lt_rolling"):
                if c.get("rolling") == "sma" and c.get("indicator") == "close":
                    reasons.append(f"close vs SMA genérico (period={c.get('period', '?')})")
                    break
                continue  # ema rolling or non-close indicators are fine
            # For value-based checks
            v = c.get("value")
            if pat_value_check(v):
                reasons.append(f"patrón genérico: {pat_indicator} {pat_op} {v}")
                break

    # 2. Check for single generic indicator without additional filter
    if len(conditions) == 1:
        ind = conditions[0].get("indicator", "")
        if ind in _GENERIC_INDICATORS:
            reasons.append(f"condición única con indicador genérico: {ind}")

    # 3. Check for invalid value types
    for c in conditions:
        v = c.get("value")
        if v is not None and not isinstance(v, (int, float)):
            # Allow "in" with list values
            if c.get("op") == "in" and isinstance(v, (list, tuple)):
                continue
            reasons.append(f"value no numérico: {v} para {c.get('indicator')}")

    # 4. Check for invalid operators
    valid_ops = {"lt", "gt", "lte", "gte", "eq", "ne", "in",
                 "cross_above", "cross_below",
                 "gt_rolling", "lt_rolling",
                 "streak_gte", "streak_lte"}
    for c in conditions:
        if c.get("op") not in valid_ops:
            reasons.append(f"operador inválido: {c.get('op')}")

    # 5. Redundancia básica contra estrategias pasadas (Fase 1: simple)
    if past_strategies:
        for past in past_strategies[-20:]:
            past_name = past.get("run_id", "")
            if name == past_name:
                reasons.append(f"nombre duplicado: {name}")
                break

    novelty = _compute_novelty(len(reasons), len(conditions))
    archetype = strategy_dict.get("archetype", "unknown")

    passes = len(reasons) == 0

    if not passes:
        logger.info("critic: rejected '%s': %s", name, "; ".join(reasons[:3]))

    return {
        "passes": passes,
        "reasons": reasons,
        "novelty_score": novelty,
        "archetype": archetype,
    }


def _compute_novelty(n_reasons: int, n_conditions: int) -> float:
    """Simple novelty score: penalize if many generic reasons found."""
    base = 1.0
    base -= n_reasons * 0.2
    if n_conditions >= 2:
        base += 0.1  # more conditions = more specific
    return max(0.1, min(1.0, base))
