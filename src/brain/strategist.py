"""Strategist — uses LLM to generate trading strategies with logical sense.

Instead of random grid search, the LLM generates strategies based on
current market context and past results.
"""
from __future__ import annotations

import json
import logging

from src.brain.llm_client import LLMClient
from src.brain.prompts import STRATEGIST_SYSTEM
from src.strategy_engine.registry import INDICATOR_REGISTRY
from src.strategy_engine.generic_calculator import register_dynamic_indicator

logger = logging.getLogger(__name__)

# Built-in indicators (from calculator.py) — everything else is agent-created
BUILTIN_INDICATORS = {
    "close", "high", "low", "open", "volume",
    "rsi", "ema", "macd", "macd_signal", "macd_hist",
    "vwap", "obi", "atr", "adx", "regime",
    "ha_close", "ha_open", "ha_high", "ha_low",
    "fear_greed",
}


def _build_indicator_list() -> tuple[str, int]:
    """Build a dynamic indicator list from the registry, grouped by source."""
    all_inds = set(INDICATOR_REGISTRY.keys())
    builtin = sorted(all_inds & BUILTIN_INDICATORS)
    custom = sorted(all_inds - BUILTIN_INDICATORS)

    lines = []
    lines.append(f"  BUILT-IN ({len(builtin)}): {', '.join(builtin)}")
    if custom:
        lines.append(f"  CREADOS POR AGENTES ({len(custom)}): {', '.join(custom)}")
    else:
        lines.append("  CREADOS POR AGENTES: (ninguno aún — sé el primero en crear uno con new_indicator)")

    return "\n".join(lines), len(all_inds)


async def generate_strategies(
    llm: LLMClient,
    market_context: str,
    previous_results: list[dict],
    n: int = 3,
    round_feedback: str = "",
) -> list[dict]:
    """Generate N strategies using the LLM.

    Returns
    -------
    strategies : list[dict]
        List of strategy dicts.
    new_count : int
        Number of new indicators created.
    """
    if not round_feedback:
        round_feedback = "Primera ronda — no hay feedback previo. Crea indicadores nuevos."

    # Build previous results summary
    prev_summary = ""
    best_example = None
    if previous_results:
        # Find the best strategy to use as inspiration
        with_trades = [r for r in previous_results if r.get("n_trades", 0) > 0]
        sorted_results = sorted(with_trades, key=lambda r: r["sharpe"], reverse=True)

        if sorted_results:
            best = sorted_results[0]
            if best["sharpe"] > 0:
                try:
                    best_rules = json.loads(best.get("rules_json", "{}"))
                    if best_rules:
                        best_example = json.dumps(best_rules, indent=2)
                except Exception:
                    pass

        lines = []
        for r in sorted_results[:10]:
            s = r.get("sharpe", 0)
            pf = r.get("profit_factor", 0)
            nt = r.get("n_trades", 0)
            if nt > 0:
                lines.append(f"  S={s:+.2f} PF={pf:.2f} T={nt}")
        if lines:
            prev_summary = "\n".join(lines)

    if not best_example:
        best_example = "Aún no hay estrategias con Sharpe positivo. Explora combinaciones nuevas."

    user_prompt = f"Genera {n} estrategias para el contexto actual."
    indicator_list, total_count = _build_indicator_list()
    system = STRATEGIST_SYSTEM.format(
        market_context=market_context,
        previous_results=prev_summary,
        best_example=best_example,
        n=n,
        indicator_list=indicator_list,
        total_count=total_count,
        round_feedback=round_feedback,
    )

    response = await llm.generate(
        system_prompt=system,
        user_prompt=user_prompt,
        response_format="json_object",
    )

    raw = response["content"]

    if isinstance(raw, str):
        try:
            strategies = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("brain: LLM returned invalid JSON: %s", raw[:200])
            return [], 0
    elif isinstance(raw, list):
        strategies = raw
    else:
        logger.warning("brain: LLM returned unexpected type: %s", type(raw))
        return [], 0

    if isinstance(strategies, dict):
        strategies = strategies.get("strategies") or strategies.get("entries") or list(strategies.values())

    if not isinstance(strategies, list):
        logger.warning("brain: expected list, got %s", type(strategies))
        return [], 0

    # Handle per-strategy new_indicators
    new_count = 0
    clean_strategies = []
    for s in strategies:
        if not isinstance(s, dict):
            continue
        # Extract and register new_indicator if present
        new_ind = s.pop("new_indicator", None) or s.pop("_new_indicator", None)
        if new_ind:
            try:
                register_dynamic_indicator(
                    name=new_ind["name"],
                    formula=new_ind["formula"],
                    params=new_ind.get("params"),
                    description=new_ind.get("description", "LLM-generated"),
                )
                new_count += 1
                logger.info("brain: created indicator '%s' = %s", new_ind["name"], new_ind["formula"])
            except Exception as e:
                logger.warning("brain: failed to register indicator: %s", e)
        clean_strategies.append(s)

    extra = f" + {new_count} new indicators" if new_count else ""
    logger.info("brain: strategist generated %d strategies%s", len(clean_strategies), extra)
    return clean_strategies, new_count
