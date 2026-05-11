"""Strategist — uses LLM to generate trading strategies with logical sense.

Instead of random grid search, the LLM generates strategies based on
current market context and past results.
"""
from __future__ import annotations

import json
import logging

from src.brain.llm_client import LLMClient
from src.brain.prompts import STRATEGIST_SYSTEM

logger = logging.getLogger(__name__)


async def generate_strategies(
    llm: LLMClient,
    market_context: str,
    previous_results: list[dict],
    n: int = 3,
) -> list[dict]:
    """Generate N strategies using the LLM.

    Parameters
    ----------
    llm : LLMClient
        Connected LLM client.
    market_context : str
        Human-readable description of current market (regime dist, etc).
    previous_results : list[dict]
        Past strategy results from the leaderboard.
    n : int
        Number of strategies to generate.

    Returns
    -------
    list[dict]
        List of strategy dicts, each with name, entry_conditions, exit.
    """
    # Build previous results summary (last 10 with trades)
    prev_summary = ""
    if previous_results:
        lines = []
        for r in previous_results[:10]:
            s = r.get("sharpe", 0)
            pf = r.get("profit_factor", 0)
            nt = r.get("n_trades", 0)
            if nt > 0:
                lines.append(f"  S={s:+.2f} PF={pf:.2f} T={nt}")
        if lines:
            prev_summary = "Resultados recientes:\n" + "\n".join(lines)

    user_prompt = f"Genera {n} estrategias para el contexto actual."
    system = STRATEGIST_SYSTEM.format(
        market_context=market_context,
        previous_results=prev_summary,
        n=n,
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
            return []
    elif isinstance(raw, list):
        strategies = raw
    else:
        logger.warning("brain: LLM returned unexpected type: %s", type(raw))
        return []

    if isinstance(strategies, dict) and "strategies" in strategies:
        strategies = strategies["strategies"]

    if not isinstance(strategies, list):
        logger.warning("brain: expected list, got %s", type(strategies))
        return []

    logger.info("brain: strategist generated %d strategies", len(strategies))
    return strategies
