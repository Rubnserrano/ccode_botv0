"""Strategist — generates hypotheses (not strategies).

Output: list of {family, reasoning, params}
Search engine converts hypotheses into concrete strategies.
"""
from __future__ import annotations

import json
import logging

from src.brain.llm_client import LLMClient
from src.brain.prompts import STRATEGIST_SYSTEM
from src.brain.concept_stats import get_avg_reward, get_concept_stats
from src.strategy_engine.hypothesis import family_description_for_prompt, list_families

logger = logging.getLogger(__name__)


def _build_concept_summary() -> str:
    """Build a summary of concept stats for the strategist prompt."""
    stats = get_concept_stats()
    if not stats:
        return "Aún no hay datos. Explora libremente."

    # Aggregate by family
    from collections import defaultdict
    by_family = defaultdict(list)
    for r in stats:
        by_family[r.get("archetype", "unknown")].append(r.get("reward", 0))

    lines = []
    for family in list_families():
        rewards = by_family.get(family, [])
        if rewards:
            positive = [r for r in rewards if r > 0]
            avg = sum(positive) / len(positive) if positive else 0
            lines.append(f"  • {family}: {len(rewards)} intentos, reward medio {avg:.2f}")
        else:
            lines.append(f"  • {family}: NO explorada aún")

    return "\n".join(lines)


async def generate_hypotheses(
    llm: LLMClient,
    market_context: str,
    previous_results: list[dict],
    n: int = 3,
    round_feedback: str = "",
) -> list[dict]:
    """Generate N hypotheses using the LLM.

    Returns
    -------
    hypotheses : list[dict]
        Each dict has: family, reasoning, params (optional)
    """
    if not round_feedback:
        round_feedback = "Primera ronda. Explora las familias libremnte."

    families_desc = family_description_for_prompt()
    concept_summary = _build_concept_summary()

    # Build previous summary
    prev_summary = ""
    best_example = None
    if previous_results:
        with_trades = [r for r in previous_results if r.get("n_trades", 0) > 0]
        sorted_results = sorted(with_trades, key=lambda r: r.get("sharpe", -999), reverse=True)
        if sorted_results:
            best = sorted_results[0]
            best_example = f"Mejor familia previa: {best.get('archetype', '?')} reward={best.get('sharpe', 0):.2f}"
        lines = [f"  {r.get('archetype', '?')}: S={r.get('sharpe', 0):+.2f}" for r in sorted_results[:5] if r.get("n_trades", 0) > 0]
        if lines:
            prev_summary = "\n".join(lines)

    if not best_example:
        best_example = "Aún no hay datos. Explora libremente."

    system = STRATEGIST_SYSTEM.format(
        market_context=market_context,
        families_description=families_desc,
        n=n,
        round_feedback=round_feedback,
        previous_results=prev_summary,
    )

    user_prompt = f"Genera {n} hipótesis para el mercado actual."

    response = await llm.generate(
        system_prompt=system,
        user_prompt=user_prompt,
        response_format="json_object",
    )

    raw = response["content"]

    if isinstance(raw, str):
        try:
            hypotheses = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("brain: LLM returned invalid JSON: %s", raw[:200])
            return []
    elif isinstance(raw, list):
        hypotheses = raw
    else:
        hypotheses = raw

    # Normalize: single object → wrap in list
    if isinstance(hypotheses, dict):
        if "family" in hypotheses:
            hypotheses = [hypotheses]
        else:
            hypotheses = hypotheses.get("hypotheses") or hypotheses.get("strategies") or list(hypotheses.values())

    if not isinstance(hypotheses, list):
        logger.warning("brain: expected list, got %s", type(hypotheses))
        return []

    # Extract reasoning text
    for h in hypotheses:
        if isinstance(h, dict):
            h["reasoning"] = h.pop("reasoning", h.pop("_reasoning", ""))

    valid = [h for h in hypotheses if isinstance(h, dict) and h.get("family") in list_families()]
    logger.info("brain: strategist generated %d hypotheses (%d valid)", len(hypotheses), len(valid))
    return valid
