"""Analyst — uses LLM to explain backtest results.

Produces structured analysis (strengths, weaknesses, suggestions)
that is saved alongside the strategy for future training.
"""
from __future__ import annotations

import json
import logging

from src.brain.llm_client import LLMClient
from src.brain.prompts import ANALYST_SYSTEM

logger = logging.getLogger(__name__)


async def analyze_strategy(
    llm: LLMClient,
    strategy_json: dict,
    summary: dict,
    market_context: str,
    days: int = 365,
    timeframe: str = "15m",
) -> dict:
    """Analyze a backtest result and return structured insights.

    Returns
    -------
    dict with keys: analysis, strengths, weaknesses, suggestions, confidence
    """
    system = ANALYST_SYSTEM.format(
        strategy_json=json.dumps(strategy_json, indent=2),
        days=days,
        timeframe=timeframe,
        sharpe=summary.get("sharpe", 0),
        win_rate=summary.get("win_rate", 0),
        profit_factor=summary.get("profit_factor", 0),
        total_pnl=summary.get("total_pnl", 0),
        n_trades=summary.get("n_trades", 0),
        max_dd=summary.get("max_dd", 0),
        passes_gates=summary.get("passes_gates", False),
        market_context=market_context,
    )
    user_prompt = "Analiza esta estrategia y sus resultados."

    response = await llm.generate(
        system_prompt=system,
        user_prompt=user_prompt,
        response_format="json_object",
        temperature=0.3,  # lower temp for analysis
    )

    raw = response["content"]
    if isinstance(raw, str):
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("brain: analyst returned invalid JSON: %s", raw[:200])
            return {"analysis": raw, "strengths": [], "weaknesses": [], "suggestions": [], "confidence": 0}
    else:
        analysis = raw

    logger.info("brain: analyst done — confidence=%.2f", analysis.get("confidence", 0))
    return analysis
