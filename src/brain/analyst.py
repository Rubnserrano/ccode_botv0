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
    wf_sharpe: float = 0.0,
    wf_sharpe_std: float = 0.0,
    wf_cv: float = 999.0,
    oos_sharpe: float = 0.0,
    oos_trades: int = 0,
) -> dict:
    """Analyze a backtest result and return structured insights.

    Parameters
    ----------
    llm : LLMClient
    strategy_json : dict
        The strategy definition.
    summary : dict
        Backtest summary (train metrics).
    market_context : str
        Current market conditions.
    days, timeframe : str
        Data config.
    wf_sharpe, wf_sharpe_std, wf_cv : float
        Walk-forward metrics.
    oos_sharpe : float
        Hold-out OOS Sharpe.
    oos_trades : int
        Number of OOS trades.

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
        wf_sharpe=wf_sharpe,
        wf_sharpe_std=wf_sharpe_std,
        wf_cv=wf_cv,
        oos_sharpe=oos_sharpe,
        oos_trades=oos_trades,
    )
    user_prompt = "Analiza esta estrategia y sus resultados."

    response = await llm.generate(
        system_prompt=system,
        user_prompt=user_prompt,
        response_format="json_object",
        temperature=0.3,
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

    logger.info("brain: analyst done — confidence=%.2f  wf_cv=%.2f  oos=%.2f",
                analysis.get("confidence", 0), wf_cv, oos_sharpe)
    return analysis
