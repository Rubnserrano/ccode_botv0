"""Telegram notifier — sends strategy findings to a Telegram chat.

Requires:
  TELEGRAM_BOT_TOKEN  (from @BotFather)
  TELEGRAM_CHAT_ID    (your chat ID)
"""
from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


async def _send(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_API_BASE}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            )
            resp.raise_for_status()
            return True
    except Exception as e:
        logger.warning("telegram: send failed: %s", e)
        return False


def _fmt_metrics(r: dict) -> str:
    """Format a strategy result into a compact one-liner."""
    name = r.get("run_id", "?")[:25]
    s = r.get("sharpe", 0)
    wr = r.get("win_rate", 0)
    pf = r.get("profit_factor", 0)
    pnl = r.get("total_pnl", 0)
    nt = r.get("n_trades", 0)
    tag = " ✅" if r.get("passes_gates") else ""
    return f"`{name:25s}` S={s:+.2f} WR={wr:.0%} PF={pf:.2f} PnL=${pnl:+.0f} T={nt}{tag}"


async def notify_start(rounds: int, n: int, days: int, timeframe: str, market_ctx: str = "") -> None:
    """Notify that the research has started with full context."""
    msg = (
        f"🤖 *Research Loop Started*\n"
        f"`{n} estrategias/ronda · {days}d {timeframe} · deepseek-chat ($0)`\n\n"
        f"📊 `{market_ctx}`\n\n"
        f"⏱ Cada 5min: resumen de mejores estrategias\n"
        f"🏆 Cada hora: leaderboard detallado\n"
        f"🚀 Sharpe > 0 → notificación inmediata"
    )
    await _send(msg)


async def notify_strategy_found(
    name: str, sharpe: float, win_rate: float, profit_factor: float,
    total_pnl: float, n_trades: int, analysis: str, rules: str,
) -> None:
    """Notify about a promising strategy with full context."""
    emoji = "🚀" if sharpe > 0.3 else "⭐" if sharpe > 0 else "📊"
    analysis_short = analysis[:200] if analysis else ""
    try:
        rules_parsed = json.loads(rules) if isinstance(rules, str) else rules
        rules_str = json.dumps(rules_parsed, indent=2)[:400]
    except Exception:
        rules_str = rules[:300]

    msg = (
        f"{emoji} *Nueva estrategia con Sharpe {sharpe:+.2f}*\n"
        f"`{name[:40]}`\n\n"
        f"📈 `Sharpe {sharpe:+.2f}`  `WR {win_rate:.0%}`\n"
        f"💰 `PF {profit_factor:.2f}`  `PnL ${total_pnl:+.0f}`  `Trades {n_trades}`\n\n"
        f"*Reglas:*\n```\n{rules_str}\n```\n"
    )
    if analysis_short:
        msg += f"\n💡 _{analysis_short}_"
    await _send(msg)


async def notify_digest(
    elapsed_h: float,
    n_tested: int,
    n_positive: int,
    best_sharpe: float,
    rounds_completed: int,
    total_cost: float,
    top_strategies: list[dict] | None = None,
    last_round_results: list[dict] | None = None,
) -> None:
    """Send a progress digest — every 5 min."""
    msg = (
        f"⏱️ *Resumen parcial — {elapsed_h:.1f}h*\n"
        f"`{n_tested} estrategias · {rounds_completed} rondas · {n_positive} con Sharpe>0`\n"
        f"`Mejor Sharpe: {best_sharpe:+.2f} · Costo: ${total_cost:.4f}`\n"
    )

    if top_strategies:
        best = top_strategies[0]
        msg += f"\n🥇 *Mejor hasta ahora:*\n"
        msg += f"{_fmt_metrics(best)}\n"

    if last_round_results and len(last_round_results) > 0:
        passed = sum(1 for r in last_round_results if r.get("sharpe", -999) > -0.1)
        msg += f"\n🔄 *Última ronda:* {len(last_round_results)} estrategias, {passed} pasaron filtro\n"

    await _send(msg)


async def notify_hourly_leaderboard(
    elapsed_h: float,
    all_results: list[dict],
) -> None:
    """Send an hourly detailed leaderboard with insights."""
    sorted_results = sorted(
        [r for r in all_results if r.get("n_trades", 0) >= 10],
        key=lambda r: r["sharpe"], reverse=True,
    )

    if not sorted_results:
        await _send(f"⏰ *Hora {elapsed_h:.0f}* — Aún pocos datos, esperando más rondas...")
        return

    # Metrics
    n_total = len(sorted_results)
    n_positive = sum(1 for r in sorted_results if r["sharpe"] > 0)
    n_near = sum(1 for r in sorted_results if -0.1 < r["sharpe"] <= 0)
    avg_sharpe = sum(r["sharpe"] for r in sorted_results) / n_total if n_total else 0
    best = sorted_results[0]

    msg = (
        f"🏆 *Leaderboard hora {elapsed_h:.0f}*\n"
        f"`{n_total} estrategias · {n_positive} positivas · {n_near} cerca`\n"
        f"`Sharpe medio: {avg_sharpe:+.2f} · Mejor: {best['sharpe']:+.2f}`\n\n"
        f"*Top 5:*\n"
    )
    for i, r in enumerate(sorted_results[:5], 1):
        msg += f"`{i}.` {_fmt_metrics(r)}\n"

    # Indicators used
    ind_counts = {}
    for r in all_results:
        try:
            rules = json.loads(r.get("rules_json", "{}"))
            for c in rules.get("entry_conditions", []):
                ind = c.get("indicator", "?")
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
        except Exception:
            pass
    if ind_counts:
        most_used = sorted(ind_counts.items(), key=lambda x: -x[1])[:5]
        msg += f"\n📊 *Indicadores más usados:*\n"
        for ind, count in most_used:
            msg += f"`{ind:15s}` {count} veces\n"

    await _send(msg)


async def notify_new_indicator(name: str, formula: str, source: str = "LLM") -> None:
    """Notify that a new indicator was created."""
    msg = (
        f"🧪 *Nuevo indicador creado*\n"
        f"`{name}` = `{formula[:120]}`"
    )
    await _send(msg)


async def notify_done(
    total_elapsed_h: float,
    total_tested: int,
    total_positive: int,
    best_sharpe: float,
    best_name: str,
    total_cost: float,
    all_results: list[dict] | None = None,
) -> None:
    """Notify that the research run is complete with full summary."""
    msg = (
        f"✅ *Research Complete*\n"
        f"`⏱ {total_elapsed_h:.1f}h  ·  {total_tested} estrategias  ·  "
        f"{total_positive} con Sharpe>0`\n"
        f"`💰 Costo total: ${total_cost:.4f}`\n\n"
    )

    if all_results:
        sorted_results = sorted(
            [r for r in all_results if r.get("n_trades", 0) >= 10],
            key=lambda r: r["sharpe"], reverse=True,
        )
        if sorted_results:
            msg += f"*🏆 Top 3 final:*\n"
            for i, r in enumerate(sorted_results[:3], 1):
                msg += f"`{i}.` {_fmt_metrics(r)}\n"

            # Best recommendation
            best = sorted_results[0]
            msg += f"\n*🥇 Mejor estrategia:* `{best['run_id']}`\n"
            msg += f"Sharpe {best['sharpe']:+.2f} · WR {best['win_rate']:.0%} · PF {best['profit_factor']:.2f}\n"

    msg += f"\n📁 `data/parquet/research/leaderboard.parquet`"
    await _send(msg)
