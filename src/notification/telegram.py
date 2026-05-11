"""Telegram notifier — sends strategy findings to a Telegram chat.

Requires:
  TELEGRAM_BOT_TOKEN  (from @BotFather)
  TELEGRAM_CHAT_ID    (your chat ID)

Usage:
    from src.notification.telegram import send_strategy, send_digest
    await send_strategy(summary, analysis)
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Rate limit: max 20 messages/min per chat
_sent_count = 0


async def _send(text: str) -> bool:
    """Send a plain text message to Telegram."""
    global _sent_count
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{_API_BASE}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            )
            resp.raise_for_status()
            _sent_count += 1
            return True
    except Exception as e:
        logger.warning("telegram: send failed: %s", e)
        return False


async def notify_start(rounds: int, n: int, days: int, timeframe: str) -> None:
    """Notify that the overnight research has started."""
    msg = (
        "🤖 *Research Loop Started*\n"
        f"• Estrategias por ronda: {n}\n"
        f"• Datos: {days}d {timeframe}\n"
        f"• Modelo: deepseek-chat ($0)\n"
        f"• Notificaré si encuentro Sharpe > 0\n"
        f"• Enviaré digest cada hora"
    )
    await _send(msg)


async def notify_strategy_found(
    name: str,
    sharpe: float,
    win_rate: float,
    profit_factor: float,
    total_pnl: float,
    n_trades: int,
    analysis: str,
    rules: str,
) -> None:
    """Notify about a promising strategy."""
    emoji = "🚀" if sharpe > 0.3 else "⭐" if sharpe > 0 else "📊"
    # Truncate long fields
    analysis_short = analysis[:200] if analysis else ""
    rules_short = rules[:300] if rules else ""

    msg = (
        f"{emoji} *Estrategia encontrada*\n"
        f"`{name[:40]}`\n"
        f"Sharpe: `{sharpe:+.2f}`  |  WinRate: `{win_rate:.1%}`\n"
        f"Profit Factor: `{profit_factor:.2f}`  |  Trades: `{n_trades}`\n"
        f"PnL: `${total_pnl:+.0f}`\n\n"
        f"*Reglas:*\n`{rules_short}`\n"
    )
    if analysis_short:
        msg += f"\n*Análisis:*\n_{analysis_short}_"
    await _send(msg)


async def notify_digest(
    elapsed_h: float,
    n_tested: int,
    n_positive: int,
    best_sharpe: float,
    rounds_completed: int,
    total_cost: float,
    top_strategies: list[dict] | None = None,
) -> None:
    """Send a progress digest with top strategies."""
    msg = (
        f"⏰ *Digest — {elapsed_h:.1f}h transcurridas*\n"
        f"• Estrategias probadas: `{n_tested}`\n"
        f"• Con Sharpe > 0: `{n_positive}`\n"
        f"• Mejor Sharpe: `{best_sharpe:+.2f}`\n"
        f"• Rondas completadas: `{rounds_completed}`\n"
        f"• Costo total LLM: `${total_cost:.4f}`\n"
    )
    if top_strategies:
        msg += f"\n*Top 3 de la sesión:*\n"
        for i, s in enumerate(top_strategies[:3], 1):
            s_name = s.get("run_id", "?")[:30]
            msg += f"`{i}. {s_name}` S={s['sharpe']:+.2f} WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} T={s['n_trades']}\n"
    await _send(msg)


async def notify_hourly_leaderboard(
    elapsed_h: float,
    all_results: list[dict],
) -> None:
    """Send an hourly detailed leaderboard."""
    sorted_results = sorted(
        [r for r in all_results if r.get("n_trades", 0) >= 10],
        key=lambda r: r["sharpe"], reverse=True,
    )
    if not sorted_results:
        await _send(f"⏰ *Hora {elapsed_h:.0f}* — Sin estrategias con suficientes trades aun.")
        return

    msg = f"🏆 *Leaderboard hora {elapsed_h:.0f}* ({len(sorted_results)} estrategias)\n\n"
    for i, r in enumerate(sorted_results[:5], 1):
        msg += (
            f"`{i}.` `{r['run_id'][:30]:30s}`\n"
            f"    S=`{r['sharpe']:+.2f}`  WR=`{r['win_rate']:.0%}`  "
            f"PF=`{r['profit_factor']:.2f}`  PnL=`${r['total_pnl']:+.0f}`  "
            f"T=`{r['n_trades']}`\n"
        )
    await _send(msg)


async def notify_new_indicator(name: str, formula: str, source: str = "LLM") -> None:
    """Notify that a new indicator was created."""
    msg = (
        f"🧪 *Nuevo indicador creado por {source}*\n"
        f"Nombre: `{name}`\n"
        f"Formula: `{formula[:100]}`"
    )
    await _send(msg)


async def notify_done(
    total_elapsed_h: float,
    total_tested: int,
    total_positive: int,
    best_sharpe: float,
    best_name: str,
    total_cost: float,
) -> None:
    """Notify that the research run is complete."""
    msg = (
        f"✅ *Research Complete*\n"
        f"• Duración: `{total_elapsed_h:.1f}h`\n"
        f"• Estrategias probadas: `{total_tested}`\n"
        f"• Con Sharpe > 0: `{total_positive}`\n"
        f"• Mejor estrategia: `{best_name}` con Sharpe `{best_sharpe:+.2f}`\n"
        f"• Costo total: `${total_cost:.4f}`\n\n"
        f"*Leaderboard:* `data/parquet/research/leaderboard.parquet`"
    )
    await _send(msg)
