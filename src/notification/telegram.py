"""Telegram notifier — sends strategy findings to a Telegram chat.
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


def _fmt_strat(r: dict) -> str:
    """One-liner: name + Sharpe + WR + PF + OOS."""
    name = r.get("run_id", "?")[:22]
    s = r.get("sharpe", 0)
    wr = r.get("win_rate", 0)
    pf = r.get("profit_factor", 0)
    pnl = r.get("total_pnl", 0)
    nt = r.get("n_trades", 0)
    oos = r.get("oos_sharpe")
    oos_s = f" OOS={oos:+.2f}" if isinstance(oos, (int, float)) and r.get("oos_trades", 0) >= 10 else ""
    of = " ⚠️" if r.get("overfit") else ""
    return f"`{name}` S={s:+.2f} WR={wr:.0%} PF={pf:.2f} ${pnl:+.0f} T={nt}{oos_s}{of}"


async def notify_start(rounds: int, n: int, days: int, timeframe: str, market_ctx: str = "") -> None:
    await _send(
        f"🤖 *Research* `{n} estrat · {days}d {timeframe}`\n"
        f"📊 `{market_ctx}`"
    )


async def notify_strategy_found(
    name: str, sharpe: float, win_rate: float, profit_factor: float,
    total_pnl: float, n_trades: int, analysis: str, rules: str,
) -> None:
    emoji = "🚀" if sharpe > 0.3 else "⭐"
    analysis_short = (analysis[:150] + "...") if analysis else ""
    try:
        rules_obj = json.loads(rules) if isinstance(rules, str) else rules
        rules_str = json.dumps(rules_obj.get("entry_conditions", rules_obj), indent=2)[:300]
    except Exception:
        rules_str = rules[:200]

    msg = (
        f"{emoji} *Sharpe {sharpe:+.2f}* — `{name[:35]}`\n"
        f"`WR {win_rate:.0%} · PF {profit_factor:.2f} · ${total_pnl:+.0f} · {n_trades} trades`\n"
        f"```\n{rules_str}\n```"
    )
    if analysis_short:
        msg += f"\n💡 _{analysis_short}_"
    await _send(msg)


async def notify_digest(
    elapsed_h: float, n_tested: int, n_positive: int, best_sharpe: float,
    rounds_completed: int, total_cost: float,
    top_strategies: list[dict] | None = None,
    last_round_results: list[dict] | None = None,
) -> None:
    msg = f"⏱️ *{elapsed_h:.1f}h* · `{n_tested} estratég · {rounds_completed} rondas · ${total_cost:.4f}`\n"

    if top_strategies:
        best = top_strategies[0]
        tag = " ✅" if best.get("passes_gates") else ""
        oos = best.get("oos_sharpe")
        oos_s = f"  OOS={oos:+.2f}" if isinstance(oos, (int, float)) and best.get("oos_trades", 0) >= 10 else ""
        of = " ⚠️OVERFIT" if best.get("overfit") else ""
        msg += f"\n🥇 `{best['run_id'][:20]}` S={best['sharpe']:+.2f} WR={best['win_rate']:.0%} PF={best['profit_factor']:.2f}${oos_s}{of}{tag}\n"

    survivors = sum(1 for r in (last_round_results or []) if r.get("n_trades", 0) > 0)
    oos_ok = sum(1 for r in (last_round_results or []) if r.get("oos_trades", 0) >= 10 and not r.get("overfit"))
    new_ind = sum(1 for r in (last_round_results or []) if r.get("run_id", "").startswith("🧪"))
    extras = []
    if survivors:
        extras.append(f"{survivors} backteadas")
    if oos_ok:
        extras.append(f"{oos_ok} pasaron OOS")
    if extras:
        msg += f"🔄 `{' · '.join(extras)}`"

    await _send(msg)


async def notify_hourly_leaderboard(elapsed_h: float, all_results: list[dict]) -> None:
    valid = [r for r in all_results if r.get("n_trades", 0) >= 10]
    if not valid:
        await _send(f"⏰ *Hora {elapsed_h:.0f}* — Sin datos aún")
        return

    valid.sort(key=lambda r: r["sharpe"], reverse=True)
    n_pos = sum(1 for r in valid if r["sharpe"] > 0)
    avg_s = sum(r["sharpe"] for r in valid) / len(valid)
    oos_ok = sum(1 for r in valid if r.get("oos_trades", 0) >= 10 and not r.get("overfit"))

    msg = f"🏆 *Hora {elapsed_h:.0f}* · `{len(valid)} estrat · {n_pos} S>0 · {oos_ok} pasaron OOS`\n`Sharpe medio: {avg_s:+.2f}`\n"
    for i, r in enumerate(valid[:5], 1):
        msg += f"`{i}.` {_fmt_strat(r)}\n"
    await _send(msg)


async def notify_new_indicator(name: str, formula: str, source: str = "LLM") -> None:
    await _send(f"🧪 `{name}` = `{formula[:100]}`")


async def notify_done(
    total_elapsed_h: float, total_tested: int, total_positive: int,
    best_sharpe: float, best_name: str, total_cost: float,
    all_results: list[dict] | None = None,
) -> None:
    msg = f"✅ *Done* `{total_elapsed_h:.1f}h · {total_tested} estrat · ${total_cost:.4f}`\n"

    if all_results:
        valid = sorted(
            [r for r in all_results if r.get("n_trades", 0) >= 10],
            key=lambda r: r["sharpe"], reverse=True,
        )
        if valid:
            msg += f"\n*Top 3:*\n"
            for i, r in enumerate(valid[:3], 1):
                msg += f"`{i}.` {_fmt_strat(r)}\n"

            best = valid[0]
            oos = best.get("oos_sharpe")
            oos_s = f"  OOS={oos:+.2f}" if isinstance(oos, (int, float)) and best.get("oos_trades", 0) >= 10 else ""
            of = " ⚠️ OVERFIT" if best.get("overfit") else ""
            msg += f"\n🥇 `{best['run_id']}` S={best['sharpe']:+.2f} WR={best['win_rate']:.0%} PF={best['profit_factor']:.2f}{oos_s}{of}"

    await _send(msg)
