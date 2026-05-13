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
    """One-liner: name + Sharpe + WR + PF + OOS + WF CV."""
    name = r.get("run_id", "?")[:22]
    s = r.get("sharpe", 0)
    wr = r.get("win_rate", 0)
    pf = r.get("profit_factor", 0)
    pnl = r.get("total_pnl", 0)
    nt = r.get("n_trades", 0)
    oos = r.get("oos_sharpe")
    oos_s = f" OOS={oos:+.2f}" if isinstance(oos, (int, float)) and r.get("oos_trades", 0) >= 10 else ""
    wf_cv = r.get("wf_cv", 999)
    cv_s = f" CV={wf_cv:.1f}" if wf_cv < 5 else ""
    of = " ⚠️" if r.get("overfit") else ""
    return f"`{name}` S={s:+.2f} WR={wr:.0%} PF={pf:.2f} ${pnl:+.0f} T={nt}{oos_s}{cv_s}{of}"


async def notify_start(rounds: int, n: int, days: int, timeframe: str, market_ctx: str = "") -> None:
    await _send(
        f"🤖 *Research* `{n} estrat · {days}d {timeframe}`\n"
        f"📊 `{market_ctx}`"
    )


async def notify_strategy_found(
    name: str,
    sharpe: float,
    win_rate: float,
    profit_factor: float,
    total_pnl: float,
    n_trades: int,
    analysis: str,
    rules: str,
    oos_sharpe: float | None = None,
    oos_trades: int = 0,
    wf_sharpe: float | None = None,
    wf_cv: float | None = None,
    max_dd: float | None = None,
    llm_explanation: str = "",
    llm_suggestions: str = "",
    reasoning: str = "",
) -> None:
    """Strategy found — with full reasoning context."""
    emoji = "🚀" if oos_sharpe and oos_sharpe > 0.5 else "⭐" if sharpe > 0 else "❌"

    msg = f"{emoji} *{name[:40]}*\n"
    msg += f"`Train S={sharpe:+.2f} · WR={win_rate:.0%} · PF={profit_factor:.2f}`\n"
    msg += f"`P&L=${total_pnl:+.0f} · Trades={n_trades}`\n"

    if max_dd is not None:
        msg += f"`MaxDD=${abs(max_dd):.0f}`\n"

    if oos_sharpe is not None and oos_trades >= 10:
        of_note = " (overfit?)" if oos_sharpe < sharpe * 0.5 else ""
        msg += f"`OOS S={oos_sharpe:+.2f} · {oos_trades} trades{of_note}`\n"

    if wf_cv is not None and wf_cv < 999:
        cv_note = " ✅ estable" if wf_cv < 1.0 else " ⚠️ inestable"
        msg += f"`WF CV={wf_cv:.2f}{cv_note}`\n"

    # Rules
    try:
        rules_obj = json.loads(rules) if isinstance(rules, str) else rules
        conds = rules_obj.get("entry_conditions", rules_obj)
        rules_str = json.dumps(conds, indent=2)[:400]
        if rules_str:
            msg += f"```\n{rules_str}\n```\n"
    except Exception:
        pass

    # LLM reasoning/analysis
    if analysis:
        msg += f"\n💡 {analysis[:300]}"
    elif llm_explanation:
        msg += f"\n💡 {llm_explanation[:300]}"

    # LLM suggestions
    if llm_suggestions:
        try:
            sug = json.loads(llm_suggestions) if isinstance(llm_suggestions, str) else llm_suggestions
            if isinstance(sug, list) and sug:
                msg += f"\n🔧 {' | '.join(str(s)[:80] for s in sug[:2])}"
        except Exception:
            pass

    await _send(msg)


async def notify_digest(
    elapsed_h: float, n_tested: int, n_positive: int, best_sharpe: float,
    rounds_completed: int, total_cost: float,
    top_strategies: list[dict] | None = None,
    last_round_results: list[dict] | None = None,
) -> None:
    msg = f"⏱️ *{elapsed_h:.1f}h* · `{n_tested} estrat · {rounds_completed} rondas · ${total_cost:.4f}`\n"

    if top_strategies:
        best = top_strategies[0]
        tag = " ✅" if best.get("passes_gates") else ""
        oos = best.get("oos_sharpe")
        oos_s = f"  OOS={oos:+.2f}" if isinstance(oos, (int, float)) and best.get("oos_trades", 0) >= 10 else ""
        wf_cv = best.get("wf_cv", 999)
        cv_s = f" CV={wf_cv:.1f}" if wf_cv < 5 else ""
        of = " ⚠️OVERFIT" if best.get("overfit") else ""
        msg += f"\n🥇 `{best['run_id'][:20]}` S={best['sharpe']:+.2f} WR={best['win_rate']:.0%} PF={best['profit_factor']:.2f}{oos_s}{cv_s}{of}{tag}\n"

    survivors = sum(1 for r in (last_round_results or []) if r.get("n_trades", 0) > 0)
    oos_ok = sum(1 for r in (last_round_results or []) if r.get("oos_trades", 0) >= 10 and not r.get("overfit"))
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
    pass  # Deprecated: too noisy, use notify_round_generation instead


async def notify_round_generation(
    market_ctx: str,
    strategies: list[dict],
    strategy_reasonings: list[str] | None = None,
    new_indicators: list[str] | None = None,
) -> None:
    """Notify what the strategist generated in this round."""
    msg_lines = []
    for i, s in enumerate(strategies[:5], 1):
        name = s.get("name", f"strategy_{i}")[:25]
        conds = s.get("entry_conditions", [])
        cond_str = "; ".join(
            f"{c.get('indicator','?')} {c.get('op','?')} {c.get('value','?')}"
            for c in conds[:2]
        )
        msg_lines.append(f"`{i}. {name}`")
        msg_lines.append(f"   `{cond_str}`")
        if strategy_reasonings and i - 1 < len(strategy_reasonings) and strategy_reasonings[i - 1]:
            reason = strategy_reasonings[i - 1][:150]
            msg_lines.append(f"   💡 {reason}")

    if not msg_lines:
        await _send(f"🧠 *Generación vacía* — el LLM no produjo estrategias válidas")
        return

    header = f"🧠 *Generación ({len(strategies)})*\n📊 `{market_ctx[:80]}`"
    if new_indicators:
        ind_str = " · ".join(f"`{n}`" for n in new_indicators)
        header += f"\n🧪 {ind_str}"

    msg = header + "\n" + "\n".join(msg_lines)
    await _send(msg)


async def notify_round_funnel(
    generated: int, fast_pass: int, full_pass: int, wf_pass: int, oos_pass: int,
    overfit: int, results: list[dict],
) -> None:
    """Notify survival funnel for a round."""
    emoji = "✅" if oos_pass > 0 else "❌" if wf_pass == 0 else "🟡"
    msg = (
        f"{emoji} *Filtro de supervivencia*\n"
        f"`Generadas: {generated}`\n"
        f"`├─ Fast filter (S>0.3, T≥10, DD<20%): {fast_pass}`\n"
        f"`├─ Walk-forward (CV<1.0): {wf_pass}`\n"
        f"`├─ OOS test: {oos_pass}`\n"
        f"`└─ Overfit detectadas: {overfit}`\n"
    )

    if results:
        best = max(results, key=lambda r: r.get("sharpe", -999))
        msg += f"\n🥇 Mejor: `{best['run_id'][:20]}` S={best['sharpe']:+.2f}"
        oos = best.get("oos_sharpe")
        if isinstance(oos, (int, float)) and best.get("oos_trades", 0) >= 10:
            msg += f" OOS={oos:+.2f}"

    await _send(msg)


async def notify_deployed(name: str, sharpe: float, oos_sharpe: float, reasoning: str = "") -> None:
    oos_s = f" OOS={oos_sharpe:+.2f}" if oos_sharpe else ""
    msg = (
        f"🚀 *Estrategia desplegada*\n"
        f"`{name[:35]}` S={sharpe:+.2f}{oos_s}\n"
        f"✅ Activa en paper trading"
    )
    if reasoning:
        msg += f"\n💡 {reasoning[:200]}"
    await _send(msg)


async def notify_regime_change(old_regime: str, new_regime: str) -> None:
    await _send(
        f"🔄 *Cambio de régimen*\n"
        f"`{old_regime} → {new_regime}`"
    )


async def notify_critical_failure(error: str) -> None:
    await _send(
        f"🔴 *Fallo crítico*\n"
        f"`{error[:200]}`"
    )


async def notify_daily_summary(
    equity: float, drawdown: float, n_trades: int,
    new_strategies: int, regime: str, n_active_strategies: int,
) -> None:
    dd_emoji = "🟢" if drawdown < 5 else "🟡" if drawdown < 10 else "🔴"
    msg = (
        f"📊 *Resumen Diario*\n"
        f"`Equity: ${equity:.2f}`  {dd_emoji}`DD: {drawdown:.1f}%`\n"
        f"`Trades: {n_trades}  |  Estrategias nuevas: {new_strategies}`\n"
        f"`Régimen: {regime}  |  Activas: {n_active_strategies}`"
    )
    await _send(msg)


async def notify_daemon_start(interval_h: int) -> None:
    await _send(
        f"🤖 *Daemon iniciado*\n"
        f"Research cada {interval_h}h · Paper trading activo"
    )


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
