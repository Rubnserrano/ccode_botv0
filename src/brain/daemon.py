"""Daemon — permanent research loop for the cognitive system.

Runs in a separate process. Every N hours:
  1. Run research (N strategies)
  2. Find best new strategy from leaderboard
  3. Apply strict filter: Sharpe OOS > 2.0 AND win_rate > 60% AND max_dd < 10%
  4. If passes filter → auto-deploy to paper trading + Telegram notify
  5. If found but below filter → Telegram notify (review)
  6. Detect regime change
  7. Check paper equity/drawdown
  8. Every 24h: Telegram daily summary

Usage:
    OPENROUTER_API_KEY="sk-or-..." .venv/bin/python3 -m src.brain.daemon
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.research.leaderboard import load_leaderboard
from src.indicators.calculator import calc_all
from src.brain.orchestrator import run_research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_INTERVAL_HOURS = 6
_STRICT_FILTER = {
    "min_sharpe_oos": 2.0,
    "min_win_rate": 0.60,
    "max_drawdown": 0.10,
}
_LAST_REGIME: str | None = None


# ─── Notifications ──────────────────────────────────────────────────────

async def _notify(method: str, *args, **kwargs) -> None:
    """Send Telegram notification if configured."""
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        return
    try:
        from src.notification.telegram import (
            notify_deployed, notify_regime_change,
            notify_critical_failure, notify_daily_summary,
            notify_daemon_start,
        )
        func = {
            "deployed": notify_deployed,
            "regime_change": notify_regime_change,
            "critical_failure": notify_critical_failure,
            "daily_summary": notify_daily_summary,
            "daemon_start": notify_daemon_start,
        }.get(method)
        if func:
            await func(*args, **kwargs)
    except Exception as e:
        logger.warning("daemon: telegram failed: %s", e)


# ─── Helpers ────────────────────────────────────────────────────────────

def _detect_current_regime() -> str:
    """Detect current market regime from latest data."""
    try:
        from src.ts_store import read as ts_read
        df = ts_read("market:binance:btcusdt", frequency="15m", limit=500)
        if df.empty:
            return "unknown"
        df = calc_all(df)
        latest = df.iloc[-1]
        regime = latest.get("regime", -1)
        regime_map = {0: "ranging", 1: "trending_up", 2: "trending_down", 3: "volatile"}
        return regime_map.get(int(regime), "unknown")
    except Exception as e:
        logger.warning("daemon: regime detection failed: %s", e)
        return "unknown"


async def _check_and_deploy(lb: pd.DataFrame) -> None:
    """Check leaderboard for strategies that pass strict filter → auto-deploy."""
    valid = lb[lb["n_trades"] > 0].copy()

    if valid.empty:
        return

    # Look for strategies that pass strict filter
    candidates = []
    for _, row in valid.iterrows():
        try:
            sharpe_oos = float(row.get("oos_sharpe", 0))
            if row.get("oos_trades", 0) < 10:
                continue
            win_rate = float(row.get("win_rate", 0))
            max_dd = abs(float(row.get("max_dd", 0)))
            total_pnl = abs(float(row.get("total_pnl", 0)))
            dd_ratio = max_dd / total_pnl if total_pnl > 0 else 1.0

            if (sharpe_oos >= _STRICT_FILTER["min_sharpe_oos"]
                    and win_rate >= _STRICT_FILTER["min_win_rate"]
                    and dd_ratio <= _STRICT_FILTER["max_drawdown"]
                    and not row.get("overfit", False)):
                candidates.append((sharpe_oos, row))
        except Exception:
            continue

    if not candidates:
        # Notify about best strategy even if below strict filter
        best = valid.loc[valid["sharpe"].idxmax()]
        if float(best.get("sharpe", 0)) > 1.0 and float(best.get("n_trades", 0)) >= 30:
            oos = best.get("oos_sharpe", 0)
            await _notify("deployed", best["run_id"], float(best["sharpe"]), float(oos) if isinstance(oos, (int, float)) else 0)
        return

    # Auto-deploy best candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best = candidates[0]

    rules_raw = best.get("rules_json", "")
    if not isinstance(rules_raw, str) or not rules_raw:
        return

    try:
        strategy_def = json.loads(rules_raw)
    except Exception:
        return

    # Save as active strategy
    active_path = Path("data/strategies/active.json")
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(strategy_def, indent=2))

    name = best["run_id"]
    sharpe = float(best["sharpe"])
    oos = best.get("oos_sharpe", 0)
    oos_v = float(oos) if isinstance(oos, (int, float)) else 0
    logger.info("daemon: auto-deployed '%s'  S=%.2f  OOS=%.2f", name, sharpe, oos_v)
    await _notify("deployed", name, sharpe, oos_v)


async def _report_daily_summary() -> None:
    """Send daily Telegram summary."""
    try:
        # Paper state
        paper_path = Path("data/paper_state.json")
        paper = {}
        if paper_path.exists():
            paper = json.loads(paper_path.read_text())

        # Leaderboard
        lb = load_leaderboard()
        recent = lb[lb["n_trades"] > 0] if not lb.empty else pd.DataFrame()
        new_today = len(recent) if not recent.empty else 0

        # Regime
        regime = _detect_current_regime()

        # Active strategy
        active_path = Path("data/strategies/active.json")
        n_active = 1 if active_path.exists() else 0

        equity = float(paper.get("equity", 0))
        initial = float(paper.get("initial_capital", 10000))
        drawdown = max(0, (initial - equity) / initial * 100) if initial > 0 else 0
        n_trades = int(paper.get("n_trades", 0))

        await _notify("daily_summary", equity, drawdown, n_trades, new_today, regime, n_active)
    except Exception as e:
        logger.warning("daemon: daily summary failed: %s", e)


async def _check_regime() -> None:
    """Check for regime change and notify."""
    global _LAST_REGIME
    current = _detect_current_regime()
    if _LAST_REGIME is not None and current != _LAST_REGIME and current != "unknown":
        logger.info("daemon: regime change  %s → %s", _LAST_REGIME, current)
        await _notify("regime_change", _LAST_REGIME, current)
    _LAST_REGIME = current


async def _check_paper_health() -> None:
    """Check paper trading health (drawdown)."""
    try:
        paper_path = Path("data/paper_state.json")
        if not paper_path.exists():
            return
        paper = json.loads(paper_path.read_text())
        equity = float(paper.get("equity", 0))
        initial = float(paper.get("initial_capital", 10000))
        if initial > 0:
            dd_pct = (initial - equity) / initial * 100
            if dd_pct > 10:
                await _notify("critical_failure", f"Paper drawdown > 10% ({dd_pct:.1f}%)")
    except Exception as e:
        logger.warning("daemon: paper health check failed: %s", e)


# ─── Main loop ──────────────────────────────────────────────────────────

async def main_loop() -> None:
    global _LAST_REGIME

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("daemon: no OPENROUTER_API_KEY")
        return

    interval_seconds = _INTERVAL_HOURS * 3600
    last_daily_summary = 0.0
    cycle = 0

    await _notify("daemon_start", _INTERVAL_HOURS)
    _LAST_REGIME = _detect_current_regime()

    logger.info("daemon: started — research every %dh", _INTERVAL_HOURS)

    while True:
        cycle += 1
        logger.info("daemon: cycle %d — starting research", cycle)

        try:
            results = await run_research(
                n_strategies=10,
                days=365,
                resample="15m",
                rounds=2,
                symbol="btcusdt",
                api_key=api_key,
            )
            logger.info("daemon: cycle %d — %d strategies generated, checking...", cycle, len(results))
        except Exception as e:
            logger.exception("daemon: research failed in cycle %d", cycle)
            await _notify("critical_failure", f"Research failed: {str(e)[:200]}")
            await asyncio.sleep(interval_seconds)
            continue

        # Check leaderboard for deploy candidates
        try:
            lb = load_leaderboard()
            await _check_and_deploy(lb)
        except Exception as e:
            logger.warning("daemon: deploy check failed: %s", e)

        # Regime check
        try:
            await _check_regime()
        except Exception as e:
            logger.warning("daemon: regime check failed: %s", e)

        # Paper health
        try:
            await _check_paper_health()
        except Exception as e:
            logger.warning("daemon: paper health check failed: %s", e)

        # Daily summary
        if time.time() - last_daily_summary > 24 * 3600:
            try:
                await _report_daily_summary()
                last_daily_summary = time.time()
            except Exception as e:
                logger.warning("daemon: daily summary failed: %s", e)

        logger.info("daemon: cycle %d complete — sleeping %dh", cycle, _INTERVAL_HOURS)
        await asyncio.sleep(interval_seconds)


def main() -> None:
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
