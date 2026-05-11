"""Orchestrator — full LLM-guided research loop.

Strategist → Backtest → Analyst → Save → Report

Usage:
    OPENROUTER_API_KEY="sk-or-..." python -m src.brain.orchestrator --n 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import pandas as pd

from src.brain.llm_client import LLMClient
from src.brain.strategist import generate_strategies
from src.brain.analyst import analyze_strategy
from src.strategy_engine.schema import validate_strategy
from src.strategy_engine.evaluator import evaluate
from src.backtesting.engine import backtest
from src.research.leaderboard import init_leaderboard, append_result, load_leaderboard
from src.indicators.calculator import calc_all

logger = logging.getLogger(__name__)

# Telegram notifier (initialized in main if env vars are set)
_telegram = None


async def _maybe_notify(method: str, *args, **kwargs) -> None:
    """Call a telegram notification method if the notifier is available."""
    if _telegram is not None:
        try:
            from src.notification.telegram import (
                notify_start, notify_strategy_found, notify_digest,
                notify_hourly_leaderboard, notify_new_indicator, notify_done,
            )
            func = {
                "start": notify_start,
                "strategy": notify_strategy_found,
                "digest": notify_digest,
                "hourly": notify_hourly_leaderboard,
                "new_indicator": notify_new_indicator,
                "done": notify_done,
            }.get(method)
            if func:
                await func(*args, **kwargs)
        except Exception as e:
            logger.warning("telegram notification failed: %s", e)


def _market_context(df: pd.DataFrame) -> str:
    """Build a human-readable market summary for the LLM."""
    regime_counts = df["regime"].value_counts(normalize=True)
    regime_str = ", ".join(
        f"{k}: {v:.0%}" for k, v in sorted(regime_counts.items())
    )

    last_price = df["close"].iloc[-1]
    price_30d_ago = df["close"].iloc[- (30 * 96):].iloc[0] if len(df) > 30 * 96 else df["close"].iloc[0]
    return_30d = (last_price - price_30d_ago) / price_30d_ago * 100

    avg_vol = df["volume"].mean()
    latest_adx = df["adx_14"].iloc[-1]

    return (
        f"Precio BTC: ${last_price:,.0f}  "
        f"Retorno 30d: {return_30d:+.1f}%  "
        f"ADX actual: {latest_adx:.0f}  "
        f"Vol promedio: {avg_vol:.0f} BTC  "
        f"Regímenes: {regime_str}"
    )


async def run_round(
    llm: LLMClient,
    df: pd.DataFrame,
    market_ctx: str,
    n: int = 5,
    days: int = 365,
    timeframe: str = "15m",
    symbol: str = "btcusdt",
) -> list[dict]:
    """One full round: generate → fast filter → full backtest → analyze → save.

    Two-phase backtest:
      Phase 1 (fast):   30d data, ~0.5s/strategy. Descartar obviamente malas.
      Phase 2 (full):   365d data, ~5s/strategy. Solo sobrevivientes.
      Analyst:          Solo sobrevivientes (menos costo LLM).
    """
    past_results = load_leaderboard().tail(15).to_dict("records")
    gen_result = await generate_strategies(llm, market_ctx, past_results, n)
    if isinstance(gen_result, tuple):
        strategies, new_ind_count = gen_result
        if new_ind_count > 0:
            logger.info("orchestrator: %d new indicators registered by LLM", new_ind_count)
            await _maybe_notify("new_indicator", "LLM Auto-Discovery", f"{new_ind_count} nuevos indicadores", "LLM")
    else:
        strategies = gen_result
    if not strategies:
        logger.warning("orchestrator: no strategies generated")
        return []

    logger.info("orchestrator: %d strategies — fast filter first", len(strategies))
    results = []

    # Slice 30d for fast filter
    fast_bars = min(2880, len(df) - 100)  # ~30d de 15m
    df_fast = df.iloc[-fast_bars:].reset_index(drop=True)

    # ── Phase 1: Fast filter (30d) ─────────────────────────────────────
    survivors = []
    for strategy_dict in strategies:
        strat_name = strategy_dict.get("name", "?")

        # Validate
        skip = False
        for cond in strategy_dict.get("entry_conditions", []):
            v = cond.get("value")
            if v is not None and not isinstance(v, (int, float)):
                skip = True
                break
        if skip:
            continue
        try:
            sd = validate_strategy(strategy_dict)
        except Exception:
            continue

        def _eval_fast(d, _sd=sd):
            return evaluate(d, _sd)

        t0 = time.time()
        _, summary = backtest(
            df_fast, _eval_fast,
            horizon=sd.exit.horizon_bars,
            warmup=50, cooldown=2,
            size_usdc=50,
            tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct,
        )
        elapsed = time.time() - t0
        sharpe = summary.get("sharpe", -999)
        n_t = summary.get("n_trades", 0)

        # Survive if Sharpe > -0.1 and at least 3 trades
        if sharpe > -0.1 and n_t >= 3:
            survivors.append((strategy_dict, sd))
            logger.info("  Fast pass: %s S=%.2f T=%d (%.1fs)", strat_name, sharpe, n_t, elapsed)

    logger.info("  Fast filter: %d/%d survived", len(survivors), len(strategies))

    if not survivors:
        logger.warning("orchestrator: no strategies survived fast filter")
        return []

    # ── Phase 2: Full validation (365d) + Analyst ───────────────────────
    for idx, (strategy_dict, sd) in enumerate(survivors):
        strat_name = sd.name

        def _eval_full(d, _sd=sd):
            return evaluate(d, _sd)

        t0 = time.time()
        trades, summary = backtest(
            df, _eval_full,
            horizon=sd.exit.horizon_bars,
            warmup=50, cooldown=2,
            size_usdc=50,
            tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct,
        )
        bt_elapsed = time.time() - t0
        n_trades = summary.get("n_trades", 0)

        # Analyst
        analysis = {}
        if n_trades > 0 and n_trades < 50000:
            try:
                analysis = await analyze_strategy(
                    llm, strategy_dict, summary, market_ctx,
                    days=days, timeframe=timeframe,
                )
            except Exception as e:
                logger.warning("orchestrator: analyst failed: %s", e)

        config_json = json.dumps({
            "timeframe": timeframe, "days": days,
            "symbol": symbol, "source": "brain",
        })

        row = {
            "run_id": strat_name,
            "generation": 0,
            "sharpe": summary.get("sharpe", 0),
            "win_rate": summary.get("win_rate", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "total_pnl": summary.get("total_pnl", 0),
            "max_dd": summary.get("max_dd", 0),
            "n_trades": n_trades,
            "passes_gates": summary.get("passes_gates", False),
            "rules_json": json.dumps(strategy_dict),
            "config_json": config_json,
            "elapsed_bt": round(bt_elapsed, 2),
            "llm_model": "deepseek/deepseek-chat",
            "llm_explanation": analysis.get("analysis", ""),
            "llm_suggestions": json.dumps(analysis.get("suggestions", [])),
            "llm_confidence": analysis.get("confidence", 0),
        }
        results.append(row)
        append_result(row, strategy_dict)

        wr = row["win_rate"]
        pf = row["profit_factor"]
        pnl = row["total_pnl"]
        s = row["sharpe"]
        tag = " ✅" if row["passes_gates"] else ""
        print(f"  [{idx+1}/{len(survivors)}] {strat_name[:30]:30s}  "
              f"S={s:+.2f}  WR={wr:.0%}  PF={pf:.2f}  PnL=${pnl:+.0f}  "
              f"T={n_trades}{tag}", flush=True)

    return results


async def main():
    parser = argparse.ArgumentParser(description="LLM-guided research loop")
    parser.add_argument("--n", type=int, default=5, help="Strategies per round")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--resample", type=str, default="15m", help="Timeframe")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument("--digest-minutes", type=int, default=15, help="Telegram digest interval (min)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter key")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or ""
    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY. Set env var or pass --api-key")
        return

    # Load data
    print(f"\nLoading {args.days}d {args.resample} data...", flush=True)
    data_path = Path("data/raw") / "binance" / args.symbol / args.resample / "data.parquet"
    df = pd.read_parquet(data_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    df = df[df["ts"] >= cutoff]
    df = calc_all(df)
    print(f"  Data: {len(df):,} rows", flush=True)

    market_ctx = _market_context(df)
    print(f"  Market: {market_ctx}", flush=True)

    init_leaderboard()
    llm = LLMClient(api_key=api_key)

    # Telegram setup
    global _telegram
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        _telegram = True
        await _maybe_notify("start", args.rounds, args.n, args.days, args.resample)
        start_time = time.time()
    else:
        _telegram = None

    all_results = []
    last_digest_time = time.time()

    try:
        for round_num in range(args.rounds):
            round_label = f"Round {round_num + 1}/{args.rounds}"
            print(f"\n{'='*60}", flush=True)
            print(f"  {round_label}", flush=True)
            print(f"{'='*60}", flush=True)

            results = await run_round(
                llm, df, market_ctx,
                n=args.n, days=args.days,
                timeframe=args.resample, symbol=args.symbol,
            )
            all_results.extend(results)

            if results:
                best = max(results, key=lambda r: r["sharpe"])
                print(f"\n  Best: S={best['sharpe']:+.2f}  "
                      f"PF={best['profit_factor']:.2f}  "
                      f"PnL=${best['total_pnl']:+.0f}", flush=True)

                # Notify on positive findings
                for r in results:
                    s = r["sharpe"]
                    if s > 0 and r["n_trades"] >= 30:
                        try:
                            rules = json.loads(r["rules_json"])
                        except Exception:
                            rules = {}
                        await _maybe_notify("strategy",
                            r["run_id"], s, r["win_rate"],
                            r["profit_factor"], r["total_pnl"],
                            r["n_trades"],
                            r.get("llm_explanation", ""),
                            json.dumps(rules.get("entry_conditions", []), indent=2),
                        )

                # Digest configurable (default 15 min)
                if time.time() - last_digest_time > args.digest_minutes * 60:
                    elapsed_h = (time.time() - start_time) / 3600
                    positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30]
                    best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0}
                    top3 = sorted(all_results, key=lambda r: r["sharpe"], reverse=True)[:3] if all_results else []
                    await _maybe_notify("digest",
                        elapsed_h, len(all_results), len(positive),
                        best_all["sharpe"], round_num + 1, llm.total_cost,
                        top3,
                    )
                    last_digest_time = time.time()

                # Hourly leaderboard
                _elapsed_h = (time.time() - start_time) / 3600
                _last_hourly = getattr(_maybe_notify, "_last_hourly", 0)
                _latest_hour = int(_elapsed_h)
                if _latest_hour > _last_hourly and _latest_hour >= 1:
                    _maybe_notify._last_hourly = _latest_hour
                    await _maybe_notify("hourly", _elapsed_h, all_results)

    finally:
        await llm.close()
        print(f"\nTotal LLM cost: ${llm.total_cost:.4f}", flush=True)
        print("Audit log: data/parquet/brain/audit.jsonl", flush=True)

        # Final digest
        if _telegram and all_results:
            elapsed_h = (time.time() - start_time) / 3600 if _telegram else 0
            positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30]
            best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0, "run_id": ""}
            await _maybe_notify("done",
                elapsed_h, len(all_results), len(positive),
                best_all["sharpe"], best_all.get("run_id", ""), llm.total_cost,
            )


if __name__ == "__main__":
    asyncio.run(main())
