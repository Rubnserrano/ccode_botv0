"""Orchestrator — full LLM-guided research loop with train/test split.

Usage:
    OPENROUTER_API_KEY="sk-or-..." python -m src.brain.orchestrator --n 10
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
from src.brain.memory import fingerprint, is_redundant, record as memory_record, get_seen_set
from src.strategy_engine.schema import validate_strategy
from src.strategy_engine.evaluator import evaluate
from src.backtesting.engine import backtest
from src.backtesting.walk_forward import walk_forward, walk_forward_summary, WalkForwardConfig
from src.research.leaderboard import init_leaderboard, append_result, load_leaderboard
from src.indicators.calculator import calc_all

logger = logging.getLogger(__name__)

_telegram = None


async def _maybe_notify(method: str, *args, **kwargs) -> None:
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
    regime_counts = df["regime"].value_counts(normalize=True)
    regime_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(regime_counts.items()))
    last_price = df["close"].iloc[-1]
    price_30d_ago = df["close"].iloc[- (30 * 96):].iloc[0] if len(df) > 30 * 96 else df["close"].iloc[0]
    return_30d = (last_price - price_30d_ago) / price_30d_ago * 100
    avg_vol = df["volume"].mean()
    latest_adx = df["adx_14"].iloc[-1]
    return (f"Precio BTC: ${last_price:,.0f}  Retorno 30d: {return_30d:+.1f}%  "
            f"ADX actual: {latest_adx:.0f}  Vol promedio: {avg_vol:.0f} BTC  Regímenes: {regime_str}")


async def run_round(
    llm: LLMClient,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    market_ctx: str,
    n: int = 10,
    days: int = 365,
    timeframe: str = "15m",
    symbol: str = "btcusdt",
    best_from_previous: list[dict] | None = None,
) -> list[dict]:
    """One round with train/test separation.

    1. Generate + dedup strategies
    2. Fast filter on last 30d of TRAIN
    3. Full backtest on TRAIN
    4. Walk-forward on TRAIN (OOS validation)
    5. Final validation on TEST (hold-out)
    6. Analyst + save
    """
    past_results = load_leaderboard().tail(20).to_dict("records")

    # Inject best strategies from previous round as evolution seeds
    if best_from_previous:
        for b in best_from_previous:
            past_results.insert(0, b)

    strategies, new_ind_count = await generate_strategies(llm, market_ctx, past_results, n)
    if new_ind_count > 0:
        logger.info("orchestrator: %d new indicators registered by LLM", new_ind_count)
        await _maybe_notify("new_indicator", "LLM", f"{new_ind_count} nuevos indicadores", "LLM")
    if not strategies:
        logger.warning("orchestrator: no strategies generated")
        return []

    # Dedup
    seen_fps = get_seen_set()
    unique = []
    for s in strategies:
        fp = fingerprint(s)
        if is_redundant(fp, seen_fps):
            logger.info("orchestrator: skipped duplicate '%s'", s.get("name", "?"))
            continue
        seen_fps.add(fp)
        memory_record(fp, s)
        unique.append(s)

    if not unique:
        logger.warning("orchestrator: all strategies are duplicates")
        return []

    strategies = unique
    logger.info("orchestrator: %d unique — fast filter on TRAIN", len(strategies))
    results = []

    # Fast filter on last 30d of TRAIN (no data leakage)
    fast_bars = min(2880, len(df_train) - 100)
    df_fast = df_train.iloc[-fast_bars:].reset_index(drop=True)

    survivors = []
    for strategy_dict in strategies:
        strat_name = strategy_dict.get("name", "?")
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

        _, summary = backtest(df_fast, _eval_fast,
            horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
            size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
        sharpe = summary.get("sharpe", -999)
        n_t = summary.get("n_trades", 0)
        if sharpe > -0.05 and n_t >= 3:
            survivors.append((strategy_dict, sd))
            logger.info("  Fast pass: %s S=%.2f T=%d", strat_name, sharpe, n_t)

    logger.info("  Fast filter: %d/%d survived", len(survivors), len(strategies))
    if not survivors:
        return []

    # Phase 2: Full backtest on TRAIN + walk-forward
    wf_cfg = WalkForwardConfig(n_splits=4, horizon=12, warmup=50, cooldown=2, size_usdc=50)

    for idx, (strategy_dict, sd) in enumerate(survivors):
        strat_name = sd.name

        def _eval_full(d, _sd=sd):
            return evaluate(d, _sd)

        # Backtest on TRAIN
        t0 = time.time()
        trades, summary = backtest(df_train, _eval_full,
            horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
            size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
        bt_elapsed = time.time() - t0
        n_trades = summary.get("n_trades", 0)
        sharpe_train = summary.get("sharpe", 0)

        # Walk-forward on TRAIN (OOS validation)
        wf_results = walk_forward(df_train, _eval_full, wf_cfg)
        wf_summary = walk_forward_summary(wf_results)
        wf_passes = wf_summary.get("passes", False)
        wf_sharpe = wf_summary.get("oos_sharpe_mean", 0)

        # TEST validation (hold-out, never seen before)
        sharpe_test = 0.0
        n_test_trades = 0
        if len(df_test) > 100:
            trades_test, summary_test = backtest(df_test, _eval_full,
                horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
                size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
            sharpe_test = summary_test.get("sharpe", 0)
            n_test_trades = summary_test.get("n_trades", 0)

        # Skip if TEST Sharpe is much worse than TRAIN (overfitting)
        overfit = sharpe_test < sharpe_train - 0.3 and n_test_trades >= 10

        # Analyst (only if passed OOS)
        analysis = {}
        if n_trades > 0 and n_trades < 50000 and not overfit:
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
            "run_id": strat_name, "generation": 0,
            "sharpe": sharpe_train, "win_rate": summary.get("win_rate", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "total_pnl": summary.get("total_pnl", 0),
            "max_dd": summary.get("max_dd", 0),
            "n_trades": n_trades, "passes_gates": summary.get("passes_gates", False),
            "rules_json": json.dumps(strategy_dict), "config_json": config_json,
            "elapsed_bt": round(bt_elapsed, 2),
            "llm_model": "deepseek/deepseek-chat",
            "llm_explanation": analysis.get("analysis", ""),
            "llm_suggestions": json.dumps(analysis.get("suggestions", [])),
            "llm_confidence": analysis.get("confidence", 0),
            # OOS metrics
            "wf_sharpe": round(wf_sharpe, 4),
            "wf_passes": wf_passes,
            "oos_sharpe": round(sharpe_test, 4),
            "oos_trades": n_test_trades,
            "overfit": overfit,
        }
        results.append(row)
        append_result(row, strategy_dict)

        tag = " ✅" if row["passes_gates"] else ""
        oos_tag = f" OOS={sharpe_test:+.2f}" if n_test_trades >= 10 else ""
        wf_tag = " WF_OK" if wf_passes else ""
        of_tag = " ⚠️ OVERFIT" if overfit else ""
        print(f"  [{idx+1}/{len(survivors)}] {strat_name[:25]:25s}  "
              f"S={sharpe_train:+.2f}  WR={row['win_rate']:.0%}  "
              f"PF={row['profit_factor']:.2f}  T={n_trades}{tag}{oos_tag}{wf_tag}{of_tag}", flush=True)

    return results


async def main():
    parser = argparse.ArgumentParser(description="LLM-guided research loop with train/test split")
    parser.add_argument("--n", type=int, default=10, help="Strategies per round")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--resample", type=str, default="15m", help="Timeframe")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds")
    parser.add_argument("--digest-minutes", type=int, default=15, help="Telegram digest interval")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter key")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or ""
    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY")
        return

    # Load + split data
    print(f"\nLoading {args.days}d {args.resample}...", flush=True)
    data_path = Path("data/raw") / "binance" / args.symbol / args.resample / "data.parquet"
    df = pd.read_parquet(data_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    df = df[df["ts"] >= cutoff]
    df = calc_all(df)

    # Train/test split (80/20 temporal)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} rows  Test: {len(df_test):,} rows", flush=True)

    market_ctx = _market_context(df_train)
    print(f"  Market: {market_ctx}", flush=True)

    init_leaderboard()
    llm = LLMClient(api_key=api_key)

    global _telegram
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        _telegram = True
        await _maybe_notify("start", args.rounds, args.n, args.days, args.resample, market_ctx)
        start_time = time.time()
    else:
        _telegram = None

    all_results = []
    last_digest_time = time.time()
    best_from_previous = []

    try:
        for round_num in range(args.rounds):
            round_label = f"Round {round_num + 1}/{args.rounds}"
            print(f"\n{'='*60}", flush=True)
            print(f"  {round_label}", flush=True)
            print(f"{'='*60}", flush=True)

            results = await run_round(
                llm, df_train, df_test, market_ctx,
                n=args.n, days=args.days,
                timeframe=args.resample, symbol=args.symbol,
                best_from_previous=best_from_previous,
            )
            all_results.extend(results)

            # Update best_from_previous for next round (evolution)
            if results:
                sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
                best_from_previous = sorted_results[:3]
                best = sorted_results[0]
                print(f"\n  Best: S={best['sharpe']:+.2f}  OOS={best.get('oos_sharpe', 0):+.2f}  "
                      f"PF={best['profit_factor']:.2f}", flush=True)

                for r in results[:1]:
                    s = r["sharpe"]
                    if s > 0 and r["n_trades"] >= 30 and not r.get("overfit", False):
                        try:
                            rules = json.loads(r["rules_json"])
                        except Exception:
                            rules = {}
                        await _maybe_notify("strategy",
                            r["run_id"], s, r["win_rate"],
                            r["profit_factor"], r["total_pnl"],
                            r["n_trades"], r.get("llm_explanation", ""),
                            json.dumps(rules.get("entry_conditions", []), indent=2),
                        )

                if time.time() - last_digest_time > args.digest_minutes * 60:
                    _elapsed = (time.time() - start_time) / 3600
                    positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30 and not r.get("overfit", False)]
                    best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0}
                    top3 = sorted(all_results, key=lambda r: r["sharpe"], reverse=True)[:3] if all_results else []
                    await _maybe_notify("digest", _elapsed, len(all_results), len(positive),
                        best_all["sharpe"], round_num + 1, llm.total_cost, top3, results)
                    last_digest_time = time.time()

                _elapsed_h = (time.time() - start_time) / 3600
                _lh = getattr(_maybe_notify, "_last_hourly", 0)
                if int(_elapsed_h) > _lh and int(_elapsed_h) >= 1:
                    _maybe_notify._last_hourly = int(_elapsed_h)
                    await _maybe_notify("hourly", _elapsed_h, all_results)

    finally:
        await llm.close()
        print(f"\nTotal LLM cost: ${llm.total_cost:.4f}", flush=True)
        if _telegram:
            _elapsed = (time.time() - start_time) / 3600
            positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30 and not r.get("overfit", False)]
            best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0, "run_id": ""}
            await _maybe_notify("done", _elapsed, len(all_results), len(positive),
                best_all["sharpe"], best_all.get("run_id", ""), llm.total_cost, all_results)


if __name__ == "__main__":
    asyncio.run(main())
