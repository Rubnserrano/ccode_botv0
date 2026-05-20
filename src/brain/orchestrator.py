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
from src.brain.strategist import generate_hypotheses
from src.brain.analyst import analyze_strategy
from src.brain.critic import heuristic_critique
from src.brain.concept_stats import reward_simple, record_outcome, expected_sharpe
from src.brain.memory import fingerprint, is_redundant, record as memory_record, get_seen_set
from src.brain.survival import log_survival, load_survival
from src.strategy_engine.schema import validate_strategy
from src.strategy_engine.search_engine import expand_hypothesis
from src.strategy_engine.evaluator import evaluate
from src.backtesting.engine import backtest
from src.backtesting.baseline import compute_baseline
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
                notify_hourly_leaderboard, notify_new_indicator,
                notify_round_generation, notify_round_funnel, notify_done,
            )
            func = {
                "start": notify_start,
                "strategy": notify_strategy_found,
                "digest": notify_digest,
                "hourly": notify_hourly_leaderboard,
                "new_indicator": notify_new_indicator,
                "round_generation": notify_round_generation,
                "round_funnel": notify_round_funnel,
                "done": notify_done,
            }.get(method)
            if func:
                await func(*args, **kwargs)
        except Exception as e:
            logger.warning("telegram notification failed: %s", e)


def _market_context(df: pd.DataFrame) -> str:
    regime_counts = df["regime"].value_counts(normalize=True)
    regime_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(regime_counts.items()))
    last = df.iloc[-1]
    last_price = last["close"]
    lookback = min(30 * 96, len(df) - 100)
    recent = df.iloc[-lookback:]
    price_30d_ago = recent.iloc[0]["close"]
    return_30d = (last_price - price_30d_ago) / price_30d_ago * 100
    avg_vol = df["volume"].mean()
    latest_adx = last["adx_14"]
    latest_rsi = last["rsi_14"]
    latest_regime = last["regime"]
    regime_name = {0: "ranging", 1: "trending_up", 2: "trending_down", 3: "volatile"}
    regime_label = regime_name.get(int(latest_regime), "unknown")
    # Volatility stats
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()
    range_pct = (recent_high - recent_low) / recent_low * 100
    avg_atr = recent["atr_14"].mean()
    return (
        f"Precio BTC: ${last_price:,.0f}  Retorno 30d: {return_30d:+.1f}%  "
        f"ADX: {latest_adx:.0f}  RSI: {latest_rsi:.0f}  "
        f"Régimen actual: {regime_label} ({latest_regime})  "
        f"Rango 30d: {range_pct:.0f}%  ATR medio: {avg_atr:.0f}  "
        f"Vol promedio: {avg_vol:.0f} BTC  "
        f"Distribución regímenes: {regime_str}"
    )


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
    round_feedback: str = "",
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

    # ─── Step 1: Generate hypotheses (LLM) with retry ────────────────
    hypotheses = await generate_hypotheses(llm, market_ctx, past_results, n, round_feedback)
    if not hypotheses:
        logger.warning("orchestrator: no hypotheses, retrying...")
        hypotheses = await generate_hypotheses(llm, market_ctx, past_results, n, "Primera ronda. Explora libremente.")
    if not hypotheses:
        logger.warning("orchestrator: no hypotheses generated after retry")
        return []

    # Notify hypotheses generation
    hypo_reasonings = [h.get("reasoning", "")[:150] for h in hypotheses]
    await _maybe_notify("round_generation", market_ctx, hypotheses, hypo_reasonings, [])

    # ─── Step 2: Expand hypotheses → concrete strategies ────────────
    all_strategies: list[dict] = []

    # Always expand the LLM's chosen family
    llm_families = set()
    for h in hypotheses:
        family = h.get("family", "")
        params = h.get("params", {})
        llm_families.add(family)
        expanded = expand_hypothesis(family, params, n_variants=15)
        all_strategies.extend(expanded)

    # Add baseline strategies from OTHER families not chosen by LLM
    from src.strategy_engine.hypothesis import list_families
    for family in list_families():
        if family not in llm_families:
            expanded = expand_hypothesis(family, n_variants=5)
            all_strategies.extend(expanded)

    if not all_strategies:
        logger.warning("orchestrator: search engine returned no strategies")
        return []

    logger.info("orchestrator: %d strategies from %d hypotheses", len(all_strategies), len(hypotheses))

    # ─── Step 3: Register dynamic indicators from search engine ─────
    from src.strategy_engine.generic_calculator import register_dynamic_indicator
    for s in all_strategies:
        ni = s.pop("new_indicator", None) or s.pop("_new_indicator", None)
        if ni and isinstance(ni, dict):
            try:
                register_dynamic_indicator(ni["name"], ni["formula"],
                                          ni.get("params"), ni.get("description", "search-engine"))
                logger.info("search: registered indicator '%s' = %s", ni["name"], ni["formula"][:60])
            except Exception as e:
                logger.warning("search: failed to register indicator '%s': %s", ni.get("name", "?"), e)

    # ─── Step 4: Critic (heuristic filter) ───────────────────────────
    past_strategies = load_leaderboard().tail(50).to_dict("records") if not load_leaderboard().empty else []
    filtered_strategies = []
    for s in all_strategies:
        critique = heuristic_critique(s, past_strategies)
        if critique["passes"]:
            filtered_strategies.append(s)
        else:
            logger.info("  Critic reject: %s — %s", s.get("name", "?"), "; ".join(critique["reasons"][:2]))

    if not filtered_strategies:
        logger.warning("orchestrator: all strategies rejected by critic")
        return []

    strategies = filtered_strategies
    logger.info("orchestrator: %d strategies after critic — fast filter on TRAIN", len(strategies))
    results = []

    # Fast filter on last 60d of TRAIN (no data leakage)
    fast_bars = min(1440, len(df_train) - 100)
    df_fast = df_train.iloc[-fast_bars:].reset_index(drop=True)

    # Track rejection reasons
    reject_reasons = {"no_trades": 0, "neg_sharpe": 0, "high_dd": 0, "low_trades": 0, "validation_error": 0, "below_baseline": 0}

    baseline = compute_baseline(
        df_fast,
        n_trials=20,
        entry_freq=0.02,
        horizon=12,
        warmup=50,
        cooldown=2,
        size_usdc=50,
        tp_pct=0.005,
        sl_pct=0.005,
    )
    baseline_sharpe = baseline["sharpe_mean"] + baseline["sharpe_std"]
    logger.info("  Baseline random Sharpe: %.3f (mean=%.3f std=%.3f)",
                baseline_sharpe, baseline["sharpe_mean"], baseline["sharpe_std"])

    survivors = []
    for strategy_dict in strategies:
        strat_name = strategy_dict.get("name", "?")
        try:
            sd = validate_strategy(strategy_dict)
        except Exception as e:
            reject_reasons["validation_error"] += 1
            logger.debug("  Fast skip %s: validation error: %s", strat_name, e)
            continue

        def _eval_fast(d, _sd=sd):
            return evaluate(d, _sd)

        _, summary = backtest(df_fast, _eval_fast,
            horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
            size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
        sharpe = summary.get("sharpe", -999)
        n_t = summary.get("n_trades", 0)
        max_dd = abs(summary.get("max_dd", 0))
        total_turnover = 50.0 * n_t if n_t > 0 else 1
        dd_pct = max_dd / total_turnover if total_turnover > 0 else 0

        if n_t < 3:
            reject_reasons["no_trades" if n_t == 0 else "low_trades"] += 1
            continue
        if sharpe < baseline_sharpe:
            reject_reasons["below_baseline"] += 1
            continue
        if dd_pct >= 0.30:
            reject_reasons["high_dd"] += 1
            continue
        survivors.append((strategy_dict, sd))
        logger.info("  Fast pass: %s S=%.2f T=%d DD=%.1f%% (baseline=%.2f)",
                    strat_name, sharpe, n_t, dd_pct * 100, baseline_sharpe)

    logger.info("  Fast filter: %d/%d survived  Reasons: %s",
                len(survivors), len(strategies), reject_reasons)
    if not survivors:
        log_survival(0, len(strategies), 0, 0, 0, 0)
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

        # Walk-forward variance penalty (A.4)
        wf_sharpe_std = 0.0
        wf_cv = 999.0
        if wf_results and len(wf_results) >= 3:
            wf_sharpes = [r.sharpe for r in wf_results if r.n_trades > 0]
            if wf_sharpes:
                wf_mean = float(pd.Series(wf_sharpes).mean())
                wf_sharpe_std = float(pd.Series(wf_sharpes).std())
                wf_cv = wf_sharpe_std / wf_mean if wf_mean > 0 else 999.0
                if wf_cv > 1.0:
                    wf_passes = False

        # TEST validation (hold-out, never seen before)
        sharpe_test = 0.0
        n_test_trades = 0
        max_dd_test = 0.0
        if len(df_test) > 100:
            trades_test, summary_test = backtest(df_test, _eval_full,
                horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
                size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
            sharpe_test = summary_test.get("sharpe", 0)
            n_test_trades = summary_test.get("n_trades", 0)
            max_dd_test = abs(summary_test.get("max_dd", 0))

        # Overfit detection: relative + absolute signal floor (A.2)
        overfit = False
        if sharpe_train < 0.5:
            overfit = True  # no hay señal suficiente, cualquier cosa es ruido
        elif n_test_trades >= 10:
            if sharpe_test < sharpe_train * 0.5:
                overfit = True  # perdió >50% del rendimiento en test

        # Max DD gate OOS (A.3)
        total_pnl_test = abs(summary_test.get("total_pnl", 0)) if n_test_trades >= 10 else 0
        dd_ratio_test = max_dd_test / total_pnl_test if total_pnl_test > 0 else 0
        if dd_ratio_test > 0.15:
            overfit = True

        # Analyst (only if passed OOS)
        analysis = {}
        if n_trades > 0 and n_trades < 50000 and not overfit:
            try:
                analysis = await analyze_strategy(
                    llm, strategy_dict, summary, market_ctx,
                    days=days, timeframe=timeframe,
                    wf_sharpe=wf_sharpe,
                    wf_sharpe_std=wf_sharpe_std,
                    wf_cv=wf_cv,
                    oos_sharpe=sharpe_test,
                    oos_trades=n_test_trades,
                )
            except Exception as e:
                logger.warning("orchestrator: analyst failed: %s", e)

        config_json = json.dumps({
            "timeframe": timeframe, "days": days,
            "symbol": symbol, "source": "brain",
        })

        archetype = strategy_dict.get("archetype", "unknown")

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
            "wf_sharpe_std": round(wf_sharpe_std, 4),
            "wf_cv": round(wf_cv, 4),
            "wf_passes": wf_passes,
            "oos_sharpe": round(sharpe_test, 4),
            "oos_trades": n_test_trades,
            "overfit": overfit,
            "archetype": archetype,
        }

        # Record concept stats
        _regime = int(df_train["regime"].mode().iloc[0]) if "regime" in df_train.columns else -1
        _novelty = 1.0  # Phase 1: no penalty; Phase 2: from embeddings
        _reward = reward_simple(
            sharpe_oos=sharpe_test, sharpe_train=sharpe_train,
            wf_sharpe=wf_sharpe, wf_cv=wf_cv,
            max_dd=summary.get("max_dd", 0), total_pnl=summary.get("total_pnl", 0),
            n_trades=n_trades, regime=_regime, novelty_score=_novelty, overfit=overfit,
        )
        record_outcome(
            archetype=archetype, regime=_regime, reward=_reward,
            passed_fast=True, passed_oos=not overfit and n_test_trades >= 10,
            sharpe_oos=sharpe_test, sharpe_train=sharpe_train, n_trades=n_trades,
        )
        results.append(row)
        append_result(row, strategy_dict)

        tag = " ✅" if row["passes_gates"] else ""
        oos_tag = f" OOS={sharpe_test:+.2f}" if n_test_trades >= 10 else ""
        wf_tag = " WF_OK" if wf_passes else ""
        of_tag = " ⚠️ OVERFIT" if overfit else ""
        cv_tag = f" CV={wf_cv:.1f}" if wf_cv < 999 else ""
        print(f"  [{idx+1}/{len(survivors)}] {strat_name[:25]:25s}  "
              f"S={sharpe_train:+.2f}  WR={row['win_rate']:.0%}  "
              f"PF={row['profit_factor']:.2f}  T={n_trades}{tag}{oos_tag}{wf_tag}{cv_tag}{of_tag}", flush=True)

    # Survival log + notification
    wf_pass_count = sum(1 for r in results if r.get("wf_passes", False))
    oos_pass_count = sum(1 for r in results if r.get("oos_trades", 0) >= 10 and not r.get("overfit", False))
    overfit_count = sum(1 for r in results if r.get("overfit", False))
    log_survival(0, len(strategies), len(survivors), wf_pass_count, oos_pass_count, overfit_count)
    await _maybe_notify("round_funnel", len(strategies), len(survivors), 0, wf_pass_count, oos_pass_count, overfit_count, results)

    return results


async def run_research(
    n_strategies: int = 10,
    days: int = 365,
    resample: str = "15m",
    rounds: int = 1,
    symbol: str = "btcusdt",
    api_key: str | None = None,
) -> list[dict]:
    """Run the full research loop programmatically (without argparse).

    Returns the list of result dicts (same as leaderboard rows).
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY") or ""
    if not api_key:
        raise ValueError("No OPENROUTER_API_KEY")

    print(f"\nLoading {days}d {resample} from TS Store...", flush=True)
    from src.ts_store import read as ts_read
    asset_id = f"market:binance:{symbol}"
    df = ts_read(asset_id, frequency=resample, limit=50000)
    if df.empty:
        # Fallback: try raw with limit
        df = ts_read(asset_id, frequency="raw", limit=50000)
    if df.empty:
        raise FileNotFoundError(f"No data for {asset_id} @ {resample} in TS Store")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff]

    # Merge derivatives data (funding rate, OI, taker ratio, long/short ratio)
    from pathlib import Path as _Path
    _deriv_dir = _Path("data/ts/derivatives/aligned") / resample
    if _deriv_dir.exists():
        for _df_file in sorted(_deriv_dir.glob("*.parquet")):
            try:
                _deriv_df = pd.read_parquet(_df_file)
                _deriv_df["ts"] = pd.to_datetime(_deriv_df["ts"], utc=True)
                df = df.merge(_deriv_df, on="ts", how="left")
                logger.info("orchestrator: merged derivatives '%s'", _df_file.stem)
            except Exception as e:
                logger.warning("orchestrator: failed to merge derivatives '%s': %s", _df_file.stem, e)

    df = calc_all(df)

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
        await _maybe_notify("start", rounds, n_strategies, days, resample, market_ctx)
        start_time = time.time()
    else:
        _telegram = None

    all_results = []
    last_digest_time = time.time()
    best_from_previous = []

    try:
        for round_num in range(rounds):
            round_label = f"Round {round_num + 1}/{rounds}"
            print(f"\n{'='*60}", flush=True)
            print(f"  {round_label}", flush=True)
            print(f"{'='*60}", flush=True)

            prev_new_indicators = sum(1 for r in all_results if r.get("run_id", "").startswith("🧪"))
            prev_best = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0}

            # Build survival stats for feedback
            survival_records = load_survival()
            recent = survival_records[-5:] if survival_records else []
            n_generated = sum(r.get("generated", 0) for r in recent) if recent else 0
            n_fast = sum(r.get("fast_filter_pass", 0) for r in recent) if recent else 0
            n_oos = sum(r.get("oos_pass", 0) for r in recent) if recent else 0
            survival_rate = f"{100 * n_oos / n_generated:.0f}%" if n_generated > 0 else "N/A"

            round_feedback = (
                f"• Ronda anterior: {len(all_results)} estrategias probadas\n"
                f"• Tasa de supervivencia (generadas → OOS): {survival_rate}\n"
                f"• Indicadores nuevos creados: {prev_new_indicators}\n"
                f"• Mejor Sharpe train: {prev_best['sharpe']:+.2f}\n"
                f"• Mejor Sharpe OOS: {prev_best.get('oos_sharpe', 0):+.2f}\n"
                f"• Objetivo: Sharpe sostenible. Más allá del train, busca consistencia en OOS."
            )

            results = await run_round(
                llm, df_train, df_test, market_ctx,
                n=n_strategies, days=days,
                timeframe=resample, symbol=symbol,
                best_from_previous=best_from_previous,
                round_feedback=round_feedback,
            )
            all_results.extend(results)

            if results:
                sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
                best_from_previous = sorted_results[:3]
                best = sorted_results[0]
                print(f"\n  Best: S={best['sharpe']:+.2f}  OOS={best.get('oos_sharpe', 0):+.2f}  "
                      f"PF={best['profit_factor']:.2f}", flush=True)

                for r in results:
                    if r.get("n_trades", 0) < 10:
                        continue
                    if r.get("overfit", False):
                        continue
                    try:
                        rules = json.loads(r["rules_json"])
                    except Exception:
                        rules = {}
                    await _maybe_notify("strategy",
                        r["run_id"], r["sharpe"], r["win_rate"],
                        r["profit_factor"], r["total_pnl"],
                        r["n_trades"], r.get("llm_explanation", ""),
                        json.dumps(rules.get("entry_conditions", []), indent=2),
                        oos_sharpe=r.get("oos_sharpe"),
                        oos_trades=r.get("oos_trades", 0),
                        wf_sharpe=r.get("wf_sharpe"),
                        wf_cv=r.get("wf_cv"),
                        max_dd=r.get("max_dd"),
                        llm_suggestions=r.get("llm_suggestions", ""),
                    )

                if time.time() - last_digest_time > 15 * 60:
                    _elapsed = (time.time() - start_time) / 3600
                    positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30 and not r.get("overfit", False)]
                    best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0}
                    top3 = sorted(all_results, key=lambda r: r["sharpe"], reverse=True)[:3] if all_results else []
                    await _maybe_notify("digest", _elapsed, len(all_results), len(positive),
                        best_all["sharpe"], round_num + 1, llm.total_cost, top3, results)
                    last_digest_time = time.time()

    finally:
        await llm.close()
        print(f"\nTotal LLM cost: ${llm.total_cost:.4f}", flush=True)
        if _telegram:
            _elapsed = (time.time() - start_time) / 3600
            positive = [r for r in all_results if r["sharpe"] > 0 and r["n_trades"] >= 30 and not r.get("overfit", False)]
            best_all = max(all_results, key=lambda r: r["sharpe"]) if all_results else {"sharpe": 0, "run_id": ""}
            await _maybe_notify("done", _elapsed, len(all_results), len(positive),
                best_all["sharpe"], best_all.get("run_id", ""), llm.total_cost, all_results)

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="LLM-guided research loop with train/test split")
    parser.add_argument("--n", type=int, default=10, help="Strategies per round")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--resample", type=str, default="15m", help="Timeframe")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter key")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    await run_research(
        n_strategies=args.n,
        days=args.days,
        resample=args.resample,
        rounds=args.rounds,
        symbol=args.symbol,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
