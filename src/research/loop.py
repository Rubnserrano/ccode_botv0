"""Automated research loop — generates, backtests, and records strategies.

Saves results progressively so partial progress survives timeout/crash.

Usage:
    python -m src.research.loop --n 100 --days 365 --resample 15m
    python -m src.research.loop --n 200 --generations 3 --pop-size 50 --days 90
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

from src.research.leaderboard import init_leaderboard, append_result, load_leaderboard

_SAVED_FLAG = False
_INTERRUPTED = False


def _handle_sigint(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n\n⏹  Interrupted — saving results...", flush=True)


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ─── Strategy space ──────────────────────────────────────────────────────────

INDICATORS = ["rsi", "regime", "volume", "adx", "atr", "macd", "ema"]
OPS = ["lt", "gt"]
VALUES_BY_INDICATOR = {
    "rsi":     list(range(20, 80, 5)),     # 20, 25, 30, ... 75
    "regime":  [0],                          # solo RANGING
    "volume":  [round(v, 1) for v in [x * 0.1 for x in range(5, 50, 5)]],
    "adx":     list(range(15, 50, 5)),
    "atr":     [round(v, 1) for v in [x * 0.1 for x in range(10, 100, 10)]],
    "macd":    [round(v, 1) for v in range(-10, 11, 2)],
    "ema":     list(range(5, 100, 10)),
}
EXITS = [
    {"tp_pct": 0.04, "sl_pct": 0.015, "horizon_bars": 48},
    {"tp_pct": 0.05, "sl_pct": 0.02,  "horizon_bars": 48},
    {"tp_pct": 0.03, "sl_pct": 0.01,  "horizon_bars": 24},
    {"tp_pct": 0.04, "sl_pct": 0.01,  "horizon_bars": 24},
    {"tp_pct": 0.06, "sl_pct": 0.02,  "horizon_bars": 48},
]


def _random_strategy(name: str) -> dict:
    n_conditions = random.randint(1, 3)
    conditions = []
    indicators_chosen = random.sample(INDICATORS, min(n_conditions, len(INDICATORS)))
    for ind in indicators_chosen:
        conditions.append({
            "indicator": ind,
            "op": random.choice(OPS),
            "value": random.choice(VALUES_BY_INDICATOR[ind]),
        })
    exit_rules = random.choice(EXITS)
    return {
        "name": name,
        "entry_conditions": conditions,
        "exit": exit_rules,
    }


def _mutate(parent: dict, name: str) -> dict:
    """Mutate a strategy by tweaking one condition or exit rule."""
    child = json.loads(json.dumps(parent))  # deep copy
    child["name"] = name
    if child["entry_conditions"] and random.random() < 0.7:
        cond = random.choice(child["entry_conditions"])
        if random.random() < 0.5:
            cond["value"] = random.choice(VALUES_BY_INDICATOR.get(cond["indicator"], [30]))
        else:
            cond["op"] = random.choice(OPS)
    else:
        child["exit"] = random.choice(EXITS)
    return child


def _crossover(p1: dict, p2: dict, name: str) -> dict:
    """Crossover: mix conditions from two parents."""
    child = {"name": name, "entry_conditions": [], "exit": random.choice(EXITS)}
    c1 = p1["entry_conditions"]
    c2 = p2["entry_conditions"]
    child["entry_conditions"] = random.sample(c1, min(1, len(c1))) + random.sample(c2, min(1, len(c2)))
    return child


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_research(
    n: int = 100,
    days: int = 365,
    resample: str = "15m",
    generations: int = 1,
    pop_size: int = 50,
    elite_ratio: float = 0.2,
    exchange: str = "binance",
    symbol: str = "btcusdt",
) -> None:
    """Run automated research loop.

    Results are saved progressively to data/parquet/research/leaderboard.parquet
    and best strategies to data/strategies/research_*.json
    """
    global _INTERRUPTED

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading {days}d {resample} data...", flush=True)
    from pathlib import Path as P
    import pandas as pd
    from src.indicators.calculator import calc_all

    data_path = P("data/raw") / exchange / symbol / resample / "data.parquet"
    df = pd.read_parquet(data_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff]
    df = calc_all(df)
    print(f"  Data: {len(df):,} rows ({days}d {resample})", flush=True)

    # ── Init output ──────────────────────────────────────────────────────
    from src.strategy_engine.schema import validate_strategy
    from src.strategy_engine.evaluator import evaluate
    from src.strategy_engine.store import save as save_strategy
    from src.backtesting.engine import backtest
    from src.research.leaderboard import init_leaderboard, append_result

    init_leaderboard()
    total_start = time.time()
    all_results = []

    # ── Generation loop ──────────────────────────────────────────────────
    population = []
    for gen in range(generations):
        if _INTERRUPTED:
            break

        gen_label = f"gen{gen}"
        print(f"\n{'='*60}", flush=True)
        print(f"Generation {gen + 1}/{generations}", flush=True)
        print(f"{'='*60}", flush=True)

        # Generate/mutate population
        if gen == 0:
            for i in range(pop_size):
                population.append(_random_strategy(f"auto_{gen_label}_{i}"))
        else:
            # Elite selection
            all_results.sort(key=lambda r: r.get("sharpe", -999), reverse=True)
            n_elite = max(2, int(pop_size * elite_ratio))
            elite = population[:n_elite]

            new_pop = list(elite)  # keep elites
            for ind in elite:
                if len(new_pop) >= pop_size:
                    break
                new_pop.append(_mutate(ind, f"auto_{gen_label}_{len(new_pop)}"))

            while len(new_pop) < pop_size:
                if len(elite) >= 2:
                    p1, p2 = random.sample(elite, 2)
                    new_pop.append(_crossover(p1, p2, f"auto_{gen_label}_{len(new_pop)}"))
                else:
                    new_pop.append(_random_strategy(f"auto_{gen_label}_{len(new_pop)}"))

            population = new_pop[:pop_size]

        # ── Evaluate population ───────────────────────────────────────────
        gen_results = []
        gen_trades = 0
        gen_start = time.time()

        for i, strategy_dict in enumerate(population):
            if _INTERRUPTED:
                break

            try:
                sd = validate_strategy(strategy_dict)
            except Exception as e:
                continue

            strat_name = sd.name

            def eval_fn(d, _sd=sd):
                return evaluate(d, _sd)

            t0 = time.time()
            trades, summary = backtest(
                df, eval_fn,
                horizon=sd.exit.horizon_bars,
                warmup=50, cooldown=2,
                size_usdc=50,
                tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct,
            )
            bt = time.time() - t0

            n_trades = summary.get("n_trades", 0)
            gen_trades += n_trades

            row = {
                "run_id": strat_name,
                "generation": gen,
                "sharpe": summary.get("sharpe", 0),
                "win_rate": summary.get("win_rate", 0),
                "profit_factor": summary.get("profit_factor", 0),
                "total_pnl": summary.get("total_pnl", 0),
                "max_dd": summary.get("max_dd", 0),
                "n_trades": n_trades,
                "passes_gates": summary.get("passes_gates", False),
                "rules_json": json.dumps(strategy_dict),
                "elapsed_bt": round(bt, 2),
            }
            gen_results.append(row)
            all_results.append(row)

            # Save each result progressively
            append_result(row, strategy_dict)
            gen_trades += 1

            # Save best strategy every 20
            if row["sharpe"] > 0 and row["n_trades"] >= 30:
                try:
                    save_strategy(sd)
                except Exception:
                    pass

            # Progress
            elapsed = time.time() - gen_start
            avg = elapsed / (i + 1)
            remaining = (len(population) - i - 1) * avg
            if (i + 1) % 10 == 0 or i == len(population) - 1:
                print(f"  [{strat_name[:30]:30s}] "
                      f"{i+1}/{len(population)}  "
                      f"S={row['sharpe']:+.2f}  "
                      f"WR={row['win_rate']:.0%}  "
                      f"PF={row['profit_factor']:.2f}  "
                      f"PnL=${row['total_pnl']:+.0f}  "
                      f"T={row['n_trades']}  "
                      f"{bt:.1f}s  "
                      f"eta={remaining:.0f}s  ",
                      flush=True)

        gen_elapsed = time.time() - gen_start
        top = sorted(gen_results, key=lambda r: r["sharpe"], reverse=True)[:3]
        print(f"\n  Gen {gen+1}: {len(gen_results)} eval'd in {gen_elapsed:.0f}s", flush=True)
        if top:
            print(f"  Top: S={top[0]['sharpe']:+.2f} | S={top[1]['sharpe']:+.2f} | S={top[2]['sharpe']:+.2f}", flush=True)

    # ── Final report ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    all_results.sort(key=lambda r: r["sharpe"], reverse=True)
    positive = [r for r in all_results if r["sharpe"] > 0]

    print(f"\n{'='*60}", flush=True)
    print(f"RESEARCH COMPLETE  |  {len(all_results)} strategies  |  {total_elapsed:.0f}s", flush=True)
    print(f"Sharpe > 0: {len(positive)}", flush=True)
    print(f"{'='*60}", flush=True)

    if all_results:
        print(f"\nTop 10 by Sharpe:", flush=True)
        for r in all_results[:10]:
            print(f"  S={r['sharpe']:+.2f}  WR={r['win_rate']:.0%}  "
                  f"PF={r['profit_factor']:.2f}  PnL=${r['total_pnl']:+.0f}  "
                  f"T={r['n_trades']}  {r['run_id']}", flush=True)

    print(f"\nFull leaderboard: data/parquet/research/leaderboard.parquet", flush=True)
    print(f"Top strategies:   data/strategies/research_*.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Automated research loop")
    parser.add_argument("--n", type=int, default=100, help="Strategies to test per generation")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--resample", type=str, default="15m", help="Resample interval")
    parser.add_argument("--generations", type=int, default=1, help="Evolution generations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population per generation")
    parser.add_argument("--symbol", type=str, default="btcusdt", help="Trading pair")
    args = parser.parse_args()

    run_research(
        n=args.n,
        days=args.days,
        resample=args.resample,
        generations=args.generations,
        pop_size=args.pop_size,
        symbol=args.symbol,
    )


if __name__ == "__main__":
    main()
