"""CLI for backtesting, walk-forward, and evolution.

Usage:
    python -m src.backtesting.cli --strategy rsi_mean_reversion --days 365
    python -m src.backtesting.cli --strategy all --days 862
    python -m src.backtesting.cli --list
    python -m src.backtesting.cli --walk-forward --strategy rsi_mean_reversion --days 862
    python -m src.backtesting.cli --evolve --strategy rsi_mean_reversion --days 862 --generations 5
"""
from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.store import read
from src.indicators.calculator import calc_all
from src.backtesting.strategies import STRATEGY_REGISTRY
from src.backtesting.engine import backtest, save_results
from src.backtesting.walk_forward import walk_forward, walk_forward_summary, WalkForwardConfig
from src.backtesting.evolution import evolve, EvolveConfig, make_strategy
from src.backtesting.genealogy import add_node, GenealogyNode


def _list_strategies():
    print("Available strategies:")
    for name in sorted(STRATEGY_REGISTRY):
        s = STRATEGY_REGISTRY[name]
        print(f"  {name:25s}  {s.description}")


def _load_data(symbol: str, days: int | None, exchange: str = "binance") -> pd.DataFrame:
    # Try features first (precomputed indicators)
    from src.features.store import read as f_read, available_range as f_range
    f_start, f_end = f_range(symbol)
    if f_start is not None:
        cutoff = None
        if days:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = f_read(symbol, start=cutoff)
        if not df.empty:
            print(f"  Features: {len(df):,} rows ({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")
            return df

    # Fallback: raw + calc_all
    df = read(exchange, symbol)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["ts"] >= cutoff]
    print(f"  Data: {len(df):,} rows ({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")
    t0 = time.time()
    df = calc_all(df)
    print(f"  Indicators: {time.time() - t0:.1f}s")
    return df


def _run_backtest(strategy_name, strategy_fn, df, args) -> tuple[pd.DataFrame, dict, str]:
    run_id = uuid.uuid4().hex[:12]
    print(f"\n[{run_id}] Running {strategy_name}...")

    config = {
        "strategy": strategy_name,
        "symbol": args.symbol,
        "horizon": args.horizon,
        "warmup": args.warmup,
        "cooldown": args.cooldown,
        "size_usdc": args.size,
        "days": args.days,
    }

    t0 = time.time()
    trades_df, summary = backtest(
        df, strategy_fn,
        horizon=args.horizon, warmup=args.warmup, cooldown=args.cooldown,
        size_usdc=args.size,
        run_id=run_id,
    )
    elapsed = time.time() - t0
    print(f"  Backtest: {elapsed:.1f}s")

    if not trades_df.empty:
        wins = (trades_df["outcome"] == "WIN").sum()
        print(f"  Trades: {len(trades_df)}  Wins: {wins}  WinRate: {summary.get('win_rate', 0):.1%}")
        print(f"  Sharpe: {summary.get('sharpe', 0):.2f}  PnL: ${summary.get('total_pnl', 0):+.2f}")
        print(f"  Gates: {'✅ PASS' if summary.get('passes_gates') else '❌ FAIL'}")
    else:
        print("  No trades")

    if not trades_df.empty and not args.no_save:
        path = save_results(run_id, trades_df, summary, strategy_name, config)
        print(f"  Saved: {path}")

    return trades_df, summary, run_id


def _run_walk_forward(strategy_name, strategy_fn, df, args):
    print(f"\n[wf] Walk-forward for {strategy_name}...")
    cfg = WalkForwardConfig(
        n_splits=args.wf_splits,
        horizon=args.horizon,
        warmup=args.warmup,
        cooldown=args.cooldown,
        size_usdc=args.size,
    )
    t0 = time.time()
    results = walk_forward(df, strategy_fn, cfg)
    elapsed = time.time() - t0
    print(f"  Walk-forward: {elapsed:.1f}s")

    summary = walk_forward_summary(results)
    print(f"  Folds: {summary.get('n_folds', 0)}  Trades: {summary.get('n_trades_total', 0)}")
    print(f"  OOS Sharpe: mean={summary.get('oos_sharpe_mean', 0):.2f}  min={summary.get('oos_sharpe_min', 0):.2f}  max={summary.get('oos_sharpe_max', 0):.2f}")
    print(f"  PnL: ${summary.get('oos_pnl_total', 0):+.2f}")
    print(f"  Passes: {'✅' if summary.get('passes') else '❌'}")

    for r in results:
        print(f"    Fold {r.fold}: Sharpe={r.sharpe:.2f}  Trades={r.n_trades}  PnL=${r.total_pnl:+.2f}")


def _run_evolution(strategy_name, df, args):
    print(f"\n[evolve] Evolution for {strategy_name}...")
    cfg = EvolveConfig(
        n_generations=args.generations,
        pop_size=args.pop_size,
        horizon=args.horizon,
        warmup=args.warmup,
        cooldown=args.cooldown,
        size_usdc=args.size,
    )
    t0 = time.time()
    pop = evolve(df, strategy_name, cfg)
    elapsed = time.time() - t0

    print(f"  Evolution: {elapsed:.1f}s  Generations: {args.generations}  Pop: {args.pop_size}")

    if pop:
        print(f"\n  Top 5 individuals:")
        for i, ind in enumerate(pop[:5]):
            print(f"    {i + 1}. {ind.strategy.name}  fitness={ind.fitness:.4f}  "
                  f"Sharpe={ind.sharpe:.2f}  trades={ind.n_trades}")
            print(f"       params={ind.strategy.params}")

            node = GenealogyNode(
                strategy_id=ind.strategy.name,
                base_name=strategy_name,
                params=ind.strategy.params,
                fitness=ind.fitness,
                sharpe=ind.sharpe,
                n_trades=ind.n_trades,
            )
            add_node(node)
            print(f"       saved to genealogy")

        # Save best
        best = pop[0]
        best_strat = make_strategy(strategy_name, best.strategy.params, name=f"{strategy_name}_best")
        if best_strat:
            print(f"\n  Best strategy params: {best.strategy.params}")
            print(f"  To use: make_strategy('{strategy_name}', {best.strategy.params})")


def main():
    parser = argparse.ArgumentParser(description="Backtesting, walk-forward, and evolution")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'")
    parser.add_argument("--symbol", default="btcusdt", help="Trading pair")
    parser.add_argument("--days", type=int, default=None, help="Days of history (default: all)")
    parser.add_argument("--horizon", type=int, default=12, help="Hold bars (default: 12)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup bars (default: 50)")
    parser.add_argument("--cooldown", type=int, default=2, help="Cooldown bars (default: 2)")
    parser.add_argument("--size", type=float, default=50.0, help="Position size in USDC")
    parser.add_argument("--list", action="store_true", help="List strategies and exit")

    # Walk-forward
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--wf-splits", type=int, default=5, help="Walk-forward folds (default: 5)")

    # Evolution
    parser.add_argument("--evolve", action="store_true", help="Run evolution")
    parser.add_argument("--generations", type=int, default=5, help="Evolution generations (default: 5)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")

    # Misc
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    args = parser.parse_args()

    if args.list:
        _list_strategies()
        return

    exchange = "binance"
    symbols = [s.strip().lower() for s in args.symbol.split(",")]

    strategies = (
        list(STRATEGY_REGISTRY.items())
        if args.strategy == "all"
        else [(args.strategy, STRATEGY_REGISTRY[args.strategy])]
    )

    total_start = time.time()

    for sym in symbols:
        df = _load_data(sym, args.days, exchange)

        for name, fn in strategies:
            try:
                if args.evolve:
                    _run_evolution(name, df, args)
                elif args.walk_forward:
                    _run_walk_forward(name, fn, df, args)
                else:
                    _run_backtest(name, fn, df, args)
            except Exception as e:
                print(f"  ERROR {name} on {sym}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nTotal time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
