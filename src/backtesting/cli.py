"""CLI for running backtests on historical data.

Usage:
    python -m src.backtesting.cli --strategy rsi_mean_reversion --days 365
    python -m src.backtesting.cli --strategy all --days 862
    python -m src.backtesting.cli --list
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


def _list_strategies():
    print("Available strategies:")
    for name in STRATEGY_REGISTRY:
        print(f"  {name}")


def _run_one(
    strategy_name: str,
    strategy_fn,
    symbol: str,
    days: int | None,
    horizon: int,
    warmup: int,
    cooldown: int,
    size_usdc: float,
    exchange: str,
) -> tuple[pd.DataFrame, dict, str]:
    run_id = uuid.uuid4().hex[:12]
    print(f"\n[{run_id}] Running {strategy_name} on {symbol}...")

    df = read(exchange, symbol)
    if df.empty:
        print(f"  No data for {symbol}")
        return pd.DataFrame(), {"error": "no data"}, run_id

    # Trim to last N days if specified
    if days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = df[df["ts"] >= cutoff]

    print(f"  Data: {len(df):,} rows ({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")

    t0 = time.time()
    df = calc_all(df)
    print(f"  Indicators: {time.time() - t0:.1f}s")

    config = {
        "strategy": strategy_name,
        "symbol": symbol,
        "horizon": horizon,
        "warmup": warmup,
        "cooldown": cooldown,
        "size_usdc": size_usdc,
        "days": days,
    }

    t0 = time.time()
    trades_df, summary = backtest(
        df, strategy_fn,
        horizon=horizon, warmup=warmup, cooldown=cooldown,
        size_usdc=size_usdc,
        run_id=run_id,
    )
    elapsed = time.time() - t0
    print(f"  Backtest: {elapsed:.1f}s")

    if not trades_df.empty:
        n_trades = len(trades_df)
        wins = (trades_df["outcome"] == "WIN").sum()
        print(f"  Trades: {n_trades}  Wins: {wins}  WinRate: {summary.get('win_rate', 0):.1%}")
        print(f"  Sharpe: {summary.get('sharpe', 0):.2f}  ProfitFactor: {summary.get('profit_factor', 0):.2f}")
        print(f"  PnL: ${summary.get('total_pnl', 0):+.2f}  MaxDD: ${summary.get('max_dd', 0):+.2f}")
        print(f"  Gates: {'✅ PASS' if summary.get('passes_gates') else '❌ FAIL'}")
    else:
        print("  No trades generated")

    return trades_df, summary, run_id


def main():
    parser = argparse.ArgumentParser(description="Run backtests on historical data")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'")
    parser.add_argument("--symbol", default="btcusdt", help="Trading pair")
    parser.add_argument("--days", type=int, default=None, help="Days of history (default: all)")
    parser.add_argument("--horizon", type=int, default=12, help="Hold bars (default: 12)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup bars (default: 50)")
    parser.add_argument("--cooldown", type=int, default=2, help="Cooldown bars (default: 2)")
    parser.add_argument("--size", type=float, default=50.0, help="Position size in USDC")
    parser.add_argument("--list", action="store_true", help="List strategies and exit")
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

    all_summaries = []
    total_start = time.time()

    for sym in symbols:
        for name, fn in strategies:
            try:
                trades_df, summary, run_id = _run_one(
                    name, fn, sym, args.days,
                    args.horizon, args.warmup, args.cooldown, args.size,
                    exchange,
                )
                if not trades_df.empty and not args.no_save:
                    config = {
                        "strategy": name, "symbol": sym, "horizon": args.horizon,
                        "warmup": args.warmup, "cooldown": args.cooldown,
                        "size_usdc": args.size, "days": args.days,
                    }
                    path = save_results(run_id, trades_df, summary, name, config)
                    print(f"  Saved: {path}")

                all_summaries.append(summary)
            except Exception as e:
                print(f"  ERROR {name} on {sym}: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed:.1f}s")

    if all_summaries:
        print("\n── Results ──")
        for s in all_summaries:
            if s.get("error"):
                print(f"  {s.get('run_id','?')}: {s['error']}")
            else:
                print(f"  {s['run_id']}: {s.get('n_trades',0):3d} trades  "
                      f"Sharpe={s.get('sharpe',0):+.2f}  "
                      f"PnL=${s.get('total_pnl',0):+.2f}  "
                      f"{'✅' if s.get('passes_gates') else '❌'}")


if __name__ == "__main__":
    main()
