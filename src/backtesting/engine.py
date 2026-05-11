"""Backtesting engine — converts strategy signals to simulated trades.

Core function:
    backtest(df, strategy_fn, ...) -> (trades_df, metrics_dict)

Signals come from strategy functions, the engine simulates entries/exits
with realistic costs and returns structured results.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.backtesting.costs import compute_costs, spot_pnl
from src.backtesting.metrics import compute_metrics


def backtest(
    df: pd.DataFrame,
    strategy_fn,
    horizon: int = 12,
    warmup: int = 50,
    cooldown: int = 2,
    size_usdc: float = 50.0,
    taker_fee: float = 0.001,
    spread_bps: float = 2.0,
    tp_pct: float = 0.005,
    sl_pct: float = 0.005,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a single backtest for one strategy on one symbol.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicators (output of calc_all). Must have:
        ts, open, high, low, close, volume + indicator columns.
    strategy_fn : callable or Strategy
        Function (df) -> pd.Series or ``Strategy`` object.
    horizon : int
        Number of bars to hold each position.
    warmup : int
        Skip first N bars (indicator burn-in).
    cooldown : int
        Minimum bars between consecutive trades.
    size_usdc : float
        Position size per trade.
    taker_fee : float
        Per-side taker fee fraction (0.001 = 0.1%).
    spread_bps : float
        Spread in basis points.
    tp_pct : float
        Take-profit as fraction of entry (0.005 = 0.5%).
    sl_pct : float
        Stop-loss as fraction of entry (0.005 = 0.5%).
    run_id : str | None
        Unique run identifier. Auto-generated if None.

    Returns
    -------
    trades_df : pd.DataFrame
        One row per simulated trade.
    summary : dict
        Metrics + metadata.
    """
    if df.empty:
        return pd.DataFrame(), {"error": "empty DataFrame"}

    run_id = run_id or uuid.uuid4().hex[:12]
    if hasattr(strategy_fn, "generate"):
        signals = strategy_fn.generate(df)
    else:
        signals = strategy_fn(df)

    trades = []
    last_entry = -cooldown - 1
    n = len(df)

    for i in range(warmup, n):
        if i + 1 >= n:
            break
        sig = signals.iloc[i]
        if sig == 0:
            continue
        if i - last_entry < cooldown:
            continue

        direction = "BUY" if sig == 1 else "SELL"
        entry_idx = i + 1
        entry_price = df.iloc[entry_idx]["open"]

        if entry_idx + horizon >= n:
            break

        entry_ts = df.iloc[entry_idx]["ts"]

        exit_price = None
        exit_reason = "horizon"
        exit_idx = entry_idx + horizon
        exit_ts = df.iloc[exit_idx]["ts"]

        for bar_idx in range(entry_idx, entry_idx + horizon):
            bar = df.iloc[bar_idx]
            hi, lo = bar["high"], bar["low"]

            if direction == "BUY":
                tp_hit = hi >= entry_price * (1 + tp_pct)
                sl_hit = lo <= entry_price * (1 - sl_pct)
            else:
                tp_hit = lo <= entry_price * (1 - tp_pct)
                sl_hit = hi >= entry_price * (1 + sl_pct)

            if tp_hit:
                exit_price = entry_price * (1 + tp_pct) if direction == "BUY" else entry_price * (1 - tp_pct)
                exit_reason = "tp"
                exit_idx = bar_idx
                exit_ts = df.iloc[bar_idx]["ts"]
                break
            if sl_hit:
                exit_price = entry_price * (1 - sl_pct) if direction == "BUY" else entry_price * (1 + sl_pct)
                exit_reason = "sl"
                exit_idx = bar_idx
                exit_ts = df.iloc[bar_idx]["ts"]
                break

        if exit_price is None:
            exit_price = df.iloc[exit_idx]["close"]

        gross = spot_pnl(direction, entry_price, exit_price, size_usdc)
        costs = compute_costs(size_usdc, entry_price, taker_fee, spread_bps)
        net = gross - costs.total
        outcome = "WIN" if net > 0 else "LOSS"

        trades.append({
            "run_id": run_id,
            "ts": entry_ts,
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "horizon_bars": exit_idx - entry_idx,
            "gross_pnl": round(gross, 2),
            "fees": round(costs.entry_fee + costs.exit_fee, 4),
            "spread_cost": round(costs.spread_cost, 4),
            "slippage": round(costs.slippage, 4),
            "net_pnl": round(net, 2),
            "outcome": outcome,
        })

        last_entry = i

    trades_df = pd.DataFrame(trades)
    total_turnover = size_usdc * len(trades)

    if trades_df.empty:
        summary = {"run_id": run_id, "n_trades": 0, "error": "no trades generated"}
    else:
        pnls = trades_df["net_pnl"].tolist()
        metrics = compute_metrics(pnls, total_turnover)
        summary = {
            "run_id": run_id,
            "n_trades": metrics.n_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate": metrics.win_rate,
            "total_pnl": metrics.total_pnl,
            "avg_pnl": metrics.avg_pnl,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "max_dd": metrics.max_dd,
            "profit_factor": metrics.profit_factor,
            "p_value": metrics.p_value,
            "passes_gates": metrics.passes_gates,
        }

    return trades_df, summary


def save_results(run_id: str, trades_df: pd.DataFrame, summary: dict, strategy_name: str, config: dict) -> Path:
    """Save backtest results to data/parquet/backtests/{run_id}/."""
    out_dir = Path("data/parquet/backtests") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_parquet(out_dir / "trades.parquet", compression="snappy")

    summary_df = pd.DataFrame([summary])
    summary_df["strategy"] = strategy_name
    summary_df["config"] = json.dumps(config)
    summary_df["run_at"] = datetime.now(timezone.utc).isoformat()
    summary_df.to_parquet(out_dir / "summary.parquet", compression="snappy")

    metadata = {
        "run_id": run_id,
        "strategy": strategy_name,
        "config": config,
        "summary": {k: v for k, v in summary.items() if k != "run_id"},
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str))

    return out_dir
