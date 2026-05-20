"""Backtesting engine — converts strategy signals to simulated trades.

Vectorized inner loop (numpy) for TP/SL evaluation.
~50-100x faster than the previous bar-by-bar Pandas iteration.

Core function:
    backtest(df, strategy_fn, ...) -> (trades_df, metrics_dict)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.costs import compute_costs, spot_pnl
from src.backtesting.metrics import compute_metrics


def backtest(
    df: pd.DataFrame,
    strategy_fn,
    horizon: int = 12,
    warmup: int = 50,
    cooldown: int = 2,
    size_usdc: float = 500.0,
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

    # Pre-extract numpy arrays for vectorized access
    opens  = df["open"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    ts_arr = df["ts"].values

    trades = []
    last_entry = -cooldown - 1
    n = len(df)

    tp_target_up   = 1 + tp_pct
    sl_target_down = 1 - sl_pct
    tp_target_down = 1 - tp_pct
    sl_target_up   = 1 + sl_pct

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
        if entry_idx + horizon >= n:
            break

        entry_price = opens[entry_idx]
        entry_ts = ts_arr[entry_idx]

        # Vectorized TP/SL over the horizon window
        end = entry_idx + horizon
        window_highs = highs[entry_idx:end]
        window_lows  = lows[entry_idx:end]

        if direction == "BUY":
            tp_mask = window_highs >= entry_price * tp_target_up
            sl_mask = window_lows  <= entry_price * sl_target_down
        else:
            tp_mask = window_lows  <= entry_price * tp_target_down
            sl_mask = window_highs >= entry_price * sl_target_up

        tp_idx = np.where(tp_mask)[0]
        sl_idx = np.where(sl_mask)[0]

        if len(tp_idx) > 0 and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0]):
            bar_offset = tp_idx[0]
            exit_price = entry_price * tp_target_up if direction == "BUY" else entry_price * tp_target_down
            exit_reason = "tp"
        elif len(sl_idx) > 0:
            bar_offset = sl_idx[0]
            exit_price = entry_price * sl_target_down if direction == "BUY" else entry_price * sl_target_up
            exit_reason = "sl"
        else:
            bar_offset = horizon - 1
            exit_price = closes[entry_idx + horizon - 1]
            exit_reason = "horizon"

        exit_idx = entry_idx + bar_offset
        exit_ts = ts_arr[exit_idx]

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
            "horizon_bars": bar_offset,
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
