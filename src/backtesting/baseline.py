"""Baseline random strategy — establishes a floor for comparison.

Generates random entry signals with configurable frequency and evaluates
them using the same backtest engine.  This lets us ask: "Is this strategy
actually better than flipping a coin?"
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtesting.engine import backtest

logger = logging.getLogger(__name__)


def random_strategy_fn(df: pd.DataFrame, entry_freq: float = 0.02, seed: int = 42):
    """Return a closure that generates random BUY signals.

    Parameters
    ----------
    entry_freq : float
        Probability of a signal on any given bar (after warmup).
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    def _fn(df_inner: pd.DataFrame) -> pd.Series:
        n = len(df_inner)
        signals = pd.Series(0, index=df_inner.index, dtype=int)
        warmup = 50
        entries = rng.random(n - warmup) < entry_freq
        signals.iloc[warmup:] = entries.astype(int)
        return signals

    return _fn


def compute_baseline(
    df: pd.DataFrame,
    n_trials: int = 100,
    entry_freq: float = 0.02,
    horizon: int = 12,
    warmup: int = 50,
    cooldown: int = 2,
    size_usdc: float = 50.0,
    tp_pct: float = 0.005,
    sl_pct: float = 0.005,
) -> dict:
    """Run multiple random-strategy backtests to establish baseline metrics.

    Returns
    -------
    dict with keys: sharpe_mean, sharpe_std, sharpe_p95, n_trades_mean,
                     win_rate_mean, max_dd_mean, profit_factor_mean
    """
    sharpes = []
    trades_counts = []
    win_rates = []
    max_dds = []
    profit_factors = []

    for i in range(n_trials):
        fn = random_strategy_fn(df, entry_freq=entry_freq, seed=i)
        _, summary = backtest(
            df, fn,
            horizon=horizon, warmup=warmup, cooldown=cooldown,
            size_usdc=size_usdc, tp_pct=tp_pct, sl_pct=sl_pct,
        )
        if summary.get("n_trades", 0) > 0:
            sharpes.append(summary["sharpe"])
            trades_counts.append(summary["n_trades"])
            win_rates.append(summary["win_rate"])
            max_dds.append(abs(summary["max_dd"]))
            profit_factors.append(summary.get("profit_factor", 0))

    if not sharpes:
        logger.warning("baseline: no random strategies produced trades")
        return {
            "sharpe_mean": 0.0, "sharpe_std": 1.0, "sharpe_p95": 0.5,
            "n_trades_mean": 0, "win_rate_mean": 0.5,
            "max_dd_mean": 0.0, "profit_factor_mean": 1.0,
        }

    sharpe_arr = np.array(sharpes)
    return {
        "sharpe_mean": float(np.mean(sharpe_arr)),
        "sharpe_std": float(np.std(sharpe_arr)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "n_trades_mean": float(np.mean(trades_counts)),
        "win_rate_mean": float(np.mean(win_rates)),
        "max_dd_mean": float(np.mean(max_dds)),
        "profit_factor_mean": float(np.mean(profit_factors)),
    }