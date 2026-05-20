"""Walk-forward analysis — out-of-sample validation.

Splits data into sequential folds. For each fold:
  Train on fold[:train_ratio], backtest on fold[train_ratio:].
Aggregates out-of-sample metrics across all folds.

This prevents overfitting by testing on data the strategy
has never seen during parameter selection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.backtesting.engine import backtest as run_backtest
from src.backtesting.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    n_splits: int = 4
    train_ratio: float = 0.7
    horizon: int = 12
    warmup: int = 50
    cooldown: int = 2
    size_usdc: float = 500.0


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_trades: int = 0
    sharpe: float = 0.0
    total_pnl: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def walk_forward(
    df: pd.DataFrame,
    strategy_fn,
    config: WalkForwardConfig | None = None,
) -> list[FoldResult]:
    """Run walk-forward analysis on a strategy.

    Uses an expanding-window approach: train = all data before the test
    window.  Fold 0 includes a ``min_train`` buffer so it always has
    training rows (fixes the "fold 0 train too small" bug).

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV+indicators dataset, sorted by ts.
    strategy_fn : callable or Strategy
        Strategy to evaluate.
    config : WalkForwardConfig, optional
        Configuration for splits and backtest params.

    Returns
    -------
    list[FoldResult]
        One result per fold.
    """
    cfg = config or WalkForwardConfig()
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    min_train = cfg.warmup + cfg.horizon + 10
    testable = n - min_train
    if testable < cfg.horizon + 5:
        logger.warning("wf: not enough data (%d total, %d min_train)", n, min_train)
        return []

    fold_size = testable // cfg.n_splits
    results = []

    for fold in range(cfg.n_splits):
        test_start_idx = min_train + fold * fold_size
        test_end_idx = min_train + (fold + 1) * fold_size if fold < cfg.n_splits - 1 else n
        train = df.iloc[:test_start_idx]
        test = df.iloc[test_start_idx:test_end_idx]

        if len(test) < cfg.horizon + 5:
            logger.warning("wf fold %d: test too small (%d rows), skipping", fold, len(test))
            continue

        trades, summary = run_backtest(
            test, strategy_fn,
            horizon=cfg.horizon,
            warmup=0,
            cooldown=cfg.cooldown,
            size_usdc=cfg.size_usdc,
        )

        result = FoldResult(
            fold=fold,
            train_start=train["ts"].iloc[0],
            train_end=train["ts"].iloc[-1],
            test_start=test["ts"].iloc[0],
            test_end=test["ts"].iloc[-1],
            n_trades=summary.get("n_trades", 0),
            sharpe=summary.get("sharpe", 0.0),
            total_pnl=summary.get("total_pnl", 0.0),
            max_dd=summary.get("max_dd", 0.0),
            win_rate=summary.get("win_rate", 0.0),
            profit_factor=summary.get("profit_factor", 0.0),
            trades_df=trades,
        )
        results.append(result)

    return results


def walk_forward_summary(results: list[FoldResult]) -> dict:
    """Aggregate walk-forward results into a single summary dict."""
    if not results:
        return {"error": "no folds completed", "oos_sharpe_mean": 0.0, "oos_sharpe_std": 0.0}

    sharpes = [r.sharpe for r in results]
    pnls = [r.total_pnl for r in results]
    trades = sum(r.n_trades for r in results)

    mean_sharpe = float(pd.Series(sharpes).mean())
    passes = trades >= 15 and mean_sharpe > 0 and min(sharpes) > -0.5

    return {
        "n_folds": len(results),
        "n_trades_total": trades,
        "oos_sharpe_mean": round(mean_sharpe, 4),
        "oos_sharpe_std": round(float(pd.Series(sharpes).std()), 4),
        "oos_sharpe_min": round(min(sharpes), 4),
        "oos_sharpe_max": round(max(sharpes), 4),
        "oos_pnl_total": round(sum(pnls), 2),
        "passes": passes,
    }
