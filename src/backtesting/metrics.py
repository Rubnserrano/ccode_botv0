"""Trade metrics: Sharpe, Sortino, MaxDD, WinRate, ProfitFactor, p-value.

All functions are pure — they take a list of PnL values and return metrics.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass(frozen=True)
class TradeMetrics:
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    sortino: float
    max_dd: float
    profit_factor: float
    p_value: float
    passes_gates: bool


def _max_drawdown(pnls: list[float]) -> float:
    """Peak-to-trough drawdown as a negative value."""
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = min(dd, cum - peak)
    return dd


def _bootstrap_pvalue(pnls: list[float], n_resamples: int = 1000) -> float:
    """Two-sided bootstrap p-value: H0 = mean is zero."""
    if len(pnls) < 5:
        return 1.0
    observed = mean(pnls)
    shifted = [p - observed for p in pnls]
    count = 0
    for _ in range(n_resamples):
        sample = [random.choice(shifted) for _ in pnls]
        if abs(mean(sample)) >= abs(observed):
            count += 1
    return count / n_resamples


def compute_metrics(pnls: list[float], total_turnover: float) -> TradeMetrics:
    """Compute all trade metrics from a list of net PnL values."""
    n = len(pnls)
    if n == 0:
        return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False)

    wins = sum(1 for p in pnls if p > 0)
    losses = n - wins
    win_rate = wins / n
    total_pnl = sum(pnls)
    avg = mean(pnls)

    sd = stdev(pnls) if n > 1 else 0.0
    sharpe = avg / sd if sd > 0 else 0.0

    downside = [p for p in pnls if p < 0]
    downside_sd = stdev(downside) if len(downside) > 1 else 0.0
    sortino = avg / downside_sd if downside_sd > 0 else float("inf") if avg > 0 else 0.0

    max_dd = _max_drawdown(pnls)
    profit_factor = sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if losses > 0 else float("inf")

    p_value = _bootstrap_pvalue(pnls)

    # Gates
    min_trades = 30
    min_win_rate = 0.43
    min_sharpe = 0.5
    max_dd_pct = 0.25
    max_p_value = 0.05

    passes = (
        n >= min_trades
        and win_rate >= min_win_rate
        and sharpe >= min_sharpe
        and (abs(max_dd) / total_turnover <= max_dd_pct if total_turnover > 0 else True)
        and p_value <= max_p_value
    )

    return TradeMetrics(
        n_trades=n, wins=wins, losses=losses,
        win_rate=round(win_rate, 4),
        total_pnl=round(total_pnl, 2),
        avg_pnl=round(avg, 2),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_dd=round(max_dd, 2),
        profit_factor=round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
        p_value=round(p_value, 4),
        passes_gates=passes,
    )
