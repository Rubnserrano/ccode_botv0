"""Cost model for backtesting: fees, spread, slippage.

Defaults match Binance spot taker fees.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeCosts:
    entry_fee: float
    exit_fee: float
    spread_cost: float
    slippage: float
    total: float


def compute_costs(
    size_usdc: float,
    entry_price: float,
    taker_fee: float = 0.001,
    spread_bps: float = 2.0,
    avg_bar_volume_usdc: float | None = None,
) -> TradeCosts:
    """Compute realistic trading costs for a single trade.

    Parameters
    ----------
    size_usdc : float
        Position size in USDC (notional).
    entry_price : float
        Entry price (used for slippage calc if avg volume provided).
    taker_fee : float
        Per-side taker fee as fraction (0.001 = 0.1%).
    spread_bps : float
        Bid-ask spread in basis points (2.0 = 0.02%).
    avg_bar_volume_usdc : float | None
        Average bar volume in USDC. If provided, slippage scales with trade size.
        If None, flat 1bp slippage is used.

    Returns
    -------
    TradeCosts with all cost components.
    """
    entry_fee = size_usdc * taker_fee
    exit_fee = size_usdc * taker_fee
    spread_cost = size_usdc * (spread_bps / 10_000)

    if avg_bar_volume_usdc and avg_bar_volume_usdc > 0:
        size_ratio = size_usdc / avg_bar_volume_usdc
        slip_bps = min(1.0 + 0.5 * size_ratio * 10_000, 20.0)
        slippage = size_usdc * (slip_bps / 10_000)
    else:
        slippage = size_usdc * 0.0001

    total = entry_fee + exit_fee + spread_cost + slippage
    return TradeCosts(entry_fee, exit_fee, spread_cost, slippage, total)


def spot_pnl(direction: str, entry_price: float, exit_price: float, size_usdc: float) -> float:
    """Gross P&L for a spot position before costs."""
    pct = (exit_price - entry_price) / entry_price
    if direction == "SELL":
        pct = -pct
    return pct * size_usdc
