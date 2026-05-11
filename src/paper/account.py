"""Paper account — simulates capital, positions, and P&L.

No external API calls. No real funds ever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Position:
    side: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    size_usdc: float
    horizon_bars: int = 0
    bars_held: int = 0
    tp_pct: float = 0.005
    sl_pct: float = 0.005


@dataclass
class PaperAccount:
    initial_capital: float = 10_000.0
    capital: float = 10_000.0
    position: Position | None = None
    trades: list[dict] = field(default_factory=list)
    total_pnl: float = 0.0
    total_fees: float = 0.0

    def open_position(self, side: str, price: float, size_usdc: float,
                      horizon_bars: int = 12, tp_pct: float = 0.005, sl_pct: float = 0.005) -> bool:
        if self.position is not None:
            return False
        if size_usdc > self.capital:
            return False
        self.position = Position(
            side=side, entry_price=price, entry_time=datetime.now(timezone.utc),
            size_usdc=size_usdc, horizon_bars=horizon_bars,
            tp_pct=tp_pct, sl_pct=sl_pct,
        )
        self.capital -= size_usdc
        return True

    def close_position(self, price: float, reason: str = "horizon") -> dict | None:
        if self.position is None:
            return None
        pos = self.position
        gross = self._pnl(pos.side, pos.entry_price, price, pos.size_usdc)
        fees = pos.size_usdc * 0.001 * 2  # entry + exit
        net = gross - fees
        self.capital += pos.size_usdc + net
        self.total_pnl += net
        self.total_fees += fees

        trade = {
            "ts": pos.entry_time.isoformat(),
            "exit_ts": datetime.now(timezone.utc).isoformat(),
            "side": pos.side,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(price, 2),
            "size_usdc": pos.size_usdc,
            "gross_pnl": round(gross, 2),
            "fees": round(fees, 4),
            "net_pnl": round(net, 2),
            "reason": reason,
            "horizon_bars": pos.bars_held,
        }
        self.trades.append(trade)
        self.position = None
        return trade

    def check_exit(self, high: float, low: float, close: float, bar_idx: int) -> str | None:
        pos = self.position
        if pos is None:
            return None
        pos.bars_held = bar_idx
        if pos.side == "LONG":
            if high >= pos.entry_price * (1 + pos.tp_pct):
                self.close_position(pos.entry_price * (1 + pos.tp_pct), "tp")
                return "tp"
            if low <= pos.entry_price * (1 - pos.sl_pct):
                self.close_position(pos.entry_price * (1 - pos.sl_pct), "sl")
                return "sl"
        else:
            if low <= pos.entry_price * (1 - pos.tp_pct):
                self.close_position(pos.entry_price * (1 - pos.tp_pct), "tp")
                return "tp"
            if high >= pos.entry_price * (1 + pos.sl_pct):
                self.close_position(pos.entry_price * (1 + pos.sl_pct), "sl")
                return "sl"
        if pos.bars_held >= pos.horizon_bars:
            self.close_position(close, "horizon")
            return "horizon"
        return None

    def equity(self) -> float:
        pos_value = 0.0
        if self.position:
            pos_value = self.position.size_usdc
        return self.capital + pos_value

    def state_dict(self) -> dict:
        pos_dict = None
        if self.position:
            pos_dict = {
                "side": self.position.side,
                "entry_price": round(self.position.entry_price, 2),
                "size_usdc": self.position.size_usdc,
                "bars_held": self.position.bars_held,
                "horizon_bars": self.position.horizon_bars,
            }
        return {
            "initial_capital": self.initial_capital,
            "capital": round(self.capital, 2),
            "equity": round(self.equity(), 2),
            "position": pos_dict,
            "total_pnl": round(self.total_pnl, 2),
            "total_fees": round(self.total_fees, 4),
            "n_trades": len(self.trades),
        }

    @staticmethod
    def _pnl(side: str, entry: float, exit_p: float, size: float) -> float:
        pct = (exit_p - entry) / entry
        if side == "SHORT":
            pct = -pct
        return pct * size
