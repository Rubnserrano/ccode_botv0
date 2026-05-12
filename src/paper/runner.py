"""Paper trading runner — evaluates strategies on each closed candle.

Connects to CandleBuffer's on_close callback (same as Persistence).
When a candle closes:
  1. Runs calc_all() on recent data
  2. Generates signals via strategy
  3. If signal and no position → open
  4. If position open → check TP/SL/horizon

Writes state.json + equity.parquet every candle close.
"""
from __future__ import annotations

import argparse
import logging
from collections import deque

import pandas as pd

from src.paper.account import PaperAccount
from src.paper.state import write_state, append_equity
from src.backtesting.strategies import STRATEGY_REGISTRY
from src.indicators.calculator import calc_all

logger = logging.getLogger(__name__)

MAX_CANDLES = 200  # enough for calc_all warmup


class PaperRunner:
    """Evaluates a strategy on real-time candle closes.

    Usage from main.py or standalone:
        runner = PaperRunner("ema_trend")
        # pass to CandleBuffer on_close:
        candle_buffers["btc"] = CandleBuffer("btc", on_close=runner.on_candle_close)
    """

    def __init__(
        self,
        strategy_name: str = "ema_trend",
        size_usdc: float = 50.0,
        horizon: int = 12,
        tp_pct: float = 0.005,
        sl_pct: float = 0.005,
        warmup: int = 50,
    ):
        self.strategy = STRATEGY_REGISTRY[strategy_name]
        self.size_usdc = size_usdc
        self.horizon = horizon
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.warmup = warmup
        self.account = PaperAccount(initial_capital=10_000.0)
        self._candles: list[dict] = []
        self._bar_counter = -1

        logger.info("paper: runner initialized — strategy=%s size=$%.0f horizon=%d tp=%.1f%% sl=%.1f%%",
                     strategy_name, size_usdc, horizon, tp_pct * 100, sl_pct * 100)

    def prefill(self, symbol: str, n_candles: int = 200, exchange: str = "binance") -> None:
        """Prefill buffer with historical data so indicators are ready immediately.

        Reads from Feature Store (or falls back to raw+calc_all),
        feeds candles into the buffer, and sets ``_bar_counter`` past warmup.
        """
        try:
            from src.features.store import read as f_read, available_range as f_range
            f_start, _ = f_range(symbol)
            if f_start is not None:
                df = f_read(symbol)
            else:
                from src.store import read as raw_read
                df = raw_read(exchange, symbol)
                from src.indicators.calculator import calc_all
                df = calc_all(df)
        except Exception as e:
            logger.warning("paper: prefill failed (%s), will warm up live", e)
            return

        df = df.iloc[-n_candles:]
        for _, row in df.iterrows():
            self._candles.append({
                "ts": row["ts"].value // 1_000_000,
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            })

        self._bar_counter = n_candles
        logger.info("paper: prefill done — %d candles loaded, warmup skipped", n_candles)

    def on_candle_close(self, candle) -> None:
        """Called by CandleBuffer when a 1m candle closes."""
        self._bar_counter += 1
        self._candles.append({
            "ts": candle.open_time_ms,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        })

        if len(self._candles) > MAX_CANDLES:
            self._candles.pop(0)

        if self._bar_counter < self.warmup:
            if self._bar_counter % 30 == 0:
                logger.info("paper: warming up %d/%d", self._bar_counter, self.warmup)
            return

        # Build DataFrame from recent candles
        df = pd.DataFrame(list(self._candles))
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = calc_all(df)
        latest = df.iloc[-1]

        # Check exit first
        if self.account.position is not None:
            exit_reason = self.account.check_exit(
                latest["high"], latest["low"], latest["close"],
                self._bar_counter,
            )
            if exit_reason:
                state = self.account.state_dict()
                write_state(state)
                append_equity(state)
                logger.info("paper: closed %s — PnL=$%.2f  capital=$%.2f",
                             exit_reason,
                             self.account.trades[-1]["net_pnl"] if self.account.trades else 0,
                             self.account.capital)

        # Check entry
        if self.account.position is None:
            sig = self.strategy.generate(df).iloc[-1]
            if sig != 0:
                direction = "LONG" if sig == 1 else "SHORT"
                ok = self.account.open_position(
                    direction, latest["close"], self.size_usdc,
                    horizon_bars=self.horizon, tp_pct=self.tp_pct, sl_pct=self.sl_pct,
                )
                if ok:
                    state = self.account.state_dict()
                    write_state(state)
                    logger.info("paper: opened %s at $%.2f  capital=$%.2f",
                                 direction, latest["close"], self.account.capital)

        # Periodic state write (every 5 candles even without trade)
        if self._bar_counter % 5 == 0:
            state = self.account.state_dict()
            write_state(state)
            append_equity(state)


def main():
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--strategy", default="ema_trend", choices=list(STRATEGY_REGISTRY),
                        help="Strategy name")
    parser.add_argument("--size", type=float, default=50.0, help="Position size USDC")
    parser.add_argument("--horizon", type=int, default=12, help="Hold bars")
    parser.add_argument("--tp", type=float, default=0.005, help="Take profit fraction")
    parser.add_argument("--sl", type=float, default=0.005, help="Stop loss fraction")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    runner = PaperRunner(
        strategy_name=args.strategy,
        size_usdc=args.size,
        horizon=args.horizon,
        tp_pct=args.tp,
        sl_pct=args.sl,
    )
    runner.account.initial_capital = args.capital
    runner.account.capital = args.capital

    print(f"Paper runner ready — strategy={args.strategy} capital=${args.capital}")
    print("Pass PaperRunner.on_candle_close to CandleBuffer on_close.")
    print("Or import and use directly:\n")
    print("  from src.paper.runner import PaperRunner")
    print("  runner = PaperRunner('ema_trend')")
    print('  candle_buffers["btc"] = CandleBuffer("btc", on_close=runner.on_candle_close)')
    print()
    print("Run standalone without WS for testing:")
    print("  python -c \"from src.paper.runner import PaperRunner\"")
    print()

    # Keep alive so the runner can be registered
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
