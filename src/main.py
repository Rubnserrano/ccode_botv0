"""Main ingestion orchestrator.

Startup:
  1. Check DataStore — download historical data if empty
  2. Create in-memory buffers (Price, Trade, Candle)
  3. Start WebSocket streams (ticker + trades/klines)
  4. Snapshot writer every 500ms → data/live_state.json
  5. Terminal report every 30s
  6. Graceful shutdown on SIGINT/SIGTERM

Usage:
    python -m src.main
    python -m src.main --symbols btcusdt,ethusdt --days 365
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.download import download_symbol
from src.store import available_range, row_count
from src.feed import (
    PriceBuffer, TradeBuffer, CandleBuffer,
    stream_tickers, stream_trades_and_klines,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXCHANGE = "binance"


def _live_state_path() -> Path:
    return DATA_DIR / "live_state.json"


# ─── Snapshot ────────────────────────────────────────────────────────────────

def _write_snapshot(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
):
    snapshot: dict = {"updated_at": datetime.now(timezone.utc).isoformat(), "assets": {}}
    for asset in prices:
        pb = prices[asset]
        tb = trades.get(asset)
        cb = candles.get(asset)
        entry: dict = {
            "price": pb.current_price,
            "return_30s_pct": round(pb.return_30s_pct, 4) if pb.return_30s_pct is not None else None,
        }
        if cb:
            entry.update({
                "volume": cb.current_volume,
                "candles_ready": cb.candle_count,
                "gaps": len(cb.detect_gaps()),
            })
        if tb:
            entry.update({
                "cvd_1m": round(tb.cvd_1m, 2),
                "vwap": round(tb.vwap(), 2) if tb.vwap() is not None else None,
                "trade_count": tb.trade_count,
            })
        snapshot["assets"][asset] = entry

    _live_state_path().write_text(json.dumps(snapshot, indent=2))


# ─── Services ────────────────────────────────────────────────────────────────

async def _snapshot_svc(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
):
    while True:
        _write_snapshot(prices, trades, candles)
        await asyncio.sleep(0.5)


async def _report_svc(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
):
    while True:
        await asyncio.sleep(30)
        now = datetime.now(timezone.utc).isoformat()
        lines = [f"\n{'─'*60}", f"  {now}", f"{'─'*60}"]
        for asset in prices:
            pb = prices[asset]
            cb = candles.get(asset)
            tb = trades.get(asset)
            price = pb.current_price
            if price is None:
                lines.append(f"  {asset}: waiting for data...")
                continue
            ret = pb.return_30s_pct
            ret_s = f"{ret:+.3f}%" if ret is not None else "n/a"
            cvd = tb.cvd_1m if tb else 0
            ready = cb.candle_count if cb else 0
            lines.append(f"  {asset}: ${price:,.2f}  30s={ret_s}  CVD={cvd:+.0f}  candles={ready}")
        print("\n".join(lines))


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="BTC data ingestion orchestrator")
    parser.add_argument("--symbols", default="btcusdt", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=90, help="Days of history if empty")
    parser.add_argument("--skip-download", action="store_true", help="Skip historical download")
    args = parser.parse_args()

    symbols = [s.strip().lower() for s in args.symbols.split(",")]
    assets = [s.replace("usdt", "") for s in symbols]

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    logger.info("orchestrator: starting — symbols=%s", symbols)

    # 1) Historical download if needed
    if not args.skip_download:
        for symbol in symbols:
            n = row_count(EXCHANGE, symbol)
            if n == 0:
                logger.info("orchestrator: no data for %s, downloading %d days", symbol, args.days)
                await download_symbol(symbol.upper(), days=args.days)
            else:
                start, end = available_range(EXCHANGE, symbol)
                logger.info("orchestrator: %s has %d rows (%s → %s)", symbol, n, start, end)

    # 2) In-memory buffers
    price_buffers: dict[str, PriceBuffer] = {}
    trade_buffers: dict[str, TradeBuffer] = {}
    candle_buffers: dict[str, CandleBuffer] = {}

    for asset in assets:
        price_buffers[asset] = PriceBuffer(asset)
        trade_buffers[asset] = TradeBuffer(asset)
        candle_buffers[asset] = CandleBuffer(asset)

    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # 3) Start services
    tasks = [
        asyncio.create_task(stream_tickers(symbols, price_buffers, tick_queue)),
        asyncio.create_task(stream_trades_and_klines(symbols, trade_buffers, candle_buffers)),
        asyncio.create_task(_snapshot_svc(price_buffers, trade_buffers, candle_buffers)),
        asyncio.create_task(_report_svc(price_buffers, trade_buffers, candle_buffers)),
    ]

    def _shutdown():
        for t in tasks:
            t.cancel()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass  # Windows

    logger.info("orchestrator: running — %d services", len(tasks))
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("orchestrator: stopped")


if __name__ == "__main__":
    asyncio.run(main())
