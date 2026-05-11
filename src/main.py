"""Main ingestion orchestrator.

Startup:
  1. Connect to TimescaleDB + init schema
  2. Create buffers + start WS streams (ticker + trades/klines) IMMEDIATELY
  3. In background: download history if needed, sync to TimescaleDB
  4. Snapshot every 500ms → data/live_state.json
  5. Terminal report every 30s
  6. Graceful shutdown on SIGINT/SIGTERM

Usage:
    export TIMESCALE_DSN="postgres://ccode:ccode@localhost:5432/ccode"
    python -m src.main
    python -m src.main --skip-download    # skip download + sync
    python -m src.main --no-tsdb          # skip TimescaleDB entirely
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from inspect import iscoroutine
from pathlib import Path

import pandas as pd

from src.download import download_symbol
from src.store import available_range, row_count, write
from src.feed import (
    PriceBuffer, TradeBuffer, CandleBuffer,
    stream_tickers, stream_trades_and_klines,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXCHANGE = "binance"

TIMESCALE_DSN = os.getenv("TIMESCALE_DSN", "postgres://ccode:ccode@localhost:5432/ccode")


def _live_state_path() -> Path:
    return DATA_DIR / "live_state.json"


# ─── Snapshot ────────────────────────────────────────────────────────────────

def _write_snapshot(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
    download_done: bool = False,
):
    snapshot: dict = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "download_complete": download_done,
        "assets": {},
    }
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


# ─── Background download + sync ──────────────────────────────────────────────

async def _download_and_sync(
    symbols: list[str],
    days: int,
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
    tsdb,
    notify_event: asyncio.Event,
):
    """Run historical download + TSDB sync in background.

    WS streams are already running — this runs concurrently.
    """
    try:
        for symbol in symbols:
            n = row_count(EXCHANGE, symbol)
            if n == 0:
                logger.info("orchestrator: no data for %s, downloading %d days (background)", symbol, days)
                await download_symbol(symbol.upper(), days=days)
            else:
                start, end = available_range(EXCHANGE, symbol)
                logger.info("orchestrator: %s has %d rows (%s → %s)", symbol, n, start, end)

        if tsdb:
            for symbol in symbols:
                try:
                    n = await tsdb.sync_from_parquet(EXCHANGE, symbol.lower())
                    logger.info("orchestrator: synced %d rows to TimescaleDB for %s", n, symbol)
                except Exception as e:
                    logger.warning("orchestrator: sync failed for %s: %s", symbol, e)
    except Exception as e:
        logger.error("orchestrator: background download/sync failed: %s", e)
    finally:
        notify_event.set()
        logger.info("orchestrator: background download + sync complete")


# ─── Services ────────────────────────────────────────────────────────────────

async def _snapshot_svc(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
    download_event: asyncio.Event,
):
    download_done = False
    while True:
        if download_event.is_set() and not download_done:
            download_done = True
        _write_snapshot(prices, trades, candles, download_done)
        await asyncio.sleep(0.5)


async def _report_svc(
    prices: dict[str, PriceBuffer],
    trades: dict[str, TradeBuffer],
    candles: dict[str, CandleBuffer],
    tsdb=None,
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
        if tsdb and tsdb.pool:
            try:
                async with tsdb.pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT COUNT(*) FROM ohlcv")
                    lines.append(f"  TimescaleDB: {row['count']:,} rows" if row else "")
            except Exception:
                lines.append("  TimescaleDB: ?")
        print("\n".join(lines))


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="BTC data ingestion orchestrator")
    parser.add_argument("--symbols", default="btcusdt", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=862, help="Days of history if empty (default: 862 ~ Jan 2024)")
    parser.add_argument("--skip-download", action="store_true", help="Skip historical download + sync")
    parser.add_argument("--no-tsdb", action="store_true", help="Skip TimescaleDB entirely")
    parser.add_argument("--paper", type=str, default=None, choices=["rsi_mean_reversion", "macd_crossover", "ema_trend", "vwap_bounce", "heikin_ashi_streak"],
                        help="Enable paper trading with strategy name")
    args = parser.parse_args()

    symbols = [s.strip().lower() for s in args.symbols.split(",")]
    assets = [s.replace("usdt", "") for s in symbols]

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # 1) TimescaleDB
    tsdb = None
    if not args.no_tsdb:
        try:
            from src.tsdb import TimescaleDB
            tsdb = TimescaleDB()
            await tsdb.connect(TIMESCALE_DSN)
            logger.info("orchestrator: TimescaleDB connected")
        except Exception as e:
            logger.warning("orchestrator: TimescaleDB unavailable (%s) — running without it", e)
            tsdb = None

    logger.info("orchestrator: starting — symbols=%s", symbols)

    # 2) Buffers (created immediately, WS will fill them)
    price_buffers: dict[str, PriceBuffer] = {}
    trade_buffers: dict[str, TradeBuffer] = {}
    candle_buffers: dict[str, CandleBuffer] = {}

    # Paper trading runner
    paper_runner = None
    if args.paper:
        from src.paper.runner import PaperRunner
        paper_runner = PaperRunner(strategy_name=args.paper)
        logger.info("orchestrator: paper trading enabled — strategy=%s", args.paper)

    for asset in assets:
        price_buffers[asset] = PriceBuffer(asset)
        trade_buffers[asset] = TradeBuffer(asset)
        on_close = _candle_closed(asset, tsdb)
        if paper_runner:
            _paper = paper_runner
            _persist = on_close

            def _chain(candle):
                result = _persist(candle)
                if iscoroutine(result):
                    asyncio.create_task(_chain_async(candle, result, _paper))
                else:
                    _paper.on_candle_close(candle)

            async def _chain_async(candle, persist_result, paper):
                await persist_result
                paper.on_candle_close(candle)

            on_close = _chain
        candle_buffers[asset] = CandleBuffer(asset, on_close=on_close)

    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # 3) Background download signal
    download_done = asyncio.Event()
    if args.skip_download:
        download_done.set()

    # 4) Start all services (WS first, then background download)
    tasks = [
        asyncio.create_task(stream_tickers(symbols, price_buffers, tick_queue)),
        asyncio.create_task(stream_trades_and_klines(symbols, trade_buffers, candle_buffers)),
        asyncio.create_task(_snapshot_svc(price_buffers, trade_buffers, candle_buffers, download_done)),
        asyncio.create_task(_report_svc(price_buffers, trade_buffers, candle_buffers, tsdb)),
    ]

    if not args.skip_download:
        tasks.append(asyncio.create_task(
            _download_and_sync(symbols, args.days, price_buffers, trade_buffers, candle_buffers, tsdb, download_done)
        ))

    def _shutdown():
        for t in tasks:
            t.cancel()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    logger.info("orchestrator: running — %d services (WS live, download in background)", len(tasks))
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        if tsdb:
            await tsdb.close()
        logger.info("orchestrator: stopped")


def _candle_closed(asset: str, tsdb):
    """Return a callback that persists closed candles to Parquet + TimescaleDB."""
    symbol_upper = f"{asset.upper()}USDT"

    async def _on_close(candle):
        row_df = pd.DataFrame([{
            "ts": datetime.fromtimestamp(candle.open_time_ms / 1000, tz=timezone.utc),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }])
        try:
            write(EXCHANGE, symbol_upper, row_df)
        except Exception as e:
            logger.error("persist: parquet write failed: %s", e)

        if tsdb:
            try:
                await tsdb.insert_candle(
                    ts=datetime.fromtimestamp(candle.open_time_ms / 1000, tz=timezone.utc),
                    symbol=symbol_upper,
                    open_p=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                )
            except Exception as e:
                logger.error("persist: tsdb insert failed: %s", e)

    return _on_close


if __name__ == "__main__":
    asyncio.run(main())
