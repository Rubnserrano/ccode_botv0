"""Binance WebSocket real-time feeds.

Three concurrent streams per symbol:
  @ticker     → PriceBuffer  (30s window, used for edge/return calc)
  @aggTrade   → TradeBuffer  (CVD, VWAP — trade-level data with aggressor side)
  @kline_1m   → CandleBuffer (RSI, MACD, EMA, Heikin Ashi — 100 candle window)
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import websockets

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443"


# ─── Tick / Price ────────────────────────────────────────────────────────────

@dataclass
class Tick:
    timestamp_ms: int
    symbol: str
    price: float


@dataclass
class PriceBuffer:
    """Rolling 30s price tick window."""
    symbol: str
    window_seconds: int = 30
    _ticks: deque = field(default_factory=deque)

    def add(self, tick: Tick):
        cutoff_ms = tick.timestamp_ms - self.window_seconds * 1000
        self._ticks.append(tick)
        while self._ticks and self._ticks[0].timestamp_ms < cutoff_ms:
            self._ticks.popleft()

    @property
    def current_price(self) -> Optional[float]:
        return self._ticks[-1].price if self._ticks else None

    @property
    def return_30s_pct(self) -> Optional[float]:
        if len(self._ticks) < 2:
            return None
        return (self._ticks[-1].price - self._ticks[0].price) / self._ticks[0].price * 100

    @property
    def last_updated(self) -> Optional[datetime]:
        if not self._ticks:
            return None
        return datetime.fromtimestamp(self._ticks[-1].timestamp_ms / 1000, tz=timezone.utc)


# ─── Trades (aggTrade stream) ────────────────────────────────────────────────

@dataclass
class Trade:
    timestamp_ms: int
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool  # True=sell aggressor, False=buy aggressor


@dataclass
class TradeBuffer:
    """Rolling 5-minute window of aggTrades. Provides CVD and VWAP."""
    symbol: str
    window_seconds: int = 300
    _trades: deque = field(default_factory=deque)

    def add(self, trade: Trade):
        cutoff_ms = trade.timestamp_ms - self.window_seconds * 1000
        self._trades.append(trade)
        while self._trades and self._trades[0].timestamp_ms < cutoff_ms:
            self._trades.popleft()

    def cvd(self, seconds: int) -> float:
        """Cumulative Volume Delta over last N seconds (USD)."""
        if not self._trades:
            return 0.0
        latest_ms = self._trades[-1].timestamp_ms
        cutoff_ms = latest_ms - seconds * 1000
        delta = 0.0
        for t in self._trades:
            if t.timestamp_ms >= cutoff_ms:
                notional = t.price * t.quantity
                delta += notional if not t.is_buyer_maker else -notional
        return delta

    def vwap(self) -> Optional[float]:
        """VWAP over the full 5-minute window."""
        if not self._trades:
            return None
        total_qty = sum(t.quantity for t in self._trades)
        if total_qty == 0:
            return None
        return sum(t.price * t.quantity for t in self._trades) / total_qty

    @property
    def cvd_1m(self) -> float:
        return self.cvd(60)

    @property
    def trade_count(self) -> int:
        return len(self._trades)


# ─── Candles (kline_1m stream) ───────────────────────────────────────────────

@dataclass
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


@dataclass
class CandleBuffer:
    """Rolling buffer of 1m candles (max 100). Tracks gaps."""
    symbol: str
    max_candles: int = 100
    _candles: deque = field(default_factory=deque)
    _last_open_time_ms: Optional[int] = None
    _gaps: list[dict] = field(default_factory=list)

    def add(self, candle: Candle):
        if self._candles and self._candles[-1].open_time_ms == candle.open_time_ms:
            self._candles[-1] = candle
        else:
            if self._last_open_time_ms is not None:
                expected = self._last_open_time_ms + 60_000
                if candle.open_time_ms > expected:
                    self._gaps.append({
                        "from_ms": expected,
                        "to_ms": candle.open_time_ms,
                        "missing_minutes": (candle.open_time_ms - expected) // 60_000,
                    })
            self._last_open_time_ms = candle.open_time_ms
            self._candles.append(candle)
            while len(self._candles) > self.max_candles:
                self._candles.popleft()

    def detect_gaps(self) -> list[dict]:
        return list(self._gaps)

    def clear_gaps(self):
        self._gaps.clear()

    def closes(self) -> list[float]:
        return [c.close for c in self._candles]

    def closed_closes(self) -> list[float]:
        return [c.close for c in self._candles if c.is_closed]

    @property
    def candle_count(self) -> int:
        return sum(1 for c in self._candles if c.is_closed)

    @property
    def current_volume(self) -> float:
        return self._candles[-1].volume if self._candles else 0.0


# ─── Stream: ticker ──────────────────────────────────────────────────────────

async def stream_tickers(
    symbols: list[str],
    buffers: dict[str, PriceBuffer],
    tick_queue: asyncio.Queue,
):
    """Stream @ticker for all symbols into PriceBuffers and tick_queue.

    symbols: lowercase like ["btcusdt", "ethusdt"]
    buffers: keyed by asset e.g. {"btc": PriceBuffer(...)}
    """
    streams = "/".join(f"{s}@ticker" for s in symbols)
    uri = f"{BINANCE_WS_BASE}/stream?streams={streams}"

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                logger.info("feed: ticker connected %s", streams)
                async for raw in ws:
                    data = json.loads(raw)
                    payload = data.get("data", data)
                    if "s" not in payload or "c" not in payload:
                        continue
                    tick = Tick(
                        timestamp_ms=payload.get("T") or payload.get("E")
                                   or int(datetime.now(timezone.utc).timestamp() * 1000),
                        symbol=payload["s"],
                        price=float(payload["c"]),
                    )
                    asset = tick.symbol.lower().replace("usdt", "")
                    if asset in buffers:
                        buffers[asset].add(tick)
                    await tick_queue.put(tick)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("feed: ticker error %s — retry 5s", e)
            await asyncio.sleep(5)


# ─── Stream: aggTrade + kline_1m ──────────────────────────────────────────────

async def stream_trades_and_klines(
    symbols: list[str],
    trade_buffers: dict[str, TradeBuffer],
    candle_buffers: dict[str, CandleBuffer],
):
    """Combined stream: @aggTrade and @kline_1m for all symbols.

    Feeds TradeBuffer (CVD/VWAP) and CandleBuffer (RSI/MACD/EMA/HA).
    """
    agg = [f"{s}@aggTrade" for s in symbols]
    kln = [f"{s}@kline_1m" for s in symbols]
    uri = f"{BINANCE_WS_BASE}/stream?streams={'/'.join(agg + kln)}"

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                logger.info("feed: trade+kline connected %s", symbols)
                async for raw in ws:
                    data = json.loads(raw)
                    payload = data.get("data", {})
                    event = payload.get("e", "")
                    if event == "aggTrade":
                        symbol = payload["s"]
                        asset = symbol.lower().replace("usdt", "")
                        if asset in trade_buffers:
                            trade_buffers[asset].add(Trade(
                                timestamp_ms=payload["T"],
                                symbol=symbol,
                                price=float(payload["p"]),
                                quantity=float(payload["q"]),
                                is_buyer_maker=payload["m"],
                            ))
                    elif event == "kline":
                        symbol = payload["s"]
                        asset = symbol.lower().replace("usdt", "")
                        k = payload["k"]
                        if asset in candle_buffers:
                            candle_buffers[asset].add(Candle(
                                open_time_ms=k["t"],
                                open=float(k["o"]),
                                high=float(k["h"]),
                                low=float(k["l"]),
                                close=float(k["c"]),
                                volume=float(k["v"]),
                                is_closed=k["x"],
                            ))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("feed: trade+kline error %s — retry 5s", e)
            await asyncio.sleep(5)
