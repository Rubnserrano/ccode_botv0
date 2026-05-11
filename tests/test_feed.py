"""Tests for src.feed — data buffers and callbacks.

Run with:  .venv/bin/python3 -m pytest tests/test_feed.py -v
"""
from __future__ import annotations

import asyncio

import pytest

from src.feed import (
    Tick, Trade, Candle,
    PriceBuffer, TradeBuffer, CandleBuffer,
)


# ─── PriceBuffer ─────────────────────────────────────────────────────────────

class TestPriceBuffer:
    def test_empty(self):
        pb = PriceBuffer("btc")
        assert pb.current_price is None
        assert pb.return_30s_pct is None
        assert pb.last_updated is None

    def test_single_tick(self):
        pb = PriceBuffer("btc")
        pb.add(Tick(timestamp_ms=1000, symbol="BTCUSDT", price=100.0))
        assert pb.current_price == 100.0
        assert pb.return_30s_pct is None

    def test_two_ticks(self):
        pb = PriceBuffer("btc")
        pb.add(Tick(timestamp_ms=1000, symbol="BTCUSDT", price=100.0))
        pb.add(Tick(timestamp_ms=2000, symbol="BTCUSDT", price=101.0))
        assert pb.current_price == 101.0
        assert pb.return_30s_pct == 1.0

    def test_window_eviction(self):
        pb = PriceBuffer("btc", window_seconds=3)
        pb.add(Tick(timestamp_ms=1000, symbol="BTCUSDT", price=100.0))
        pb.add(Tick(timestamp_ms=2000, symbol="BTCUSDT", price=101.0))
        pb.add(Tick(timestamp_ms=5000, symbol="BTCUSDT", price=102.0))
        assert pb.current_price == 102.0


# ─── TradeBuffer ─────────────────────────────────────────────────────────────

class TestTradeBuffer:
    def test_empty(self):
        tb = TradeBuffer("btc")
        assert tb.cvd_1m == 0.0
        assert tb.trade_count == 0
        assert tb.vwap() is None
        assert tb.cvd_total == 0.0

    def test_buyer_trades(self):
        tb = TradeBuffer("btc", window_seconds=300)
        tb.add(Trade(timestamp_ms=1000, symbol="BTCUSDT", price=100.0, quantity=1.0, is_buyer_maker=False))
        tb.add(Trade(timestamp_ms=2000, symbol="BTCUSDT", price=101.0, quantity=2.0, is_buyer_maker=False))
        assert tb.cvd_1m == pytest.approx(100 * 1 + 101 * 2)
        assert tb.trade_count == 2

    def test_seller_trades(self):
        tb = TradeBuffer("btc", window_seconds=300)
        tb.add(Trade(timestamp_ms=1000, symbol="BTCUSDT", price=100.0, quantity=1.0, is_buyer_maker=True))
        assert tb.cvd_1m == pytest.approx(-100.0)

    def test_mixed_trades(self):
        tb = TradeBuffer("btc", window_seconds=300)
        tb.add(Trade(timestamp_ms=1000, symbol="BTCUSDT", price=100.0, quantity=1.0, is_buyer_maker=False))
        tb.add(Trade(timestamp_ms=2000, symbol="BTCUSDT", price=101.0, quantity=2.0, is_buyer_maker=True))
        assert tb.cvd_1m == pytest.approx(100 - 202)

    def test_vwap(self):
        tb = TradeBuffer("btc", window_seconds=300)
        tb.add(Trade(timestamp_ms=1000, symbol="BTCUSDT", price=100.0, quantity=1.0, is_buyer_maker=False))
        tb.add(Trade(timestamp_ms=2000, symbol="BTCUSDT", price=200.0, quantity=3.0, is_buyer_maker=True))
        assert tb.vwap() == pytest.approx(175.0)

    def test_window_eviction(self):
        tb = TradeBuffer("btc", window_seconds=5)
        tb.add(Trade(timestamp_ms=1000, symbol="BTCUSDT", price=100.0, quantity=1.0, is_buyer_maker=False))
        tb.add(Trade(timestamp_ms=7000, symbol="BTCUSDT", price=200.0, quantity=1.0, is_buyer_maker=True))
        assert tb.trade_count == 1
        assert tb.cvd_total == -200.0


# ─── CandleBuffer ────────────────────────────────────────────────────────────

class TestCandleBuffer:
    def test_empty(self):
        cb = CandleBuffer("btc")
        assert cb.candle_count == 0
        assert cb.closes() == []
        assert cb.current_volume == 0.0

    def test_add_open_candle(self):
        cb = CandleBuffer("btc")
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=False))
        assert cb.candle_count == 0
        assert cb.current_volume == 1000.0

    def test_closed_candle(self):
        cb = CandleBuffer("btc")
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        assert cb.candle_count == 1
        assert len(cb.closed_closes()) == 1

    def test_max_candles(self):
        cb = CandleBuffer("btc", max_candles=3)
        for i in range(5):
            cb.add(Candle(open_time_ms=i * 60_000, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        assert cb.candle_count == 3

    def test_gap_detection(self):
        cb = CandleBuffer("btc")
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        cb.add(Candle(open_time_ms=300_000, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        gaps = cb.detect_gaps()
        assert len(gaps) == 1
        assert gaps[0]["missing_minutes"] == 4

    def test_no_gap(self):
        cb = CandleBuffer("btc")
        for i in range(3):
            cb.add(Candle(open_time_ms=i * 60_000, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        assert len(cb.detect_gaps()) == 0


# ─── CandleBuffer on_close callback ─────────────────────────────────────────

class TestOnClose:
    def test_on_close_called(self):
        events = []
        cb = CandleBuffer("btc", on_close=lambda c: events.append(c))
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=False))
        assert len(events) == 0
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        assert len(events) == 1

    def test_on_close_dedup(self):
        events = []
        cb = CandleBuffer("btc", on_close=lambda c: events.append(c))
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=101.0, volume=1000, is_closed=True))
        assert len(events) == 1

    def test_on_close_new_candle(self):
        events = []
        cb = CandleBuffer("btc", on_close=lambda c: events.append(c))
        cb.add(Candle(open_time_ms=60_000, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
        assert len(events) == 1

    def test_async_callback_creates_task(self):
        """Async callback should not crash — _fire_on_close creates a task."""
        events = []
        async def async_cb(c):
            events.append(c)

        async def run():
            cb = CandleBuffer("btc", on_close=async_cb)
            cb.add(Candle(open_time_ms=0, open=100, high=101, low=99, close=100.5, volume=1000, is_closed=True))
            await asyncio.sleep(0.01)
            return len(events)

        n = asyncio.run(run())
        assert n == 1
