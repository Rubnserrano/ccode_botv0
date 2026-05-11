"""Tests for src.store — Parquet DataStore.

Run with:  .venv/bin/python3 -m pytest tests/test_store.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.store import write, read, available_range, row_count

EXCHANGE = "test_exchange"
SYMBOL = "testsym"


def _df(ts_list: list[str], **overrides) -> pd.DataFrame:
    n = len(ts_list)
    return pd.DataFrame({
        "ts": pd.to_datetime(ts_list, utc=True),
        "open": overrides.get("open", [100.0] * n),
        "high": overrides.get("high", [101.0] * n),
        "low": overrides.get("low", [99.0] * n),
        "close": overrides.get("close", [100.5] * n),
        "volume": overrides.get("volume", [1000.0] * n),
    })


class TestWrite:
    def test_write_empty(self):
        assert write(EXCHANGE, SYMBOL, pd.DataFrame()) == 0

    def test_write_one_batch(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00"])
        n = write(EXCHANGE, SYMBOL, df)
        assert n == 2

    def test_write_two_months(self):
        df = _df(["2026-04-30 23:59:00", "2026-05-01 00:00:00"])
        n = write(EXCHANGE, SYMBOL, df)
        assert n == 2
        files = list((Path("data/raw") / EXCHANGE / SYMBOL).glob("*.parquet"))
        assert len(files) == 2

    def test_write_dedup(self):
        ts = "2026-05-01 00:00:00"
        df1 = _df([ts], close=[100.0])
        df2 = _df([ts], close=[200.0])
        write(EXCHANGE, SYMBOL, df1)
        write(EXCHANGE, SYMBOL, df2)
        result = read(EXCHANGE, SYMBOL)
        assert len(result) == 1
        assert result["close"].iloc[0] == 200.0


class TestRead:
    def test_read_empty(self):
        df = read(EXCHANGE, SYMBOL)
        assert df.empty

    def test_read_all(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00"])
        write(EXCHANGE, SYMBOL, df)
        result = read(EXCHANGE, SYMBOL)
        assert len(result) == 2

    def test_read_range(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00", "2026-05-01 00:02:00"])
        write(EXCHANGE, SYMBOL, df)
        start = datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc)
        result = read(EXCHANGE, SYMBOL, start=start)
        assert len(result) == 2

    def test_read_range_naive(self):
        df = _df(["2026-05-01 00:00:00"])
        write(EXCHANGE, SYMBOL, df)
        start = datetime(2026, 5, 1, 0, 0)
        result = read(EXCHANGE, SYMBOL, start=start)
        assert len(result) == 1


class TestMetadata:
    def test_available_range_empty(self):
        start, end = available_range(EXCHANGE, SYMBOL)
        assert start is None and end is None

    def test_available_range(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-02 00:00:00"])
        write(EXCHANGE, SYMBOL, df)
        start, end = available_range(EXCHANGE, SYMBOL)
        assert start is not None and end is not None
        assert start < end

    def test_row_count(self):
        df = _df([f"2026-05-01 00:{i:02d}:00" for i in range(5)])
        write(EXCHANGE, SYMBOL, df)
        assert row_count(EXCHANGE, SYMBOL) == 5
