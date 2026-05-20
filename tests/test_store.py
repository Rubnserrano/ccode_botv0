"""Tests for src.ts_store — Universal TimeSeries Store.

Run with:  .venv/bin/python3 -m pytest tests/test_store.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.ts_store import write, read, delete, catalog, _parse_asset_id

ASSET_ID = "market:test_exchange:testsym"
FREQ = "raw"


@pytest.fixture(autouse=True)
def _clean_test_data():
    base = Path("data/ts/market") / ASSET_ID
    if base.exists():
        shutil.rmtree(base)
    yield
    if base.exists():
        shutil.rmtree(base)


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
        assert write(ASSET_ID, pd.DataFrame(), frequency=FREQ) == 0

    def test_write_one_batch(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00"])
        n = write(ASSET_ID, df, frequency=FREQ)
        assert n == 2

    def test_write_two_months(self):
        df = _df(["2026-04-30 23:59:00", "2026-05-01 00:00:00"])
        n = write(ASSET_ID, df, frequency=FREQ)
        assert n == 2
        freq_dir = Path("data/ts/market") / ASSET_ID / FREQ
        files = list(freq_dir.glob("*.parquet"))
        assert len(files) == 2

    def test_write_dedup(self):
        ts = "2026-05-01 00:00:00"
        df1 = _df([ts], close=[100.0])
        df2 = _df([ts], close=[200.0])
        write(ASSET_ID, df1, frequency=FREQ)
        write(ASSET_ID, df2, frequency=FREQ)
        result = read(ASSET_ID, frequency=FREQ)
        assert len(result) == 1
        assert result["close"].iloc[0] == 200.0

    def test_write_with_extra_columns(self):
        df = _df(["2026-05-01 00:00:00"])
        df["rsi_14"] = [55.0]
        write(ASSET_ID, df, frequency="features")
        result = read(ASSET_ID, frequency="features")
        assert "rsi_14" in result.columns


class TestRead:
    def test_read_empty(self):
        df = read("market:test_exchange:nonexistent", frequency=FREQ)
        assert df.empty

    def test_read_all(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00"])
        write(ASSET_ID, df, frequency=FREQ)
        result = read(ASSET_ID, frequency=FREQ)
        assert len(result) == 2

    def test_read_range(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00", "2026-05-01 00:02:00"])
        write(ASSET_ID, df, frequency=FREQ)
        start = datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc)
        result = read(ASSET_ID, frequency=FREQ, start=start)
        assert len(result) == 2

    def test_read_range_naive(self):
        df = _df(["2026-05-01 00:00:00"])
        write(ASSET_ID, df, frequency=FREQ)
        start = datetime(2026, 5, 1, 0, 0)
        result = read(ASSET_ID, frequency=FREQ, start=start)
        assert len(result) == 1

    def test_read_with_columns(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00"])
        write(ASSET_ID, df, frequency=FREQ)
        result = read(ASSET_ID, frequency=FREQ, columns=["ts", "close"])
        assert "close" in result.columns
        assert "open" not in result.columns

    def test_read_with_limit(self):
        df = _df(["2026-05-01 00:00:00", "2026-05-01 00:01:00", "2026-05-01 00:02:00"])
        write(ASSET_ID, df, frequency=FREQ)
        result = read(ASSET_ID, frequency=FREQ, limit=2)
        assert len(result) == 2


class TestDelete:
    def test_delete(self):
        df = _df(["2026-05-01 00:00:00"])
        write(ASSET_ID, df, frequency=FREQ)
        delete(ASSET_ID)
        result = read(ASSET_ID, frequency=FREQ)
        assert result.empty


class TestCatalog:
    def test_catalog_empty(self):
        cat = catalog()
        assert isinstance(cat, pd.DataFrame)

    def test_parse_asset_id_three_parts(self):
        st, src, name = _parse_asset_id("market:binance:btcusdt")
        assert st == "market"
        assert src == "binance"
        assert name == "btcusdt"

    def test_parse_asset_id_two_parts(self):
        st, src, name = _parse_asset_id("sentiment:fear_greed")
        assert st == "sentiment"
        assert name == "fear_greed"

    def test_parse_asset_id_one_part(self):
        st, src, name = _parse_asset_id("btcusdt")
        assert st == "other"