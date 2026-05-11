"""TimescaleDB client for OHLCV storage.

Hypertable schema with continuous aggregates for 5m and 1h.
Used alongside Parquet — Parquet is source of truth, TimescaleDB for fast queries.

Usage:
    tsdb = TimescaleDB()
    await tsdb.connect("postgres://ccode:ccode@localhost:5432/ccode")
    await tsdb.sync_from_parquet("binance", "btcusdt")
    await tsdb.insert_candle(...)
    await tsdb.close()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import asyncpg

from src.store import read, available_range

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ts      TIMESTAMPTZ NOT NULL,
    symbol  TEXT NOT NULL,
    open    DOUBLE PRECISION NOT NULL,
    high    DOUBLE PRECISION NOT NULL,
    low     DOUBLE PRECISION NOT NULL,
    close   DOUBLE PRECISION NOT NULL,
    volume  DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('ohlcv', 'ts', chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_uniq ON ohlcv (ts, symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts ON ohlcv (symbol, ts DESC);

-- Continuous aggregate: 5m
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', ts) AS bucket,
    symbol,
    FIRST(open, ts) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, ts) AS close,
    SUM(volume) AS volume
FROM ohlcv
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_5m',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

-- Continuous aggregate: 1h
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS bucket,
    symbol,
    FIRST(open, ts) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, ts) AS close,
    SUM(volume) AS volume
FROM ohlcv
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);
"""


class TimescaleDB:
    """Async client for TimescaleDB OHLCV storage."""

    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self, dsn: str, min_size: int = 2, max_size: int = 5) -> None:
        """Connect to TimescaleDB and initialize schema."""
        self.pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        await self._init_schema()
        logger.info("tsdb: connected and schema ready")

    async def _init_schema(self) -> None:
        """Create hypertable, indexes, and continuous aggregates if not exist."""
        async with self.pool.acquire() as conn:
            for stmt in _SCHEMA_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        await conn.execute(stmt + ";")
                    except Exception as e:
                        logger.warning("tsdb: schema stmt skipped: %s", e)

    async def bulk_insert(self, rows: list[tuple]) -> int:
        """Insert many rows via COPY.

        Each row: (ts, symbol, open, high, low, close, volume)
        Returns row count inserted.
        """
        if not rows:
            return 0
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                "ohlcv",
                records=rows,
                columns=["ts", "symbol", "open", "high", "low", "close", "volume"],
            )
        return len(rows)

    async def insert_candle(
        self,
        ts: datetime,
        symbol: str,
        open_p: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Insert a single closed candle (real-time path)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO ohlcv (ts, symbol, open, high, low, close, volume) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                "ON CONFLICT (ts, symbol) DO NOTHING",
                ts, symbol, open_p, high, low, close, volume,
            )

    async def latest_ts(self, symbol: str) -> datetime | None:
        """Return the most recent timestamp for a symbol."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(ts) FROM ohlcv WHERE symbol = $1", symbol
            )
            if row and row[0]:
                return row[0].replace(tzinfo=timezone.utc)
        return None

    async def sync_from_parquet(self, exchange: str, symbol: str) -> int:
        """Sync Parquet data to TimescaleDB incrementally.

        Reads Parquet data newer than the latest TimescaleDB entry
        and bulk-inserts it. Returns row count synced.
        """
        latest = await self.latest_ts(symbol)
        start_param = latest + timedelta(milliseconds=1) if latest else None
        df = read(exchange, symbol, start=start_param)
        if df.empty:
            logger.info("tsdb: sync %s — no new data", symbol)
            return 0
        rows = [
            (
                row["ts"].to_pydatetime(),
                symbol.upper(),
                row["open"], row["high"], row["low"],
                row["close"], row["volume"],
            )
            for _, row in df.iterrows()
        ]
        n = await self.bulk_insert(rows)
        logger.info("tsdb: synced %d rows for %s (from %s)", n, symbol, latest or "beginning")
        return n

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("tsdb: disconnected")
