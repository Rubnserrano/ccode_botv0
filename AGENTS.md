# ccode_botv0 — BTC Data Ingestion Engine

Paper trading only. Never touch real funds.

## Core Identity
BTC data ingestion engine: download historical OHLCV from Binance, store in Parquet + TimescaleDB, stream real-time via WebSocket.

## Safety Rules (MANDATORY — never skip)
1. **Paper only** — never reference real funds
2. **Propose before commit** — never commit/push without human approval
3. **Check git status before any change** — if dirty working tree, stop and report
4. **Run tests after every change** — never leave code untested
5. **Never delete data files** - `data/raw/*` are precious
6. **Always use `.venv/bin/python3`** — not system python
7. **Always `await asyncio.sleep()`** — never `time.sleep()`
8. **Max 3 retries on failure** — log and escalate after 3

## Workflow: Protección contra crash de terminal

1. **ANTES de empezar cualquier tarea o fase:**
   - Crear rama desde `main`
   - Crear PR en GitHub describiendo QUÉ se va a hacer y POR QUÉ
   - Push rama inmediatamente

2. **DURANTE la tarea:**
   - Commits periódicos (cada archivo o feature pequeño)
   - Push frecuente a la rama (al menos cada 2-3 cambios)

3. **SI el usuario dice "se me cerró la terminal" (RECUPERACIÓN):**
   - Lo PRIMERO: `git status`, `git log --oneline -5`, `git branch`
   - Verificar último commit y cambios sin commit
   - Reportar estado al usuario y preguntar por dónde seguir

## Project Structure
```
ccode_botv0/
├── docker-compose.yml          # TimescaleDB (2-pg16, puerto 5432)
├── requirements.txt            # httpx, websockets, pandas, pyarrow, asyncpg
├── data/
│   └── raw/binance/btcusdt/
│       ├── 2024-01.parquet     # ← particiones mensuales
│       └── ...
├── src/
│   ├── store.py                # DataStore Parquet (write/read/available_range)
│   ├── download.py             # Descarga histórica REST (incremental)
│   ├── resample.py             # Remuestreo 1m → 5m/15m/1h/1d
│   ├── feed.py                 # WebSocket: PriceBuffer, TradeBuffer, CandleBuffer
│   ├── tsdb.py                 # TimescaleDB: asyncpg, hypertable, continuous aggregates
│   └── main.py                 # Orquestador: download + sync + WS + persist
```

## Architecture
```
Binance REST ──→ download.py ──→ Parquet (store.py) ──→ Backtesting
                └─→ tsdb.sync_from_parquet() ──→ TimescaleDB (hypertable)

Binance WS ──→ feed.py (CandleBuffer) ──→ on_close callback
                                         ├─→ store.write() ──→ Parquet
                                         └─→ tsdb.insert_candle() ──→ TimescaleDB
```

## TimescaleDB Schema
```sql
hypertable: ohlcv (ts, symbol, open, high, low, close, volume)
  Chunk: 1 month, Unique: (ts, symbol)
  Continuous aggregates: ohlcv_5m, ohlcv_1h
```

## Quick Start
```bash
# 1. Start TimescaleDB
docker compose up -d

# 2. Run orchestrator (downloads history if empty, syncs to TSDB, starts WS)
TIMESCALE_DSN="postgres://ccode:ccode@localhost:5432/ccode" .venv/bin/python3 -m src.main

# 3. Download history manually (optional)
.venv/bin/python3 -m src.download --symbol BTCUSDT --days 862

# 4. Check TimescaleDB
.venv/bin/python3 -c "import asyncio; from src.tsdb import TimescaleDB; ..."
```

## Completed Phases
- Phase 0: Project scaffold (git init, deps, directories)
- Phase 1: DataStore (Parquet monthly partitions)
- Phase 2: Binance historical downloader (REST, incremental)
- Phase 3: OHLCV resampler (1m → 5m/15m/1h/1d)
- Phase 4: WebSocket feeds (ticker, trades, candles)
- Phase 5: Main orchestrator (download + WS + snapshot)
- Phase 6: TimescaleDB integration (asyncpg, hypertable, continuous aggregates, dual persistence)
