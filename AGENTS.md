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
- Phase 7: WebSocket hardening (heartbeat, backoff), async download, unit tests (31)
- Phase 8: Dockerize app (Dockerfile + compose with TSDB + app services)
- Phase 9: OHLCV indicator calculator (RSI, MACD, EMA, VWAP, OBI, Heikin-Ashi, ATR, ADX, regime detection)
- Phase 10: Backtesting engine (costs, metrics, 5 strategies, CLI, walk-forward, evolution, genealogy)
- Phase 11: Strategy DSL prep (Strategy dataclass), walk-forward, evolution engine, genealogy
- Phase 12: Paper trading (PaperAccount, runner, decoupled Streamlit dashboard)
- Phase 13: Feature Store (precomputed indicators in Parquet, builder CLI, real-time updates)
- Phase 14: Parallel backtests CLI (--parallel flag, --tp, --sl flags)
- Phase 15: Strategy Engine — JSON-defined strategies via evaluator
- Phase 16: Automated research loop with progressive leaderboard
- Phase 17: LLM Brain — Strategist + Analyst via OpenRouter (deepseek-chat)

## How to Add a New Indicator (Feature)

Example: adding ATR (Average True Range) as indicator #14.

### Step 1: Add the function to `src/indicators/calculator.py`

```python
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility indicator."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()
```

Rules:
- Pure function: no I/O, no side effects, no global state
- Input: pd.Series or pd.DataFrame (OHLCV)
- Output: pd.Series or pd.DataFrame
- Return NaN for periods before the indicator has enough data
- All values must be float64 (pyarrow will infer this automatically)

### Step 2: Register it in `calc_all()`

```python
def calc_all(df: pd.DataFrame) -> pd.DataFrame:
    ...
    result["atr_14"] = calc_atr(result, 14)
    return result
```

### Step 3: Rebuild the Feature Store

```bash
# Build Docker image with the new code
docker compose build app
docker compose up -d app

# Rebuild ALL features with the new indicator
docker compose exec app python -m src.features.builder --symbol BTCUSDT --rebuild
```

- ``--rebuild`` forces full recalculation of all months.
- Without ``--rebuild``, only months after the latest feature are built (incremental).
- The builder reads raw Parquet, runs calc_all(), and writes to data/features/.
- Build time for 1.24M rows: ~40 seconds.

### Step 4: Verify

```bash
docker compose exec app python -c "
import pyarrow.parquet as pq
pf = pq.read_table('data/features/btcusdt/2026-05.parquet')
print('Columns:', pf.column_names)  # should include atr_14
print('Latest ATR:', pf.column('atr_14')[-1])
"
```

### What happens automatically after this

| Componente | Se actualiza solo? |
|-----------|-------------------|
| **Histórico** (Feature Store) | ✅ Sí — tras rebuild |
| **Real-time** (main.py) | ✅ Sí — calc_all() ya lo incluye |
| **Backtesting CLI** | ✅ Sí — lee de features directamente |
| **Paper trading** | ✅ Sí — usa calc_all() en on_candle_close |
| **Dashboard** | ❌ No — dashboard no toca features |
| **TimescaleDB** | ❌ No — TSDB solo guarda OHLCV, no indicadores |
| **Estrategias existentes** | ❌ No — no usan atr_14 hasta que las edites |

### To use the new indicator in a strategy

```python
def my_new_strategy(df: pd.DataFrame) -> pd.Series:
    sig = pd.Series(0, index=df.index)
    sig[df["atr_14"] > df["atr_14"].rolling(100).mean()] = 1  # high vol → BUY
    return sig
```

## Agent Integration — API Layer

The system exposes a FastAPI for agents to interact with programmatically.

### Starting the API

```bash
# Inside Docker:
docker compose exec app python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or add to docker-compose.yml as a service
```

### Agent Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | System overview + available endpoints |
| `GET` | `/indicators` | List all registered indicators |
| `POST` | `/indicators/test` | Test a formula without registering |
| `POST` | `/indicators` | Register a new indicator |
| `GET` | `/leaderboard?top=10&min_sharpe=0` | Research results |
| `GET` | `/strategies` | Saved strategies |
| `GET` | `/system` | System state (data volumes, paper trading) |

### Registering a New Indicator (without coding)

Any agent can create new indicators at runtime using formulas:

```bash
curl -X POST http://localhost:8000/indicators \
  -H "Content-Type: application/json" \
  -d '{
    "name": "momentum_12",
    "formula": "close - close.shift(12)",
    "description": "12-period momentum"
  }'
```

After registration, the indicator is immediately available in any strategy JSON:

```json
{"indicator": "momentum_12", "op": "gt", "value": 0}
```

### Formula Language

Formulas use Python syntax with access to:

**Available columns** (use by name directly):
`close`, `high`, `low`, `open`, `volume`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `ema_9`, `ema_21`, `ema_50`, `vwap`, `obi`, `atr_14`, `adx_14`, `regime`, `ha_open`, `ha_high`, `ha_low`, `ha_close`

**Available functions** (call by name):
| Function | Purpose | Example |
|----------|---------|---------|
| `sma(series, period)` | Simple Moving Average | `sma(close, 20)` |
| `ema(series, period)` | Exponential Moving Average | `ema(close, 50)` |
| `std(series, period)` | Rolling Std Dev | `std(close, 20)` |
| `shift(series, n)` | Shift back N periods | `shift(close, 1)` |
| `diff(series, n)` | Difference over N periods | `diff(close, 12)` |
| `pct_change(series, n)` | % change over N periods | `pct_change(close, 1)` |
| `max_roll(series, n)` | Rolling max | `max_roll(high, 50)` |
| `min_roll(series, n)` | Rolling min | `min_roll(low, 50)` |

**Math**: `abs(x)`, `max(a,b)`, `min(a,b)`, `sum(list)`, `round(x)`, `sqrt(x)`, `log(x)`, `log10(x)`

**Parameters**: `{param_name}` is replaced at registration time.

**Also valid**: pandas method syntax on columns — `close.shift(12)` is the same as `shift(close, 12)`.

### ⛔ RESTRICTIONS — What agents CANNOT do

These are blocked for security:
- ❌ **No `import` statements** — `"__import__('os')"` will fail
- ❌ **No file I/O** — `"open('/etc/passwd')"` will fail
- ❌ **No `exec` / `eval`** — nested eval is not allowed
- ❌ **No network calls** — only math and data operations
- ❌ **No variable assignment** — `x = close + 1` will fail (use expressions directly)

Formulas are **mathematical expressions only**. They operate on existing columns.
They cannot create loops, write files, or access the internet.

### Examples

```python
# Simple momentum
"close - close.shift(12)"

# Bollinger-like: close vs SMA
"(close - sma(close, 20)) / std(close, 20)"

# Volume spike detection
"volume > sma(volume, 50) * 1.5"

# Normalized price position
"(close - min_roll(low, 50)) / (max_roll(high, 50) - min_roll(low, 50)) * 100"

# Custom RSI-like (with parameter)
"ema(gain, {period}) / ema(loss, {period})"
# where gain/loss are precomputed in the Feature Store
```

### Testing a Formula First

```bash
curl -X POST http://localhost:8000/indicators/test \
  -H "Content-Type: application/json" \
  -d '{"formula": "volume > sma(volume, 50) * 1.5", "sample_limit": 5}'
```

Returns sample values on recent data so the agent can verify the formula works.

**Important:** The API server needs to be running separately:
```bash
docker compose exec -d app python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Adding a New Data Source

Any agent can register a new external data source at runtime.
A registered source costs ~1KB on disk. It is NOT fetched until:
  a) A strategy references its column
  b) An agent calls `POST /data-sources/{name}/fetch`

```
POST /data-sources (register)
  ↓
data/sources/{name}.json  (1KB, 0 CPU)
  ↓ (first use triggers fetch)
data/external/{name}/{YYYY-MM}.parquet  (raw data)
  ↓ (aligner)
data/external_aligned/{timeframe}/{name}.parquet  (regular timestamps)
  ↓ (feature rebuild)
data/features/btcusdt/{YYYY-MM}.parquet  (column available)
  ↓ (strategy uses it)
{"indicator": "vix", "op": "lt", "value": 25}  ← auto-discovered
```

#### Registering a Data Source

```bash
curl -X POST http://localhost:8000/data-sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fred_vix",
    "url": "https://api.stlouisfed.org/fred/series/observations",
    "description": "VIX volatility index",
    "params": {"series_id": "VIXCLS", "file_type": "json"},
    "api_key": "your_fred_api_key",
    "schedule": "1h",
    "parse": {
      "type": "json",
      "timestamp_field": "observations[].date",
      "value_field": "observations[].value",
      "value_transform": "float"
    },
    "columns": {"value": "vix"},
    "align": {
      "method": "ffill",
      "decay_periods": 0,
      "target_timeframes": ["15m", "1h"]
    }
  }'
```

#### Definition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Unique identifier (alphanumeric + underscores) |
| `url` | ✅ | HTTPS endpoint |
| `schedule` | ✅ | Fetch interval: "5m", "1h", "1d" |
| `parse.type` | ✅ | "json" or "csv" |
| `parse.timestamp_field` | ✅ | Dotted path to timestamp field (supports arrays: `obs[].date`) |
| `parse.value_field` | ✅ | Dotted path to value field |
| `columns` | ✅ | Map `{"value": "vix"}` — column name in Feature Store |
| `align.method` | ✅ | "ffill", "interpolate", "sum", "avg" |
| `api_key` | ❌ | API key (stored in JSON, unencrypted for POC) |

#### Fetching and Using

```bash
# Force fetch immediately
curl -X POST http://localhost:8000/data-sources/fred_vix/fetch

# The column is now available in any strategy
```

```json
{"indicator": "vix", "op": "lt", "value": 25}
```

#### Response on Fetch

```json
{
  "status": "fetched",
  "name": "fred_vix",
  "rows": 365,
  "columns": ["vix"],
  "timeframes": ["15m", "1h"]
}
```

#### Edge Cases

| Scenario | Behavior |
|----------|----------|
| **No historical data** (new source) | Only data from first fetch. Backtest limited to that range |
| **Payload > 1MB** | Truncates to last 365 days |
| **Irregular timestamps** (news) | Aligner does ffill with optional decay |
| **API down** | Column has NaN. Strategies don't generate signals |
| **Duplicate registration** | 409 Conflict — use DELETE first |
| **News/sentiment** | Multiple events per vela → averaged |

### Running Research via API

```bash
curl -X POST http://localhost:8000/research/run \
  -H "Content-Type: application/json" \
  -d '{"n_strategies": 10, "days": 365, "resample": "15m", "rounds": 2}'
```

Returns immediately (async). Progress is visible in the logs:
```bash
docker compose logs app -f
```

Or check the leaderboard:
```bash
curl http://localhost:8000/leaderboard?top=5
```

### Agent Workflow (recommended)

```
1. GET /system       → understand current state (data, indicators)
2. POST /indicators/test → test a new indicator formula
3. POST /indicators  → register the indicator
4. POST /research/run → run research with the new indicator
5. GET /leaderboard  → see results
6. GET /strategies   → inspect successful strategies
```
