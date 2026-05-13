# ccode-botv0 — Autonomous Trading Research System (ATRS)

Paper trading only. Never touch real funds.

## 1. Core Identity

Sistema de investigación de trading algorítmico con paper trading, backtesting, y agente cognitivo auto-mejorable. Orientado a rentabilidad mediante exploración sistemática de estrategias.

## 2. Safety Rules (MANDATORY — never skip)

1. **Paper only** — never reference real funds
2. **Propose before commit** — never commit/push without human approval
3. **Check git status before any change** — if dirty working tree, stop and report
4. **Run tests after every change** — never leave code untested
5. **Never delete data files** — `data/raw/*`, `data/ts/*`, `data/features/*` are precious
6. **Always use `.venv/bin/python3`** — not system python
7. **Always `await asyncio.sleep()`** — never `time.sleep()`
8. **Max 3 retries on failure** — log and escalate after 3

## 3. Architecture Overview

### 3.1 General Vision

```
                     ┌──────────────────────────────┐
                     │     Message Bus / Cola        │
                     │  (asyncio.Queue → Redis)      │
                     └──────┬──────────────┬─────────┘
                            │              │
            ┌───────────────┘              └───────────────┐
            ▼                                              ▼
   ┌──────────────────┐                          ┌──────────────────┐
   │  Data Agent       │                          │  Research Pool    │
   │  (ingest, fuentes)│                          │  (N workers)      │
   │                   │                          │                   │
   │  Binance REST/WS  │                          │  Worker 1 (LLM)   │
   │  MCP sources      │                          │  Worker 2 (LLM)   │
   │  RSS/Twitter    │                          │  Worker 3 (gen.) │
   │  Scheduler      │                          │  Coordinator      │
   └────────┬─────────┘                          └────────┬──────────┘
            │                                              │
            ▼                                              ▼
   ┌──────────────────┐                          ┌──────────────────┐
   │  TS Store         │                          │  Leaderboard     │
   │  (data central)   │                          │  + Memory        │
   └────────┬─────────┘                          └────────┬──────────┘
            │                                              │
            ▼                                              ▼
   ┌──────────────────┐                          ┌──────────────────┐
   │  Monitor Agent    │                          │  Paper Agent      │
   │  (métricas,       │                          │  (ejecución)      │
   │   alertas)        │                          │                   │
   └──────────────────┘                          └──────────────────┘
                            ▲
                            │
                      ┌─────┴──────┐
                      │  API Layer  │
                      │  (FastAPI)  │
                      └────────────┘
```

### 3.2 POC Scope

La POC simplifica la visión general a un sistema monoproceso con componentes paralelizables:

```
main.py ─── lanza en paralelo (asyncio.gather):
  ├── data_agent:    WS + ingestion → TS Store
  ├── research_pool: N workers → leaderboard + memoria
  ├── paper_runner:  estrategias activas → equity
  └── API:           FastAPI para agentes

Comunicación: archivos compartidos (TS Store, leaderboard JSONL, memoria SQLite)
No hay cola de mensajes aún — los workers son independientes.
```

La POC termina cuando el sistema es capaz de:
- Generar estrategias de forma autónoma (vía LLM)
- Validarlas con train/test split + walk-forward + OOS
- Ejecutarlas en paper trading
- Demostrar (o refutar) su rentabilidad
- Un agente LLM puede entender el estado completo en 1-2 llamadas API

## 4. Project Structure

```
src/
├── __init__.py
├── download.py           # Binance REST historical download (legacy)
├── feed.py               # WebSocket: PriceBuffer, TradeBuffer, CandleBuffer
├── main.py               # Orchestrator: data ingestion + paper + agents
├── query.py              # Unified multi-asset query engine
├── resample.py           # OHLCV resampling 1m → 5m/15m/1h/1d
├── store.py              # Raw Parquet DataStore (legacy — usar ts_store.py)
├── ts_aligner.py         # Universal time-series aligner
├── ts_catalog.py         # Asset catalog with metadata
├── ts_store.py           # Universal TS Store (Parquet partitions by source_type/asset/freq)
├── tsdb.py               # TimescaleDB async client (legacy)
│
├── backtesting/          # Backtesting engine
│   ├── engine.py         # Vectorized backtest with TP/SL
│   ├── cli.py            # CLI with --parallel, --tp, --sl
│   ├── costs.py          # Trade cost model
│   ├── metrics.py        # Sharpe, Sortino, MaxDD, p-value
│   ├── strategy.py       # Strategy dataclass
│   ├── strategies.py     # 5 built-in strategies
│   ├── walk_forward.py   # Walk-forward analysis
│   ├── evolution.py      # Genetic algorithm optimizer
│   └── genealogy.py      # Strategy lineage tracking
│
├── brain/                # LLM Cognitive Layer
│   ├── strategist.py     # Generates strategies via OpenRouter
│   ├── analyst.py        # Evaluates backtest results
│   ├── orchestrator.py   # Full research loop (train/test + OOS)
│   ├── mcp_agent.py      # Tool-calling agent (no MCP protocol)
│   ├── memory.py         # Fingerprint deduplication
│   ├── llm_client.py     # OpenRouter wrapper with fallback chain
│   └── prompts.py        # Strategist + Analyst system prompts
│
├── features/             # Feature Store
│   ├── builder.py        # CLI: rebuild features from raw data
│   └── store.py          # Parquet read/write for features
│
├── indicators/           # Technical indicators
│   └── calculator.py     # calc_all(): RSI, MACD, EMA, VWAP, OBI, HA, ATR, ADX, regime
│
├── intelligence/         # External data sources
│   ├── models.py         # DataSource dataclass
│   ├── registry.py       # CRUD for source definitions
│   ├── fetcher.py        # HTTP fetch + parse (JSON/CSV)
│   ├── aligner.py        # Align to standard timeframes
│   └── orchestrator.py   # Background scheduler
│
├── mcp_servers/          # MCP servers (unused — see TECH_DEBT.md)
│   ├── query_server.py
│   ├── backtest_server.py
│   └── memory_server.py
│
├── notification/         # Telegram notifications
│   └── telegram.py
│
├── paper/                # Paper trading
│   ├── account.py        # PaperAccount (open/close/equity)
│   ├── runner.py         # PaperRunner (CandleBuffer → signals)
│   └── state.py          # State persistence
│
├── research/             # Research loop (legacy — usar brain/)
│   ├── leaderboard.py    # Leaderboard persistence (JSONL + Parquet)
│   └── loop.py           # Genetic algorithm loop (pre-LLM)
│
└── strategy_engine/      # JSON-defined strategies
    ├── schema.py         # StrategyDef, Condition, ExitRules
    ├── evaluator.py      # Evaluate strategies against DataFrame
    ├── registry.py       # INDICATOR_REGISTRY + ROLLING_REGISTRY
    ├── generic_calculator.py  # Runtime formula evaluator
    └── store.py          # Save/load strategy JSON files

api/
└── app.py                # FastAPI: agent interaction endpoints

dashboard/
└── app.py                # Streamlit dashboard (paper trading view)

data/
├── ts/                   # Universal TS Store (market, sentiment, onchain, derivatives,...)
│   ├── catalog.parquet   # Asset catalog
│   └── {source_type}/{asset_id}/{frequency}/{YYYY-MM}.parquet
├── raw/                  # Legacy raw storage (being migrated)
├── features/             # Precomputed indicators
├── external/             # Raw external data source fetches
├── external_aligned/     # Time-aligned external data
├── sources/              # Data source definitions (JSON)
├── strategies/           # Saved strategy definitions (JSON)
├── parquet/              # Research artifacts (leaderboard, genealogy, audit)
└── paper_state.json      # Paper trading state

tests/
├── test_store.py         # DataStore tests (11 tests)
├── test_feed.py          # WebSocket/buffer tests (20 tests)
└── conftest.py
```

## 5. Completed Phases

| Phase | What | Notes |
|---|---|---|
| 0 | Project scaffold | git init, deps, directories |
| 1 | DataStore | Parquet monthly partitions |
| 2 | Binance downloader | REST, incremental |
| 3 | OHLCV resampler | 1m → 5m/15m/1h/1d |
| 4 | WebSocket feeds | ticker, trades, candles |
| 5 | Main orchestrator | download + WS + snapshot |
| 6 | TimescaleDB | asyncpg, hypertable, continuous aggregates |
| 7 | Hardening | heartbeat, backoff, tests (31) |
| 8 | Dockerize | Dockerfile + compose |
| 9 | Indicators calculator | RSI, MACD, EMA, VWAP, OBI, HA, ATR, ADX, regime |
| 10 | Backtesting engine | costs, metrics, 5 strategies, CLI, walk-forward, evolution, genealogy |
| 11 | Strategy DSL | Strategy dataclass, walk-forward, evolution, genealogy |
| 12 | Paper trading | PaperAccount, runner, Streamlit dashboard |
| 13 | Feature Store | Precomputed indicators, builder CLI |
| 14 | Parallel backtests | --parallel, --tp, --sl flags |
| 15 | Strategy Engine | JSON-defined strategies via evaluator |
| 16 | Research loop | Genetic algorithm + leaderboard |
| 17 | LLM Brain | Strategist + Analyst via OpenRouter |
| 18 | Agent API | FastAPI endpoints |
| 19 | External data sources | VIX, Fear & Greed, custom sources |
| 20 | Overnight research | Automated night loop + Telegram |
| 21 | Overnight fixes | Rate limit, backoff, signaling |
| 22 | Train/test split | OOS validation, strategy templates |
| 23 | Universal TS Store + MCP Servers | Unified time-series storage, MCP servers |

### Known Inefficiencies (detected post-Phase 23)

| Inefficiency | Location | Impact | Planned Fix |
|---|---|---|---|
| **Dual stores** | `store.py` (old) vs `ts_store.py` (new) | Data scattered across `data/raw/` and `data/ts/`. `main.py` uses old, `query.py` uses new | Phase 26: Unified Store |
| **Competing research loops** | `src/research/loop.py` (genetic) vs `src/brain/orchestrator.py` (LLM) | Both write to same leaderboard with different quality levels | Phase 26: deprecate genetic loop |
| **MCP servers unused** | `src/mcp_servers/*.py` | 5KB dead code. `mcp_agent.py` reimplements tools inline | Phase 31: remove or rewire |
| **Test gap** | Only 2/50+ modules tested | 31 tests cover only `store.py` and `feed.py` | Phase 25: Test Foundation |
| **Dual data trees** | `data/raw/` (old) + `data/ts/` (new) + `data/features/` (orphan) | Fragmented storage, hard to discover | Phase 26: migrate to TS Store |
| **API key in shell script** | `research_loop.sh` line 17 | Hardcoded key (security risk) | Phase 25: move to `.env` |

## 6. Cognitive Architecture

### Current (Phase 23)

```
Strategist (deepseek-chat)
  │  "genera N estrategias JSON"
  ▼
Fast filter (30d de train data)
  │
  ▼
Full backtest + walk-forward (train)
  │
  ▼
OOS validation (hold-out test)
  │
  ▼
Analyst (deepseek-chat)
  │  "evalúa resultados"
  ▼
Leaderboard + Telegram
```

Limitaciones actuales:
- **Memoria**: solo fingerprint dedup (coincidencia exacta de strings). No hay memoria episódica ni semántica.
- **Razonamiento**: single-step LLM call. No hay chain-of-thought ni reflexión.
- **Especialización**: un mismo modelo (deepseek-chat) para todo, diferenciado solo por prompt.
- **Aprendizaje cross-session**: cada ejecución empieza desde cero (excepto el leaderboard JSONL).

### Target (post-Phase 31)

```
1. Strategist genera hipótesis + reasoning explícito
2. Backtest rápido (fast filter 30d)
3. Si pasa → full backtest + walk-forward + OOS
4. Analyst evalúa + extrae lecciones → memoria semántica
5. Reflexión: compara hipótesis vs resultado real
6. Refinamiento: strategist recibe feedback, itera (max 3)
7. Meta-evaluación: "¿Estas estrategias son genuinamente diferentes?"
8. Las lecciones se consolidan en memoria episódica + semántica
9. El strategist consulta memoria antes de generar nuevas ideas
```

Ver [BACKLOG.md](./BACKLOG.md) para el roadmap completo.

## 7. How to Run (POC-focused)

### Start data ingestion + paper trading

```bash
docker compose up -d                                          # TimescaleDB
TIMESCALE_DSN="postgres://ccode:ccode@localhost:5432/ccode"  \
  .venv/bin/python3 -m src.main                               # Ingest + paper
```

### Run research (LLM brain)

```bash
.venv/bin/python3 -m src.brain.orchestrator \
  --rounds 2 --n-strategies 5 --days 365 --resample 15m
```

Resultados en `data/parquet/research/leaderboard.parquet` y Telegram si está configurado.

### Run MCP agent (tool-calling)

```bash
.venv/bin/python3 -m src.brain.mcp_agent --days 180 --frequency 15m
```

Modo interactivo: escribe queries en lenguaje natural y el agente usa tools para responder.

### Start API for agents

```bash
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Run specific backtest

```bash
.venv/bin/python3 -m src.backtesting.cli \
  --strategy ema_trend --resample 15m --walk-forward --parallel
```

### Rebuild feature store (after adding indicators)

```bash
docker compose exec app python -m src.features.builder --symbol BTCUSDT --rebuild
```

## 8. Agent Integration — API Layer

The system exposes a FastAPI for agents to interact with programmatically.

### Available Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | System overview + available endpoints |
| `GET` | `/indicators` | List all registered indicators |
| `POST` | `/indicators/test` | Test a formula without registering |
| `POST` | `/indicators` | Register a new indicator |
| `GET` | `/leaderboard?top=10&min_sharpe=0` | Research results |
| `GET` | `/strategies` | Saved strategies |
| `GET` | `/system` | System state (data volumes, paper trading) |
| `POST` | `/data-sources` | Register external data source |
| `GET` | `/data-sources` | List registered sources |
| `GET` | `/data-sources/{name}` | Get source details |
| `DELETE` | `/data-sources/{name}` | Delete source + data |
| `POST` | `/data-sources/{name}/fetch` | Force immediate fetch |

**Próximamente**: `POST /research/run` — lanzar research loop via API.

### Agent Workflow (recommended)

```
1. GET /system       → understand current state (data, indicators, paper)
2. POST /indicators/test → test a new indicator formula
3. POST /indicators  → register the indicator
4. GET /leaderboard  → see existing results
5. GET /strategies   → inspect successful strategies
```

## 9. Formula Language & Indicators

Formulas use Python syntax with access to available columns and helper functions.

**Available columns** (use by name directly):
`close`, `high`, `low`, `open`, `volume`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `ema_9`, `ema_21`, `ema_50`, `vwap`, `obi`, `atr_14`, `adx_14`, `regime`, `ha_open`, `ha_high`, `ha_low`, `ha_close`

**Available functions**:

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

### ⛔ Restricted

- ❌ No `import` statements
- ❌ No file I/O
- ❌ No `exec` / `eval`
- ❌ No network calls
- ❌ No variable assignment

Formulas are **mathematical expressions only**. They operate on existing columns.

### Registering a New Indicator

```bash
curl -X POST http://localhost:8000/indicators \
  -H "Content-Type: application/json" \
  -d '{"name": "momentum_12", "formula": "close - close.shift(12)", "description": "12-period momentum"}'
```

After registration, the indicator is immediately available in any strategy:

```json
{"indicator": "momentum_12", "op": "gt", "value": 0}
```

### Testing a Formula First

```bash
curl -X POST http://localhost:8000/indicators/test \
  -H "Content-Type: application/json" \
  -d '{"formula": "volume > sma(volume, 50) * 1.5", "sample_limit": 5}'
```

## 10. Data Sources

### Architecture

```
POST /data-sources (register)
  ↓
data/sources/{name}.json  (definition, ~1KB)
  ↓ (on first use or manual fetch)
data/external/{name}/{YYYY-MM}.parquet  (raw data)
  ↓ (aligner)
data/external_aligned/{timeframe}/{name}.parquet  (regular timestamps)
  ↓ (feature rebuild)
data/features/btcusdt/{YYYY-MM}.parquet  (column available)
  ↓ (strategy uses it)
{"indicator": "vix", "op": "lt", "value": 25}
```

### Registering a Data Source

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
      "target_timeframes": ["15m", "1h"]
    }
  }'
```

### Definition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Unique identifier (alphanumeric + underscores) |
| `url` | ✅ | HTTPS endpoint |
| `schedule` | ✅ | Fetch interval: "5m", "1h", "1d" |
| `parse.type` | ✅ | "json" or "csv" |
| `parse.timestamp_field` | ✅ | Dotted path to timestamp field |
| `parse.value_field` | ✅ | Dotted path to value field |
| `columns` | ✅ | Map `{"value": "column_name"}` |
| `align.method` | ✅ | "ffill", "interpolate", "sum", "avg" |
| `api_key` | ❌ | API key (stored in JSON, unencrypted for POC) |

### Concrete Example: Fear & Greed Index (already registered)

```bash
# View the registered source
curl http://localhost:8000/data-sources/fear_greed

# Force a fresh fetch
curl -X POST http://localhost:8000/data-sources/fear_greed/fetch

# Use in any strategy
```
```json
{"indicator": "fear_greed", "op": "lt", "value": 25}
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| **No historical data** (new source) | Only data from first fetch. Backtest limited to that range |
| **Payload > 1MB** | Truncates to last 365 days |
| **Irregular timestamps** | Aligner does ffill with optional decay |
| **API down** | Column has NaN. Strategies don't generate signals |
| **Duplicate registration** | 409 Conflict — use DELETE first |

## 11. Open Questions

Decisiones pendientes que afectan al diseño futuro. Cualquier agente debe conocerlas.

| # | Pregunta | Decisión Actual |
|---|---|---|
| Q1 | MCP como protocolo interno? | ❌ No para la POC. Tool calling directo. |
| Q2 | Computar indicadores on the fly o precomputar todo? | On the fly + LRU cache. Persistir solo si se usa 3+ veces. |
| Q3 | Cola de mensajes para distribución? | SQLite + asyncio.Queue para POC. Redis si multi-host. |
| Q4 | Feature engineering automático o controlado? | Automático con sugerencia humana. |
| Q5 | Feature store versionado? | Pendiente. Mientras, overwrite. |
| Q6 | Portfolio optimization en POC? | No. Apuntado para Phase 31. |
| Q7 | Evaluación de cuentas fondeadas? | No tocar hasta post-POC. |
| Q8 | Eliminar legacy (store.py, research/loop.py, MCP servers)? | Sí, en Phase 31. |
| Q9 | Embeddings locales o vía API? | Locales con sentence-transformers. |
| Q10 | Tests en CI/CD? | No para la POC. Tests manuales via pytest. |

## 12. Roadmap

| Fase | Qué | Días | Dependencia |
|---|---|---|---|
| **24** | Foundation Fix (docs, identidad, backlog) | 1-2 | — |
| **25** | Test Foundation (cobertura >70% crítico) | 7-10 | 24 |
| **26** | Unified Store (resolver dualidades) | 4-5 | 25 |
| **27** | True Memory (episódica + semántica) | 5-6 | 25, 26 |
| **28** | Multi-step Reasoning (ciclo reflexivo) | 4-5 | 27 |
| **29** | Distributed Specialization (workers paralelos) | 5-6 | 28 |
| **30** | Real-time Cognitive Loop (brain + live data) | 4-5 | 29 |
| **31** | Meta-cognition & Portfolio | 5-7 | 30 |
| **32+** | External Intelligence (RSS, MCP sources, papers) | 6-8 | 31 |

Ver [BACKLOG.md](./BACKLOG.md) para tareas detalladas por fase.
