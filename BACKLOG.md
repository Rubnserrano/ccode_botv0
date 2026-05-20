# Backlog — ccode-botv0 ATRS

## Fase 25 — Test Foundation

**Objetivo**: Cobertura de tests en todos los módulos críticos. Sin tests no puedes refactorizar nada sin riesgo.

| ID | Tarea | Prioridad | Esfuerzo |
|---|---|---|---|
| 25.1 | Tests para `src/indicators/calculator.py` (17 tests, uno por indicador) | 🔴 | 1d |
| 25.2 | Tests para `src/strategy_engine/` (schema, evaluator, registry, generic_calculator, store — ~15 tests) | 🔴 | 1.5d |
| 25.3 | Tests para `src/backtesting/engine.py` (~10 tests: signals, TP/SL, trade simulation, save) | 🔴 | 1d |
| 25.4 | Tests para `src/backtesting/metrics.py` (~8 tests: Sharpe, Sortino, p-value, quality gates) | 🔴 | 0.5d |
| 25.5 | Tests para `src/paper/account.py` + `runner.py` (~8 tests: open/close, equity, state, exits) | 🔴 | 1d |
| 25.6 | Tests para `src/ts_store.py` + `ts_aligner.py` (~10 tests: write/read/dedup/align) | 🟡 | 1d |
| 25.7 | Tests para `src/intelligence/fetcher.py` + `aligner.py` (~8 tests mocked HTTP) | 🟡 | 1d |
| 25.8 | Tests para `src/brain/orchestrator.py` + `strategist.py` (~10 tests mocked LLM) | 🟡 | 1d |
| 25.9 | Tests para `api/app.py` (~8 tests httpx) | 🟡 | 0.5d |
| 25.10 | Tests para `src/tsdb.py` (~5 tests mocked asyncpg) | 🟢 | 0.5d |
| 25.11 | Tests para `src/download.py` (~5 tests mocked httpx) | 🟢 | 0.5d |
| 25.12 | Mover API key de `research_loop.sh` a `.env` | 🔴 | 0.1d |

**🔴 crítico / 🟡 importante / 🟢 nice to have**

---

## Fase 26 — Unified Store

**Objetivo**: Resolver dualidades arquitectónicas que fragmentan los datos.

| ID | Tarea | Detalle |
|---|---|---|
| 26.1 | Migrar `main.py` de `store.py` a `ts_store.py` | Unificar escritura de datos en caliente |
| 26.2 | Migrar datos de `data/raw/` → `data/ts/market/binance:btcusdt/` | Unificar árbol de datos históricos |
| 26.3 | Migrar `data/features/` a `data/ts/market/binance:btcusdt/features/` | Features como una frecuencia más del TS Store |
| 26.4 | Actualizar `src/features/builder.py` para usar `ts_store` | Pipeline completo unificado |
| 26.5 | Actualizar `src/query.py` como único punto de consulta | Unificar queries |
| 26.6 | Deprecar `src/store.py` (marcar legacy, no eliminar hasta Fase 31) | |
| 26.7 | Deprecar `src/research/loop.py` (genético) | Unificar research en brain |
| 26.8 | Agregar metadata a `ts_catalog.py` | última_actualización, row_count, checksum, fuentes_que_lo_usan |
| 26.9 | Exponer `GET /catalog` via API | Para que agentes descubran datos disponibles |

**Riesgo**: Migración de datos. Hacer en dos pasos: (1) escribir en ambos stores, (2) cortar al nuevo.

---

## Fase 27 — True Memory

**Objetivo**: Pasar de fingerprint dedup a memoria episódica + semántica persistente.

| ID | Tarea | Detalle |
|---|---|---|
| 27.1 | Crear `src/memory/episodic.py` | Cada intento: estrategia, resultado, métricas OOS, condiciones de mercado, timestamp. SQLite + Parquet |
| 27.2 | Crear `src/memory/semantic.py` | Hechos extraídos con embeddings. Búsqueda por similitud coseno |
| 27.3 | Crear `src/memory/procedural.py` | Templates de estrategias que funcionaron, parámetros óptimos por régimen |
| 27.4 | Modificar `analyst.py` para extraer "lessons learned" | → guardar en memoria semántica |
| 27.5 | Modificar `strategist.py` para consultar memoria | "qué funcionó en régimen similar?" |
| 27.6 | Crear `src/memory/store.py` | Persistencia centralizada |
| 27.7 | Añadir `GET /memory` endpoint | Para que agentes consulten memoria |

**Dependencia externa**: `sentence-transformers` o API de embeddings.

---

## Fase 28 — Multi-step Reasoning

**Objetivo**: Pipeline reflexivo con iteración Strategist ↔ Analyst ↔ Reflexión.

| ID | Tarea | Detalle |
|---|---|---|
| 28.1 | Añadir campo `reasoning` a strategist | "Por qué esta estrategia debería funcionar" |
| 28.2 | analyst devuelve `refined_hypothesis` + `suggested_modifications` | No solo evaluar, proponer |
| 28.3 | Crear `src/brain/reflector.py` | Compara hipótesis vs resultado, detecta sesgos |
| 28.4 | Loop de refinamiento en orchestrator | Max 3 iteraciones |
| 28.5 | Estado interno en mcp_agent.py | Tool loop con memoria de contexto |
| 28.6 | Compute-on-demand + LRU cache para parámetros infinitos | Cache 50 items, persistir si se usa 3+ veces |
| 28.7 | Meta-evaluación en analyst | "¿Estas N estrategias son genuinamente diferentes?" |

---

## Fase 29 — Distributed Specialization

**Objetivo**: Workers paralelos y especialización de modelos.

| ID | Tarea | Detalle |
|---|---|---|
| 29.1 | Diseñar `src/core/message_bus.py` | Cola asyncio.Queue o SQLite-based |
| 29.2 | Crear `ResearchWorker` base class | Worker independiente que corre backtests |
| 29.3 | Research pool con N workers | Paralelo con asyncio.gather |
| 29.4 | Cheap pre-filter | Heurística o modelo pequeño antes de backtest completo |
| 29.5 | Model specialization | Strategist = deepseek-chat, Analyst = claude-haiku |
| 29.6 | Separar data_agent de research_pool en main.py | Data corre siempre, research bajo demanda |
| 29.7 | Refactorizar main.py como launcher | Lanza agentes como corrutinas independientes |
| 29.8 | `GET /brain-context` endpoint | Vista unificada ~2KB para agentes |

---

## Fase 30 — Real-time Cognitive Loop

**Objetivo**: El brain reacciona a datos en vivo.

| ID | Tarea | Detalle |
|---|---|---|
| 30.1 | Crear `src/brain/realtime.py` | Escucha CandleBuffer.on_close |
| 30.2 | Regime change detection | Si cambia el régimen, lanza research acotado |
| 30.3 | Paper trading feedback | Si drawdown > 10%, brain investiga |
| 30.4 | Continuous learning | Cada N velas: "mis señales pasadas fueron correctas?" |
| 30.5 | Alertas cognitivas via Telegram | "Régimen cambió, sin estrategias activas" |
| 30.6 | Background scheduler en main.py | Research loop permanente |

---

## Fase 31 — Meta-cognition & Portfolio

**Objetivo**: El sistema se evalúa a sí mismo y optimiza portfolios.

| ID | Tarea | Detalle |
|---|---|---|
| 31.1 | Portfolio optimizer | Combinaciones no correlacionadas del leaderboard |
| 31.2 | Uncertainty quantification | Calibrar confidence del Analyst |
| 31.3 | Curriculum learning | Exploración progresiva: simple → complejo |
| 31.4 | Self-evaluation | "Mi Sharpe promedio mejora?" → ajusta hiperparámetros |
| 31.5 | Strategy archetypes | Clasificar por arquetipo, balancear exploración |
| 31.6 | Knowledge distillation | "Strategy handbook" desde memoria semántica |
| 31.7 | Limpieza final | Eliminar store.py, research/loop.py, MCP servers legacy |

---

## Fase 32+ — External Intelligence

**Objetivo**: El sistema extrae información del mundo real.

| ID | Tarea | Prioridad POC |
|---|---|---|
| 32.1 | MCP Client genérico (`src/intelligence/mcp_source.py`) | 🟡 |
| 32.2 | RSS/News ingestion | 🟡 |
| 32.3 | Twitter/X ingestion | 🔴 |
| 32.4 | Paper scraper (arxiv, SSRN) | 🟢 |
| 32.5 | External signal ingestion (señales de otros bots) | 🟡 |
| 32.6 | Metadata catalog mejorado (calidad, frescura) | 🟡 |
| 32.7 | Data quality pipeline (checks, alertas) | 🔴 |
| 32.8 | Cola de ingesta para 100+ fuentes | 🟢 |

---

## Preguntas Abiertas

| # | Pregunta | Propuesta | Decidido? |
|---|---|---|---|
| Q1 | MCP como protocolo interno? | ❌ No para POC | ✅ Sí |
| Q2 | Indicadores on the fly o precomputados? | On the fly + LRU cache | ✅ Sí |
| Q3 | Cola de mensajes? | SQLite + asyncio.Queue (Redis post-POC) | 🟡 Pendiente |
| Q4 | Feature engineering automático? | Automático con supervisión | 🟡 Pendiente |
| Q5 | Feature store versionado? | Sí, pero post-POC | 🟡 Pendiente |
| Q6 | Portfolio optimization en POC? | No | ✅ Sí |
| Q7 | Cuentas fondeadas? | Post-POC | ✅ Sí |
| Q8 | Eliminar legacy? | Phase 31 | ✅ Sí |
| Q9 | Embeddings locales o API? | Locales (sentence-transformers) | 🟡 Pendiente |
| Q10 | CI/CD? | No para POC | ✅ Sí |
