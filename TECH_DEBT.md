# Technical Debt — ccode-botv0 ATRS

Deuda técnica documentada para resolver en fases futuras.

## 🔴 Crítica

### 1. Dual Stores: `store.py` vs `ts_store.py`

**Problema**: `src/store.py` (original Parquet DataStore) y `src/ts_store.py` (Universal TS Store) tienen la misma función pero formatos incompatibles. `main.py` usa la vieja, `query.py` usa la nueva. Los datos se escriben en `data/raw/` (old) y `data/ts/` (new).

**Impacto**: Datos fragmentados. Un agente no sabe qué store consultar. Features builder lee de la vieja.

**Fix planeado**: Phase 26 — Unified Store. Migrar todo a TS Store, deprecar `store.py`.

### 2. Competing Research Loops

**Problema**: `src/research/loop.py` (genetic algorithm, pre-LLM) y `src/brain/orchestrator.py` (LLM-guided) ambos escriben al mismo leaderboard. Producen estrategias de diferente calidad y formato.

**Impacto**: Leaderboard mezcla estrategias de distinta calidad. Dificulta comparación.

**Fix planeado**: Phase 26 — deprecar `research/loop.py`. Unificar todo en brain.

### 3. Test Gap

**Problema**: Solo 2/50+ módulos tienen tests (31 tests total). Módulos críticos sin cobertura: backtesting engine, indicators calculator, strategy evaluator, brain orchestrator, paper trading, API.

**Impacto**: Cualquier refactor es riesgoso. Bugs en métricas (Sharpe, p-value) pasan desapercibidos.

**Fix planeado**: Phase 25 — Test Foundation.

### 4. API Key Hardcodeada

**Problema**: `research_loop.sh:17` tiene la API key de OpenRouter hardcodeada.

**Impacto**: Riesgo de seguridad si el script se comparte o se sube a GitHub.

**Fix**: Mover a `.env`. Fácil, hacer ya en Phase 25.

## 🟡 Importante

### 5. MCP Servers Huérfanos

**Problema**: `src/mcp_servers/*.py` (3 archivos, ~700 líneas) existen pero no están conectados a ningún host. `mcp_agent.py` reimplementa toda la lógica inline en vez de usar los servers.

**Impacto**: 5KB de dead code. Confunde a nuevos desarrolladores.

**Fix planeado**: Phase 31 — eliminar o rewire si se necesita post-POC.

### 6. Dual Data Trees

**Problema**: Los datos viven en `data/raw/` (old), `data/ts/` (new), `data/features/` (orphan), `data/external/` y `data/external_aligned/`. Sin jerarquía clara.

**Impacto**: Difícil para un agente descubrir qué datos existen sin leer código.

**Fix planeado**: Phase 26 — unificar en `data/ts/{source_type}/{asset_id}/{freq}/`.

### 7. Dependencias Implícitas

**Problema**: `numpy` y `streamlit` se importan en código pero no están en `requirements.txt`. `altair` tampoco.

**Impacto**: `pip install -r requirements.txt` no instala todo lo necesario.

**Fix**: Añadido en Phase 24.

### 8. Legacy `data/raw/binance/btcusdt/15m/data.parquet`

**Problema**: El API endpoint `POST /indicators/test` lee de `data/raw/binance/btcusdt/15m/data.parquet` (un archivo único) mientras el TS Store tiene particiones mensuales en `data/ts/market/binance:btcusdt/15m/`.

**Impacto**: El test de fórmulas usa datos posiblemente inconsistentes con el resto del sistema.

**Fix**: Phase 26 — migrar el endpoint a TS Store.

### 9. `POST /research/run` Documentado Pero No Implementado

**Problema**: El endpoint aparece en documentación y en `GET /` pero no tiene handler.

**Impacto**: Confunde a agentes que intentan usarlo.

**Fix**: documentación corregida en Phase 24. Implementación en Phase 25+.

## 🟢 Leve

### 10. `src/research/__init__.py` y `src/brain/__init__.py` Vacuos

**Problema**: Archivos `__init__.py` vacíos.

**Impacto**: Ninguno. Es práctica estándar.

**Fix**: No requiere acción.

### 11. Sin CI/CD

**Problema**: No hay GitHub Actions, Makefile, ni pyproject.toml.

**Impacto**: Tests requieren ejecución manual.

**Fix**: Post-POC si se necesita.

### 12. Sin Type Hints Completos

**Problema**: Muchos módulos carecen de type hints (especialmente `backtesting/`, `brain/`, `intelligence/`).

**Impacto**: Menos documentación implícita, harder para LLMs.

**Fix**: Progresivo. Idealmente en cada fase.
