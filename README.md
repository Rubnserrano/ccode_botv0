# ccode-botv0 — Autonomous Trading Research System (ATRS)

Paper trading. Strategy research. Self-improving agents.

```
Data Layer  →  Research Layer  →  Cognitive Layer
```

Descarga histórica de Binance, WebSocket en vivo, indicadores técnicos, backtesting con walk-forward + OOS, paper trading, y un agente LLM que genera, evalúa y refina estrategias de forma autónoma.

**Documentación completa**: [`AGENTS.md`](AGENTS.md)
**Backlog y roadmap**: [`BACKLOG.md`](BACKLOG.md)
**Deuda técnica**: [`TECH_DEBT.md`](TECH_DEBT.md)

## Quick Start

```bash
.venv/bin/python3 -m src.main                                          # ingestión + paper
.venv/bin/python3 -m src.brain.orchestrator --rounds 2 --n-strategies 5  # research loop
.venv/bin/uvicorn api.app:app --port 8000                                # API para agentes
```
