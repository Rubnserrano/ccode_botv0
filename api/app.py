"""API Layer for agent interaction — FastAPI endpoints.

Agents can:
  - Register new indicators (formula-based)
  - Register new data sources
  - Launch research runs
  - Query results

Usage:
    .venv/bin/uvicorn api.app:app --port 8000
    # OR via docker:
    docker compose exec app python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.strategy_engine.generic_calculator import calc_formula, register_dynamic_indicator
from src.strategy_engine.registry import INDICATOR_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="ccode_botv0 Agent API", version="0.1.0")


# ─── Schemas ─────────────────────────────────────────────────────────────────

class IndicatorRequest(BaseModel):
    name: str
    formula: str
    params: dict | None = None
    description: str = ""


class DataSourceRequest(BaseModel):
    name: str
    url: str
    schedule: str = "1h"
    column_mapping: dict[str, str] = {}
    description: str = ""


class ResearchRequest(BaseModel):
    n_strategies: int = 10
    days: int = 365
    resample: str = "15m"
    rounds: int = 1


class FormulaTestRequest(BaseModel):
    formula: str
    params: dict | None = None
    sample_limit: int = 5


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """System overview — available endpoints and state."""
    return {
        "name": "ccode_botv0 Agent API",
        "version": "0.1.0",
        "endpoints": {
            "GET  /": "This help",
            "GET  /indicators": "List registered indicators",
            "POST /indicators": "Register a new indicator (formula)",
            "POST /indicators/test": "Test a formula without registering",
            "GET  /leaderboard": "Research leaderboard",
            "GET  /strategies": "Saved strategies",
            "POST /research/run": "Run research loop",
            "POST /data-sources": "Register a new data source",
            "GET  /system": "System state (containers, data stats)",
        },
        "total_indicators": len(INDICATOR_REGISTRY),
    }


@app.get("/indicators")
def list_indicators():
    """List all registered indicators."""
    return {
        "total": len(INDICATOR_REGISTRY),
        "indicators": sorted(INDICATOR_REGISTRY.keys()),
    }


@app.post("/indicators/test")
def test_indicator(req: FormulaTestRequest):
    """Test a formula on the latest market data without registering."""
    from src.indicators.calculator import calc_all
    data_path = Path("data/raw/binance/btcusdt/15m/data.parquet")
    if not data_path.exists():
        raise HTTPException(404, "No 15m data found")

    df = pd.read_parquet(data_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.tail(200)  # last 200 bars
    df = calc_all(df)

    result = calc_formula(df, req.formula, req.params)

    # Return last N values
    values = result.tail(req.sample_limit).tolist()
    ts_values = df["ts"].tail(req.sample_limit).dt.isoformat().tolist()

    return {
        "formula": req.formula,
        "sample": [{"ts": t, "value": v} for t, v in zip(ts_values, values)],
        "last_value": values[-1] if values else None,
    }


@app.post("/indicators")
def create_indicator(req: IndicatorRequest):
    """Register a new indicator. After registration, rebuild features."""
    if req.name in INDICATOR_REGISTRY:
        raise HTTPException(409, f"Indicator '{req.name}' already exists")

    try:
        register_dynamic_indicator(req.name, req.formula, req.params, req.description)
    except Exception as e:
        raise HTTPException(400, str(e))

    return {
        "status": "registered",
        "name": req.name,
        "formula": req.formula,
        "note": "Run `python -m src.features.builder --symbol BTCUSDT --rebuild` to backfill",
    }


@app.get("/leaderboard")
def get_leaderboard(top: int = 10, min_sharpe: float = -999):
    """Get top strategies from the research leaderboard."""
    path = Path("data/parquet/research/leaderboard.parquet")
    if not path.exists():
        raise HTTPException(404, "No research data yet. Run research loop first.")

    df = pd.read_parquet(path)
    df = df[df["n_trades"] > 0]
    df = df[df["sharpe"] >= min_sharpe]
    df = df.sort_values("sharpe", ascending=False).head(top)

    results = []
    for _, r in df.iterrows():
        entry = {
            "run_id": r["run_id"],
            "sharpe": round(r["sharpe"], 4),
            "win_rate": round(r["win_rate"], 4),
            "profit_factor": round(r["profit_factor"], 4),
            "total_pnl": round(r["total_pnl"], 2),
            "n_trades": r["n_trades"],
            "passes_gates": bool(r["passes_gates"]),
        }
        # Parse JSON fields safely
        for field in ["rules_json", "llm_explanation", "llm_suggestions"]:
            val = r.get(field, "")
            if isinstance(val, str) and val and val != "nan":
                entry[field.replace("_json", "")] = val[:500] if field == "llm_explanation" else val
        results.append(entry)

    return {"total": len(df), "top": results}


@app.get("/strategies")
def list_strategies():
    """List saved strategies."""
    strat_dir = Path("data/strategies")
    if not strat_dir.exists():
        return {"strategies": []}

    strategies = []
    for p in sorted(strat_dir.glob("*.json")):
        strategies.append({
            "name": p.stem,
            "path": str(p),
            "updated": p.stat().st_mtime,
        })
    return {"total": len(strategies), "strategies": strategies}


@app.get("/system")
def system_status():
    """System state — data volumes, container status, etc."""
    from src.store import row_count as raw_count
    from src.features.store import row_count as feat_count

    info = {
        "parquet_raw": raw_count("binance", "btcusdt"),
        "parquet_features": feat_count("btcusdt"),
        "indicators_total": len(INDICATOR_REGISTRY),
    }

    # Paper trading state
    state_path = Path("data/paper_state.json")
    if state_path.exists():
        info["paper_trading"] = json.loads(state_path.read_text())

    return info
