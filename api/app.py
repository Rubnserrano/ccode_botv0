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
from src.intelligence.models import DataSource, ParseConfig, AlignConfig, RateLimit
from src.intelligence.registry import save_source, load_source, list_sources as list_data_sources, delete_source, try_auto_discover
from src.intelligence.fetcher import fetch_source
from src.intelligence.aligner import align_source

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
    try:
        from src.indicators.calculator import calc_all
        data_path = Path("data/raw/binance/btcusdt/15m/data.parquet")
        if not data_path.exists():
            # Fallback to 1m data
            data_path = Path("data/raw/binance/btcusdt/1m/data.parquet") if not data_path.exists() else data_path
            if not data_path.exists():
                raise HTTPException(404, "No data found (looked in 15m/ and 1m/)")

        df = pd.read_parquet(data_path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.tail(200)
        df = calc_all(df)

        result = calc_formula(df, req.formula, req.params)

        values = result.tail(req.sample_limit).tolist()
        ts_values = df["ts"].tail(req.sample_limit).astype(str).tolist()

        return {
            "formula": req.formula,
            "sample": [{"ts": t, "value": round(v, 4) if isinstance(v, float) else v} for t, v in zip(ts_values, values)],
            "last_value": round(values[-1], 4) if values and isinstance(values[-1], float) else (values[-1] if values else None),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Formula error: {e}")


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


# ─── Data Source endpoints ───────────────────────────────────────────────────

class DataSourceRequest(BaseModel):
    name: str
    url: str
    description: str = ""
    params: dict = {}
    headers: dict = {}
    schedule: str = "1h"
    parse: dict = {"type": "json", "timestamp_field": "ts", "value_field": "value"}
    columns: dict = {"value": "value"}
    align: dict = {"method": "ffill", "target_timeframes": ["15m"]}
    api_key: str = ""


@app.post("/data-sources")
def create_data_source(req: DataSourceRequest):
    """Register a new external data source.

    The source is saved but NOT fetched until:
      - A strategy references its column
      - POST /data-sources/{name}/fetch is called
    """
    parse_cfg = ParseConfig(**req.parse)
    align_cfg = AlignConfig(**req.align)
    rate_limit = RateLimit()
    ds = DataSource(
        name=req.name, url=req.url, description=req.description,
        params=req.params, headers=req.headers,
        schedule=req.schedule, parse=parse_cfg,
        columns=req.columns, align=align_cfg,
        rate_limit=rate_limit, api_key=req.api_key,
    )
    errors = ds.validate()
    if errors:
        raise HTTPException(400, f"Validation errors: {', '.join(errors)}")

    # Check duplicate
    existing = load_source(req.name)
    if existing:
        raise HTTPException(409, f"Data source '{req.name}' already exists")

    save_source(ds)
    return {
        "status": "registered",
        "name": req.name,
        "schedule": req.schedule,
        "note": "Source saved. Fetch on first use, or POST /data-sources/{name}/fetch",
    }


@app.get("/data-sources")
def list_all_data_sources():
    """List all registered data sources."""
    return {"total": len(list_data_sources()), "sources": list_data_sources()}


@app.get("/data-sources/{name}")
def get_data_source(name: str):
    """Get details of a specific data source."""
    ds = load_source(name)
    if ds is None:
        raise HTTPException(404, f"Data source '{name}' not found")
    return ds.to_dict()


@app.delete("/data-sources/{name}")
def remove_data_source(name: str):
    """Delete a data source and all its data."""
    if load_source(name) is None:
        raise HTTPException(404, f"Data source '{name}' not found")
    delete_source(name)
    return {"status": "deleted", "name": name}


@app.post("/data-sources/{name}/fetch")
def fetch_data_source(name: str):
    """Force an immediate fetch of a data source.

    The data is downloaded, aligned to configured timeframes,
    and the column is registered in the indicator registry.
    """
    ds = load_source(name)
    if ds is None:
        raise HTTPException(404, f"Data source '{name}' not found")

    df = fetch_source(ds)
    if df.empty:
        raise HTTPException(502, f"Fetch returned no data for '{name}'")

    for tf in ds.align.target_timeframes:
        align_source(df, ds, tf)

    # Register column in indicator registry
    from src.strategy_engine.registry import INDICATOR_REGISTRY
    col_name = list(ds.columns.values())[0]
    if col_name not in INDICATOR_REGISTRY:
        INDICATOR_REGISTRY[col_name] = lambda df, p, _c=col_name: df[_c] if _c in df.columns else pd.Series(0, index=df.index)

    # Update metadata
    from datetime import datetime, timezone
    ds.total_rows = len(df)
    ds.last_fetch_at = datetime.now(timezone.utc).isoformat()
    save_source(ds)

    return {
        "status": "fetched",
        "name": name,
        "rows": len(df),
        "columns": list(ds.columns.values()),
        "timeframes": ds.align.target_timeframes,
    }
