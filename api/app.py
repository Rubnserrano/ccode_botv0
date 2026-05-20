"""API Layer for agent interaction — FastAPI endpoints.

Agents can:
  - Register new indicators (formula-based)
  - Register new data sources
  - Launch research runs (async)
  - View system state (brain-context)
  - View backtest charts (Plotly JSON)
  - Deploy strategies to paper trading

Usage:
    .venv/bin/uvicorn api.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import math
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
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


# ─── In-memory research run tracker ──────────────────────────────────────

_active_runs: dict[str, dict] = {}


# ─── Schemas ─────────────────────────────────────────────────────────────

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
    symbol: str = "btcusdt"
    api_key: str = ""


class FormulaTestRequest(BaseModel):
    formula: str
    params: dict | None = None
    sample_limit: int = 5


# ─── Helpers ─────────────────────────────────────────────────────────────

def _load_paper_state() -> dict:
    path = Path("data/paper_state.json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"equity": 0, "capital": 0, "n_trades": 0, "position": None}


def _leaderboard_summary() -> dict:
    from src.research.leaderboard import load_leaderboard
    try:
        df = load_leaderboard()
        valid = df[df["n_trades"] > 0]
        if valid.empty:
            return {"total": 0, "best_sharpe": 0, "best_oos_sharpe": 0}
        best = valid.loc[valid["sharpe"].idxmax()]
        best_oos = None
        if "oos_sharpe" in valid.columns:
            oos_valid = valid[valid["oos_trades"] >= 10]
            if not oos_valid.empty:
                best_oos_idx = oos_valid["oos_sharpe"].idxmax()
                best_oos = float(oos_valid.loc[best_oos_idx, "oos_sharpe"])
        return {
            "total": len(valid),
            "best_sharpe": round(float(best["sharpe"]), 4),
            "best_oos_sharpe": round(float(best_oos), 4) if best_oos is not None else None,
        }
    except Exception:
        return {"total": 0, "best_sharpe": 0, "best_oos_sharpe": 0}


# ─── Async research runner ───────────────────────────────────────────────

async def _run_research_background(run_id: str, req: ResearchRequest) -> None:
    from src.brain.orchestrator import run_research
    _active_runs[run_id] = {"status": "running", "progress": "starting"}
    try:
        results = await run_research(
            n_strategies=req.n_strategies,
            days=req.days,
            resample=req.resample,
            rounds=req.rounds,
            symbol=req.symbol,
            api_key=req.api_key or None,
        )
        _active_runs[run_id] = {
            "status": "completed",
            "progress": "done",
            "results_count": len(results),
        }
    except Exception as e:
        logger.exception("research run %s failed", run_id)
        _active_runs[run_id] = {"status": "failed", "error": str(e)}


# ─── Endpoints ───────────────────────────────────────────────────────────

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
            "POST /research/run": "Run research loop (async)",
            "GET  /research/{run_id}": "Check research run status",
            "GET  /backtest/{run_id}/chart": "Plotly JSON equity chart",
            "POST /strategies/{name}/deploy": "Deploy strategy to paper trading",
            "POST /data-sources": "Register a new data source",
            "GET  /data-sources": "List all data sources",
            "GET  /system": "System state",
            "GET  /brain-context": "Unified system context for agents",
        },
        "total_indicators": len(INDICATOR_REGISTRY),
    }


@app.get("/brain-context")
def brain_context():
    """Unified system context — market, data, research, paper in one call."""
    from src.ts_catalog import get_catalog

    paper = _load_paper_state()
    lb = _leaderboard_summary()

    catalog = get_catalog()
    total_rows = 0
    sources = set()
    latest_ts = None
    if not catalog.empty:
        total_rows = int(catalog["rows"].sum()) if "rows" in catalog.columns else 0
        sources = set(catalog["source_type"].unique())
        if "max_ts" in catalog.columns:
            try:
                latest_ts = str(catalog["max_ts"].max())
            except Exception:
                pass

    regime = "unknown"
    active_strat_path = Path("data/strategies/active.json")
    active_strategy = None
    if active_strat_path.exists():
        try:
            active_strategy = json.loads(active_strat_path.read_text()).get("name", "unknown")
        except Exception:
            pass

    ctx = {
        "market": {
            "regime": regime,
            "data_sources": list(sources),
        },
        "data": {
            "total_rows": total_rows,
            "latest_timestamp": latest_ts or "unknown",
            "indicators_available": len(INDICATOR_REGISTRY),
        },
        "research": {
            "total_strategies_tested": lb["total"],
            "best_sharpe": lb["best_sharpe"],
            "best_oos_sharpe": lb["best_oos_sharpe"],
            "active_runs": len(_active_runs),
        },
        "paper": {
            "equity": paper.get("equity", 0),
            "capital": paper.get("capital", 0),
            "n_trades": paper.get("n_trades", 0),
            "position": paper.get("position"),
            "active_strategy": active_strategy,
        },
        "suggested_action": _suggest_action(lb, paper),
    }
    return ctx


def _safe_float(v: Any) -> float:
    """Safely convert a value to float, handling inf/nan."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return 0.0
        return round(f, 4)
    except (ValueError, TypeError):
        return 0.0


def _suggest_action(lb: dict, paper: dict) -> str:
    if lb["total"] == 0:
        return "Run research to discover strategies (POST /research/run)"
    if paper.get("position") is None and lb["best_oos_sharpe"] and lb["best_oos_sharpe"] > 1.0:
        return "Best strategy has Sharpe OOS > 1.0 — consider deploying (POST /strategies/{name}/deploy)"
    if lb["best_oos_sharpe"] and lb["best_oos_sharpe"] < 1.0:
        return "No strong strategies found — run more research"
    return "System running normally — check leaderboard for new strategies"


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
        from src.ts_store import read as ts_read

        df = ts_read("market:binance:btcusdt", frequency="15m", limit=200)
        if df.empty:
            df = ts_read("market:binance:btcusdt", frequency="raw", limit=200)
        if df.empty:
            raise HTTPException(404, "No data found in TS Store")

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
        "note": "Now available in any strategy definition",
    }


@app.get("/leaderboard")
def get_leaderboard(top: int = 10, min_sharpe: float = -999):
    """Get top strategies from the research leaderboard."""
    from src.research.leaderboard import load_leaderboard

    df = load_leaderboard()
    if df.empty:
        raise HTTPException(404, "No research data yet. Run research loop first.")
    df = df[df["n_trades"] > 0]
    df = df[df["sharpe"] >= min_sharpe]
    df = df.sort_values("sharpe", ascending=False).head(top)

    results = []
    for _, r in df.iterrows():
        entry = {
            "run_id": r["run_id"],
            "sharpe": round(r["sharpe"], 4),
            "win_rate": round(r["win_rate"], 4),
            "profit_factor": _safe_float(r["profit_factor"]),
            "total_pnl": round(r["total_pnl"], 2),
            "n_trades": r["n_trades"],
            "passes_gates": bool(r["passes_gates"]),
        }
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
        if p.name in ("active.json",):
            continue
        strategies.append({
            "name": p.stem,
            "path": str(p),
            "updated": p.stat().st_mtime,
        })
    return {"total": len(strategies), "strategies": strategies}


@app.get("/system")
def system_status():
    """System state — data volumes, paper trading, etc."""
    from src.ts_catalog import get_catalog
    cat = get_catalog()
    total_rows = int(cat["rows"].sum()) if not cat.empty else 0
    sources = len(cat["asset_id"].unique()) if not cat.empty else 0

    info = {
        "ts_store_total_rows": total_rows,
        "ts_store_assets": sources,
        "ts_store_frequencies": sorted(cat["frequency"].unique().tolist()) if not cat.empty else [],
        "indicators_total": len(INDICATOR_REGISTRY),
    }

    state_path = Path("data/paper_state.json")
    if state_path.exists():
        info["paper_trading"] = json.loads(state_path.read_text())

    return info


# ─── Research endpoints ──────────────────────────────────────────────────

@app.post("/research/run")
async def run_research_endpoint(req: ResearchRequest):
    """Start a research run in the background. Returns immediately with run_id."""
    run_id = uuid.uuid4().hex[:12]
    asyncio.create_task(_run_research_background(run_id, req))
    return {
        "run_id": run_id,
        "status": "started",
        "config": req.model_dump(),
    }


@app.get("/research/{run_id}")
def get_research_status(run_id: str):
    """Check the status of a research run."""
    run = _active_runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"Research run '{run_id}' not found")
    return {"run_id": run_id, **run}


# ─── Strategy deploy ─────────────────────────────────────────────────────

@app.post("/strategies/{name}/deploy")
def deploy_strategy(name: str):
    """Deploy a strategy (from leaderboard or saved file) to paper trading.

    Saves the strategy definition to data/strategies/active.json.
    The PaperRunner will pick it up on the next candle close.
    """
    # Try leaderboard first
    from src.research.leaderboard import load_leaderboard
    strategy_def = None

    try:
        df = load_leaderboard()
        row = df[df["run_id"] == name]
        if not row.empty:
            rules_raw = row.iloc[0].get("rules_json", "")
            if isinstance(rules_raw, str) and rules_raw:
                strategy_def = json.loads(rules_raw)
    except Exception:
        pass

    # Fallback: try saved strategy file
    if strategy_def is None:
        strat_path = Path("data/strategies") / f"{name}.json"
        if strat_path.exists():
            try:
                strategy_def = json.loads(strat_path.read_text())
            except Exception:
                pass

    if strategy_def is None:
        raise HTTPException(404, f"Strategy '{name}' not found in leaderboard or saved files")

    # Validate
    from src.strategy_engine.schema import validate_strategy
    try:
        sd = validate_strategy(strategy_def)
    except Exception as e:
        raise HTTPException(400, f"Invalid strategy: {e}")

    # Save as active strategy
    active_path = Path("data/strategies/active.json")
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(strategy_def, indent=2))

    logger.info("API: strategy '%s' deployed to paper trading", sd.name)

    # Try Telegram notification
    try:
        from src.notification.telegram import notify_deployed
        asyncio.create_task(notify_deployed(sd.name, 0, 0))
    except Exception:
        pass

    return {
        "status": "deployed",
        "name": sd.name,
        "entry_conditions": len(sd.entry_conditions),
        "exit": {"tp_pct": sd.exit.tp_pct, "sl_pct": sd.exit.sl_pct, "horizon": sd.exit.horizon_bars},
        "note": "Active in paper trading. Check /brain-context for status.",
    }


# ─── Backtest chart ──────────────────────────────────────────────────────

@app.get("/backtest/{run_id}/chart")
def backtest_chart(run_id: str):
    """Return a Plotly JSON chart for a strategy from the leaderboard.

    Equity curves for train and test periods with train/test split marked.
    """
    from src.research.leaderboard import load_leaderboard

    df_lb = load_leaderboard()
    if df_lb.empty:
        raise HTTPException(404, "No research data yet")

    row = df_lb[df_lb["run_id"] == run_id]
    if row.empty:
        raise HTTPException(404, f"Strategy '{run_id}' not found in leaderboard")

    row = row.iloc[0]
    rules_raw = row.get("rules_json", "")
    config_raw = row.get("config_json", "")

    if not isinstance(rules_raw, str) or not rules_raw:
        raise HTTPException(400, "Strategy has no rules_json")
    if not isinstance(config_raw, str) or not config_raw:
        raise HTTPException(400, "Strategy has no config_json")

    try:
        strategy_def = json.loads(rules_raw)
        config = json.loads(config_raw)
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    from src.strategy_engine.schema import validate_strategy
    from src.strategy_engine.evaluator import evaluate
    from src.backtesting.engine import backtest
    from src.indicators.calculator import calc_all

    try:
        sd = validate_strategy(strategy_def)
    except Exception as e:
        raise HTTPException(400, f"Strategy validation: {e}")

    timeframe = config.get("resample", config.get("timeframe", "15m"))
    days = config.get("days", 365)
    symbol = config.get("symbol", "btcusdt")

    # Load data from TS Store
    asset_id = f"market:binance:{symbol}" if ":" not in symbol else symbol
    from src.ts_store import read as ts_read

    df = ts_read(asset_id, frequency=timeframe, limit=50000)
    if df.empty:
        raise HTTPException(404, f"No data found for {asset_id} @ {timeframe}")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff]
    if len(df) < 200:
        raise HTTPException(400, "Not enough data rows")
    df = calc_all(df)

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    def eval_fn(d):
        return evaluate(d, sd)

    trades_train, _ = backtest(df_train, eval_fn,
        horizon=sd.exit.horizon_bars, warmup=50, cooldown=2,
        size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)

    trades_test, _ = backtest(df_test, eval_fn,
        horizon=sd.exit.horizon_bars, warmup=0, cooldown=2,
        size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)

    def _build_equity_curve(trades, ts_start):
        if trades.empty:
            return []
        trades = trades.sort_values("ts").reset_index(drop=True)
        curve = [{"ts": str(ts_start), "equity": 0}]
        cum = 0.0
        for _, t in trades.iterrows():
            cum += t.get("net_pnl", 0)
            curve.append({"ts": str(t["ts"]) if hasattr(t["ts"], "strftime") else str(t["ts"]), "equity": round(cum, 2)})
        return curve

    eq_train = _build_equity_curve(trades_train, df_train["ts"].iloc[0])
    eq_test = _build_equity_curve(trades_test, df_test["ts"].iloc[0])

    split_ts = str(df_train["ts"].iloc[-1])

    # Build Plotly JSON
    import plotly.graph_objects as go
    fig = go.Figure()

    if eq_train:
        fig.add_trace(go.Scatter(
            x=[p["ts"] for p in eq_train],
            y=[p["equity"] for p in eq_train],
            mode="lines", name="Train (in-sample)",
            line=dict(color="blue", width=2),
        ))

    if eq_test:
        fig.add_trace(go.Scatter(
            x=[p["ts"] for p in eq_test],
            y=[p["equity"] for p in eq_test],
            mode="lines", name="Test (out-of-sample)",
            line=dict(color="green", width=2),
        ))

    fig.add_vline(
        x=split_ts, line_dash="dash", line_color="red",
        annotation_text="Train/Test Split",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{run_id[:30]} — Equity Curve",
        xaxis_title="Time",
        yaxis_title="Cumulative P&L ($)",
        hovermode="x unified",
        template="plotly_white",
    )

    chart_json = json.loads(fig.to_json())
    metrics = {
        "sharpe_train": round(float(row.get("sharpe", 0)), 4),
        "sharpe_test": round(float(row.get("oos_sharpe", 0)), 4),
        "win_rate": round(float(row.get("win_rate", 0)), 4),
        "profit_factor": round(float(row.get("profit_factor", 0)), 4),
        "n_trades_train": len(trades_train),
        "n_trades_test": len(trades_test),
        "overfit": bool(row.get("overfit", False)),
    }

    return {"run_id": run_id, "chart": chart_json, "metrics": metrics}


# ─── Catalog endpoint ───────────────────────────────────────────────────

@app.get("/catalog")
def get_catalog():
    """Return metadata for all time-series assets in the TS Store."""
    from src.ts_catalog import get_catalog as _get_catalog
    cat = _get_catalog(force_refresh=False)
    if cat.empty:
        return {"assets": []}

    assets = []
    for _, r in cat.iterrows():
        assets.append({
            "asset_id": r.get("asset_id", ""),
            "source_type": r.get("source_type", ""),
            "frequency": r.get("frequency", ""),
            "rows": int(r.get("rows", 0)),
            "min_ts": str(r.get("min_ts", ""))[:10],
            "max_ts": str(r.get("max_ts", ""))[:10],
            "last_updated": str(r.get("last_updated", "")),
        })
    return {"total": len(assets), "assets": assets}


# ─── Data Source endpoints ───────────────────────────────────────────────

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
    """Register a new external data source."""
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
    """Force an immediate fetch of a data source."""
    ds = load_source(name)
    if ds is None:
        raise HTTPException(404, f"Data source '{name}' not found")

    df = fetch_source(ds)
    if df.empty:
        raise HTTPException(502, f"Fetch returned no data for '{name}'")

    for tf in ds.align.target_timeframes:
        align_source(df, ds, tf)

    from src.strategy_engine.registry import INDICATOR_REGISTRY
    col_name = list(ds.columns.values())[0]
    if col_name not in INDICATOR_REGISTRY:
        INDICATOR_REGISTRY[col_name] = lambda df, p, _c=col_name: df[_c] if _c in df.columns else pd.Series(0, index=df.index)

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
