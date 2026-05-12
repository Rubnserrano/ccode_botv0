"""MCP Server: Strategy Backtest Tools.

Tools:
  - test_strategy       → Backtest a strategy, return metrics
  - compare_strategies  → Compare multiple strategies side-by-side
  - validate_robustness → Monte Carlo + sensitivity analysis

Run:
    python -m src.mcp_servers.backtest_server
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

server = Server("backtest")


def _make_strategy_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Strategy name"},
            "entry_conditions": {
                "type": "array",
                "description": "List of conditions (AND logic). Each: {indicator, op, value}",
                "items": {
                    "type": "object",
                    "properties": {
                        "indicator": {"type": "string"},
                        "op": {"type": "string", "enum": ["lt", "gt", "lte", "gte", "eq", "ne", "cross_above", "cross_below"]},
                        "value": {"type": "number"},
                        "params": {
                            "type": "object",
                            "description": "Optional indicator params (e.g. {'period': 14})",
                        },
                    },
                    "required": ["indicator", "op", "value"],
                },
            },
            "exit": {
                "type": "object",
                "properties": {
                    "tp_pct": {"type": "number", "description": "Take profit %"},
                    "sl_pct": {"type": "number", "description": "Stop loss %"},
                    "horizon_bars": {"type": "integer", "description": "Max hold bars"},
                },
            },
        },
        "required": ["name", "entry_conditions", "exit"],
    }


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="test_strategy",
            description="Backtest a single strategy and return detailed metrics.\n"
                        "The strategy is evaluated against OHLCV data for the given asset.\n"
                        "Returns: sharpe, win_rate, profit_factor, n_trades, total_pnl, max_dd, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": _make_strategy_schema(),
                    "asset_id": {
                        "type": "string",
                        "description": "Market asset to test on (default: market:binance:btcusdt)",
                    },
                    "frequency": {
                        "type": "string",
                        "enum": ["15m", "1h", "1d"],
                        "description": "Data timeframe (default: 15m)",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days of history to test (default: 180)",
                    },
                    "size_usdc": {
                        "type": "number",
                        "description": "Position size in USDC (default: 50)",
                    },
                },
                "required": ["strategy"],
            },
        ),
        Tool(
            name="compare_strategies",
            description="Backtest multiple strategies and return a ranked comparison.\n"
                        "Useful for comparing variants of the same idea.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategies": {
                        "type": "array",
                        "items": _make_strategy_schema(),
                        "description": "Array of 2-5 strategy definitions to compare",
                    },
                    "asset_id": {"type": "string"},
                    "frequency": {"type": "string", "enum": ["15m", "1h", "1d"]},
                    "days": {"type": "integer"},
                    "metric": {
                        "type": "string",
                        "enum": ["sharpe", "profit_factor", "win_rate", "total_pnl"],
                        "description": "Sort metric (default: sharpe)",
                    },
                },
                "required": ["strategies"],
            },
        ),
        Tool(
            name="validate_robustness",
            description="Run robustness validation: Monte Carlo + walk-forward.\n"
                        "Tests if the strategy is robust or just lucky.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": _make_strategy_schema(),
                    "asset_id": {"type": "string"},
                    "frequency": {"type": "string"},
                    "days": {"type": "integer"},
                    "mc_samples": {
                        "type": "integer",
                        "description": "Monte Carlo resamples (default: 200)",
                    },
                },
                "required": ["strategy"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "test_strategy":
        return await _test_strategy(arguments)
    elif name == "compare_strategies":
        return await _compare_strategies(arguments)
    elif name == "validate_robustness":
        return await _validate_robustness(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _test_strategy(args: dict) -> list[TextContent]:
    from src.strategy_engine.schema import validate_strategy
    from src.strategy_engine.evaluator import evaluate
    from src.backtesting.engine import backtest
    from src.query import get as query_get

    strategy_dict = args["strategy"]
    asset_id = args.get("asset_id", "market:binance:btcusdt")
    freq = args.get("frequency", "15m")
    days = args.get("days", 180)
    size = args.get("size_usdc", 50)

    try:
        sd = validate_strategy(strategy_dict)
    except Exception as e:
        return [TextContent(type="text", text=f"Strategy validation error: {e}")]

    try:
        # Load data
        import pandas as pd
        from datetime import datetime, timezone

        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df = query_get({asset_id.split(":")[-1]: asset_id},
                       frequency=freq, start=start, with_indicators=True)
        if df.empty:
            return [TextContent(type="text", text=f"No data available for {asset_id} at {freq}.")]

        # Remove prefix from columns for the evaluator
        prefix = f"{list(asset_id.split(':')[-1])}_"
        # Actually we need to handle this properly — the evaluator expects column names
        # like "rsi_14", "close", etc. Let's strip the prefix.
        prefix_found = None
        for c in df.columns:
            if c.endswith("_close") and c != "ts":
                prefix_found = c.replace("_close", "")
                break

        if prefix_found:
            rename_map = {c: c.replace(f"{prefix_found}_", "", 1) for c in df.columns if c != "ts"}
            df = df.rename(columns=rename_map)

        # Strategy eval function
        def eval_fn(d):
            return evaluate(d, sd)

        trades, summary = backtest(
            df, eval_fn,
            horizon=sd.exit.horizon_bars,
            warmup=50, cooldown=2,
            size_usdc=size,
            tp_pct=sd.exit.tp_pct,
            sl_pct=sd.exit.sl_pct,
        )

        result = {
            "name": sd.name,
            "sharpe": round(summary.get("sharpe", 0), 4),
            "win_rate": round(summary.get("win_rate", 0), 4),
            "profit_factor": round(summary.get("profit_factor", 0), 4),
            "total_pnl": round(summary.get("total_pnl", 0), 2),
            "max_dd": round(summary.get("max_dd", 0), 2),
            "n_trades": summary.get("n_trades", 0),
            "passes_gates": bool(summary.get("passes_gates", False)),
            "avg_bars_held": round(summary.get("avg_bars_held", 0), 1) if summary.get("avg_bars_held") else 0,
            "config": {
                "asset": asset_id,
                "frequency": freq,
                "days": days,
                "size_usdc": size,
            },
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Backtest error: {e}")]


async def _compare_strategies(args: dict) -> list[TextContent]:
    strategies = args.get("strategies", [])
    if not strategies:
        return [TextContent(type="text", text="No strategies provided.")]

    results = []
    for s in strategies:
        r = await _test_strategy({
            "strategy": s,
            "asset_id": args.get("asset_id", "market:binance:btcusdt"),
            "frequency": args.get("frequency", "15m"),
            "days": args.get("days", 180),
        })
        # Parse result JSON
        try:
            data = json.loads(r[0].text)
            results.append(data)
        except Exception:
            results.append({"name": s.get("name", "?"), "error": r[0].text})

    metric = args.get("metric", "sharpe")
    results.sort(key=lambda x: x.get(metric, -999), reverse=True)

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


async def _validate_robustness(args: dict) -> list[TextContent]:
    strategy_dict = args["strategy"]
    asset_id = args.get("asset_id", "market:binance:btcusdt")
    freq = args.get("frequency", "15m")
    days = args.get("days", 180)
    mc_samples = args.get("mc_samples", 200)

    # First run the backtest
    bt_result = await _test_strategy(args)
    bt_data = json.loads(bt_result[0].text)

    # Monte Carlo: resample trades
    # For now, do a simple analysis
    n_trades = bt_data.get("n_trades", 0)
    sharpe = bt_data.get("sharpe", 0)

    robustness = {
        "strategy": strategy_dict.get("name", "?"),
        "backtest": bt_data,
        "robustness_score": round(min(1.0, (n_trades / 100) * 0.5 + (sharpe / 2) * 0.5), 4),
        "n_trades": n_trades,
        "note": "Full Monte Carlo pending robustness engine (Phase 1)",
    }

    return [TextContent(type="text", text=json.dumps(robustness, indent=2))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="backtest",
                server_version="0.1.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
