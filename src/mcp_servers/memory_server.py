"""MCP Server: Research Memory Tools.

Tools:
  - search_past_strategies  → Find similar strategies by concept
  - get_leaderboard         → Top strategies from past research
  - save_strategy           → Persist a strategy with results

Run:
    python -m src.mcp_servers.memory_server
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

server = Server("memory")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_past_strategies",
            description="Search past research results by concept or indicator.\n"
                        "Returns strategies that used similar concepts with their performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept to search for (e.g. momentum, volatility, rsi, atr)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                    },
                    "min_sharpe": {
                        "type": "number",
                        "description": "Minimum Sharpe filter (default: 0)",
                    },
                },
                "required": ["concept"],
            },
        ),
        Tool(
            name="get_leaderboard",
            description="Get top strategies from the research leaderboard.",
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of strategies (default: 10)",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["sharpe", "profit_factor", "win_rate", "n_trades"],
                        "description": "Sort metric (default: sharpe)",
                    },
                    "min_trades": {
                        "type": "integer",
                        "description": "Minimum trades filter (default: 10)",
                    },
                },
            },
        ),
        Tool(
            name="save_strategy",
            description="Save a successfully tested strategy to the leaderboard and disk.\n"
                        "Call this after test_strategy returns good metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "object",
                        "description": "Full strategy definition (name, entry_conditions, exit)",
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Backtest metrics to save (sharpe, win_rate, etc.)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human description of the strategy",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source identifier (default: mcp_agent)",
                    },
                },
                "required": ["strategy", "metrics"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_past_strategies":
        return await _search_strategies(arguments)
    elif name == "get_leaderboard":
        return await _get_leaderboard(arguments)
    elif name == "save_strategy":
        return await _save_strategy(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search_strategies(args: dict) -> list[TextContent]:
    from src.research.leaderboard import load_leaderboard

    concept = args.get("concept", "").lower()
    top_k = args.get("top_k", 5)
    min_sharpe = args.get("min_sharpe", 0)

    try:
        df = load_leaderboard()
    except FileNotFoundError:
        return [TextContent(type="text", text="No research data yet. Run research first.")]

    if df.empty:
        return [TextContent(type="text", text="No research data yet.")]

    # Filter by concept match in rules_json
    if concept:
        mask = df.get("rules_json", "").str.lower().str.contains(concept, na=False)
        df = df[mask]

    df = df[df["n_trades"] > 0]
    df = df[df["sharpe"] >= min_sharpe]
    df = df.sort_values("sharpe", ascending=False).head(top_k)

    if df.empty:
        return [TextContent(type="text", text=f"No strategies found matching '{concept}' with Sharpe >= {min_sharpe}.")]

    results = []
    for _, r in df.iterrows():
        entry = {
            "run_id": r.get("run_id", "?"),
            "sharpe": round(float(r.get("sharpe", 0)), 4),
            "win_rate": round(float(r.get("win_rate", 0)), 4),
            "profit_factor": round(float(r.get("profit_factor", 0)), 4),
            "total_pnl": round(float(r.get("total_pnl", 0)), 2),
            "n_trades": int(r.get("n_trades", 0)),
        }
        # Include rules if available
        rules = r.get("rules_json", "")
        if isinstance(rules, str) and rules and rules != "nan":
            try:
                entry["rules"] = json.loads(rules)
            except Exception:
                entry["rules_raw"] = rules[:200]
        results.append(entry)

    return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]


async def _get_leaderboard(args: dict) -> list[TextContent]:
    from src.research.leaderboard import load_leaderboard

    top_n = args.get("top_n", 10)
    metric = args.get("metric", "sharpe")
    min_trades = args.get("min_trades", 10)

    try:
        df = load_leaderboard()
    except FileNotFoundError:
        return [TextContent(type="text", text="No leaderboard data yet.")]

    if df.empty:
        return [TextContent(type="text", text="Leaderboard is empty.")]

    df = df[df["n_trades"] >= min_trades]
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).head(top_n)

    results = []
    for _, r in df.iterrows():
        results.append({
            "run_id": r.get("run_id", "?"),
            "sharpe": round(float(r.get("sharpe", 0)), 4),
            "win_rate": round(float(r.get("win_rate", 0)), 4),
            "profit_factor": round(float(r.get("profit_factor", 0)), 4),
            "n_trades": int(r.get("n_trades", 0)),
            "passes_gates": bool(r.get("passes_gates", False)),
        })

    return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]


async def _save_strategy(args: dict) -> list[TextContent]:
    from src.research.leaderboard import append_result
    from src.strategy_engine.store import save as save_strategy
    from src.strategy_engine.schema import validate_strategy

    strategy_dict = args.get("strategy", {})
    metrics = args.get("metrics", {})
    description = args.get("description", "")
    source = args.get("source", "mcp_agent")

    try:
        sd = validate_strategy(strategy_dict)
    except Exception as e:
        return [TextContent(type="text", text=f"Invalid strategy: {e}")]

    row = {
        "run_id": sd.name,
        "generation": 0,
        "sharpe": metrics.get("sharpe", 0),
        "win_rate": metrics.get("win_rate", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "total_pnl": metrics.get("total_pnl", 0),
        "max_dd": metrics.get("max_dd", 0),
        "n_trades": metrics.get("n_trades", 0),
        "passes_gates": metrics.get("passes_gates", False),
        "rules_json": json.dumps(strategy_dict),
        "config_json": json.dumps({"source": source, "description": description}),
        "elapsed_bt": metrics.get("elapsed_bt", 0),
    }

    try:
        append_result(row, strategy_dict)
    except Exception as e:
        return [TextContent(type="text", text=f"Save failed: {e}")]

    return [TextContent(type="text", text=json.dumps({
        "status": "saved",
        "name": sd.name,
        "sharpe": metrics.get("sharpe", 0),
        "path": f"data/strategies/research_{sd.name}.json",
    }, indent=2))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memory",
                server_version="0.1.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
