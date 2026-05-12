"""MCP Server: Market Data Query Tools.

Tools:
  - query_market       → Multi-asset aligned market data
  - list_assets        → Available data sources
  - get_regime         → Current market regime

Run:
    python -m src.mcp_servers.query_server
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

server = Server("query")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_market",
            description="Get market data for one or more assets aligned to the same timeframe.\n"
                        "Each asset can be a different type (market, sentiment) — all get aligned.\n"
                        "Example assets: {'btc': 'market:binance:btcusdt', 'fear': 'sentiment:alternative:fear_greed'}",
            inputSchema={
                "type": "object",
                "properties": {
                    "assets": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dict mapping alias → asset_id. E.g. {'btc': 'market:binance:btcusdt'}",
                    },
                    "frequency": {
                        "type": "string",
                        "enum": ["15m", "1h", "1d"],
                        "description": "Target timeframe (default: 15m)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows (default: 100)",
                    },
                    "with_indicators": {
                        "type": "boolean",
                        "description": "Compute indicators for market assets (default: true)",
                    },
                },
                "required": ["assets"],
            },
        ),
        Tool(
            name="list_available_assets",
            description="List all available data assets with their metadata (time range, row count, source type).",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["market", "sentiment", "macro", "news", "derivatives", "onchain"],
                        "description": "Filter by source type (optional)",
                    }
                },
            },
        ),
        Tool(
            name="get_market_regime",
            description="Get the current market regime distribution (trending/ranging/volatile).\n"
                        "Useful for understanding what type of market we're in before designing a strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset to analyze (default: market:binance:btcusdt)",
                    },
                    "frequency": {
                        "type": "string",
                        "enum": ["15m", "1h", "1d"],
                        "description": "Timeframe (default: 15m)",
                    },
                    "lookback_bars": {
                        "type": "integer",
                        "description": "Number of bars to analyze (default: 200)",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_market":
        return await _query_market(arguments)
    elif name == "list_available_assets":
        return await _list_assets(arguments)
    elif name == "get_market_regime":
        return await _get_regime(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _query_market(args: dict) -> list[TextContent]:
    from src.query import get as query_get

    assets = args.get("assets", {})
    freq = args.get("frequency", "15m")
    limit = args.get("limit", 100)
    indicators = args.get("with_indicators", True)

    try:
        df = query_get(assets, frequency=freq, limit=limit, with_indicators=indicators)
        if df.empty:
            return [TextContent(type="text", text="No data available for the requested assets.")]

        # Convert to records for JSON
        records = df.tail(limit).to_dict(orient="records")

        # Build summary
        summary = {
            "rows": len(records),
            "columns": [c for c in df.columns if c != "ts"],
            "time_range": {
                "from": str(df["ts"].iloc[0]),
                "to": str(df["ts"].iloc[-1]),
            },
            "assets_returned": list(assets.keys()),
            "data": records,
        }
        return [TextContent(type="text", text=json.dumps(summary, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Query error: {e}")]


async def _list_assets(args: dict) -> list[TextContent]:
    from src.query import list_available

    source_type = args.get("source_type")
    try:
        assets = list_available(source_type)
        if not assets:
            return [TextContent(type="text", text="No assets found.")]
        return [TextContent(type="text", text=json.dumps(assets, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"List error: {e}")]


async def _get_regime(args: dict) -> list[TextContent]:
    from src.query import get_regime as query_regime

    asset_id = args.get("asset_id", "market:binance:btcusdt")
    freq = args.get("frequency", "15m")
    lookback = args.get("lookback_bars", 200)

    try:
        regime = query_regime(asset_id, frequency=freq, lookback_bars=lookback)
        if not regime:
            return [TextContent(type="text", text="Could not determine market regime (insufficient data).")]
        return [TextContent(type="text", text=json.dumps(regime, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Regime error: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="query",
                server_version="0.1.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
