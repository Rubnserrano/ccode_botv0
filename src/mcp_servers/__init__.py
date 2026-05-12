"""MCP Servers for the Autonomous Research System.

Each server wraps a domain of functionality as MCP tools
that LLMs can call via tool_calling.

Servers:
  - query_server: Market data, assets, regime
  - backtest_server: Strategy testing, comparison, robustness
  - memory_server: Past strategies, leaderboard, persistence
"""
