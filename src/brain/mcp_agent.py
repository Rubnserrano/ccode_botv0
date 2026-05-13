"""MCP Agent — Autonomous research loop using tool_calling.

The agent:
  1. Connects to MCP tools (query, backtest, memory)
  2. Tells the LLM: "You have tools. Use them to research."
  3. LLM calls tools → agent executes → returns results
  4. LLM iterates until satisfied
  5. Final strategies are saved to leaderboard

Usage:
    OPENROUTER_API_KEY="sk-or-..." python -m src.brain.mcp_agent --rounds 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

_telegram = None


async def _maybe_notify(method: str, *args, **kwargs) -> None:
    """Call a telegram notification method if configured."""
    global _telegram
    if _telegram is None:
        if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            _telegram = True
        else:
            _telegram = False
    if not _telegram:
        return
    try:
        from src.notification.telegram import (
            notify_start, notify_strategy_found, notify_digest,
            notify_hourly_leaderboard, notify_done,
        )
        func = {
            "start": notify_start,
            "strategy": notify_strategy_found,
            "digest": notify_digest,
            "hourly": notify_hourly_leaderboard,
            "done": notify_done,
        }.get(method)
        if func:
            await func(*args, **kwargs)
    except Exception as e:
        logger.warning("telegram notification failed: %s", e)


# ─── Tool definitions (OpenAI-compatible format) ──────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_market",
            "description": "Get market data for assets aligned to the same 15m/1h/1d timeframe.\n"
                           "Automatically aligns different source types (market, sentiment, macro) to the same grid.\n"
                           "Returns OHLCV + indicators for market assets + any external sources requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assets": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dict mapping alias → asset_id. E.g. {\"btc\": \"market:binance:btcusdt\", \"fear\": \"sentiment:alternative:fear_greed\"}",
                    },
                    "frequency": {"type": "string", "enum": ["15m", "1h", "1d"], "description": "Default: 15m"},
                    "limit": {"type": "integer", "description": "Rows to return. Default: 100"},
                    "with_indicators": {"type": "boolean", "description": "Compute RSI, MACD, EMA, ATR, etc. Default: true"},
                },
                "required": ["assets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_assets",
            "description": "List all available data sources with their time ranges and row counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["market", "sentiment", "macro", "news"],
                        "description": "Filter by type (optional)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Get current market regime distribution (trending/ranging/volatile).\n"
                           "Essential for understanding what type of strategy might work now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Default: market:binance:btcusdt"},
                    "frequency": {"type": "string", "enum": ["15m", "1h", "1d"]},
                    "lookback_bars": {"type": "integer", "description": "Default: 200"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_strategy",
            "description": "Backtest a strategy and return metrics: sharpe, win_rate, profit_factor, n_trades, PnL, max_dd.\n"
                           "Entry conditions are AND logic. Exit rules: tp_pct, sl_pct, horizon_bars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entry_conditions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "indicator": {"type": "string"},
                                        "op": {"type": "string", "enum": ["lt", "gt", "lte", "gte", "eq", "cross_above", "cross_below"]},
                                        "value": {"type": "number"},
                                    },
                                    "required": ["indicator", "op", "value"],
                                },
                            },
                            "exit": {
                                "type": "object",
                                "properties": {
                                    "tp_pct": {"type": "number"},
                                    "sl_pct": {"type": "number"},
                                    "horizon_bars": {"type": "integer"},
                                },
                            },
                        },
                        "required": ["name", "entry_conditions", "exit"],
                    },
                    "asset_id": {"type": "string", "description": "Default: market:binance:btcusdt"},
                    "frequency": {"type": "string", "enum": ["15m", "1h", "1d"]},
                    "days": {"type": "integer", "description": "History to test. Default: 180"},
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": "Test multiple strategies side-by-side and return a ranked comparison.\n"
                           "Useful for A/B testing variants of the same idea (different params, different filters).",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategies": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Array of 2-5 strategy objects (same format as test_strategy)",
                    },
                    "asset_id": {"type": "string"},
                    "frequency": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "string", "enum": ["sharpe", "profit_factor", "win_rate"]},
                },
                "required": ["strategies"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_past_strategies",
            "description": "Search past research results for strategies that used a specific concept or indicator.\n"
                           "Learn from what worked (and didn't work) before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "E.g. momentum, volatility, rsi, atr, fear_greed"},
                    "top_k": {"type": "integer", "description": "Default: 5"},
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leaderboard",
            "description": "Get top strategies from the research leaderboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Default: 10"},
                    "min_trades": {"type": "integer", "description": "Default: 10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_strategy",
            "description": "Save a strategy that passed validation to the leaderboard for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "object", "description": "Strategy object (name, entry_conditions, exit)"},
                    "metrics": {"type": "object", "description": "Backtest metrics (sharpe, win_rate, profit_factor, n_trades, total_pnl)"},
                    "description": {"type": "string"},
                },
                "required": ["strategy", "metrics"],
            },
        },
    },
]


# ─── Tool handlers ────────────────────────────────────────────────────────

async def _handle_tool_call(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    from src.query import get as query_get, get_regime, list_available
    from src.research.leaderboard import append_result, load_leaderboard
    from src.strategy_engine.store import save as save_strat
    from src.strategy_engine.schema import validate_strategy
    from src.strategy_engine.evaluator import evaluate
    from src.backtesting.engine import backtest
    import pandas as pd

    if name == "query_market":
        assets = args.get("assets", {})
        freq = args.get("frequency", "15m")
        limit = args.get("limit", 100)
        indicators = args.get("with_indicators", True)
        df = query_get(assets, frequency=freq, limit=limit, with_indicators=indicators)
        if df.empty:
            return "No data available."
        records = df.tail(limit).to_dict(orient="records")
        summary = {
            "rows": len(records),
            "columns": [c for c in df.columns if c != "ts"],
            "time_range": {"from": str(df["ts"].iloc[0]), "to": str(df["ts"].iloc[-1])},
            "data_sample": records[-5:],
        }
        return json.dumps(summary, indent=2, default=str)

    elif name == "list_available_assets":
        assets = list_available(args.get("source_type"))
        return json.dumps(assets, indent=2, default=str) if assets else "No assets found."

    elif name == "get_market_regime":
        r = get_regime(
            args.get("asset_id", "market:binance:btcusdt"),
            args.get("frequency", "15m"),
            args.get("lookback_bars", 200),
        )
        return json.dumps(r, indent=2, default=str) if r else "Could not determine regime."

    elif name == "test_strategy":
        strategy_dict = args["strategy"]
        asset_id = args.get("asset_id", "market:binance:btcusdt")
        freq = args.get("frequency", "15m")
        days = args.get("days", 180)

        # Validate strategy
        try:
            sd = validate_strategy(strategy_dict)
        except Exception as e:
            return f"Strategy validation error: {e}"

        # Validate values are numeric
        for c in sd.entry_conditions:
            if c.value is not None and not isinstance(c.value, (int, float)):
                return f"Invalid non-numeric value '{c.value}' for indicator '{c.indicator}'. Use numeric values only."

        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        name_part = asset_id.split(":")[-1]
        df = query_get({name_part: asset_id}, frequency=freq, start=start, with_indicators=True)
        if df.empty:
            return f"No data for {asset_id} at {freq}."

        # Strip prefix from columns
        prefix = None
        for c in df.columns:
            if c.endswith("_close") and c != "ts":
                prefix = c.replace("_close", "")
                break
        if prefix:
            df = df.rename(columns={c: c.replace(f"{prefix}_", "", 1) for c in df.columns if c != "ts"})

        def eval_fn(d):
            return evaluate(d, sd)

        try:
            trades, summary = backtest(
                df, eval_fn,
                horizon=sd.exit.horizon_bars,
                warmup=50, cooldown=2,
                size_usdc=50,
                tp_pct=sd.exit.tp_pct,
                sl_pct=sd.exit.sl_pct,
            )
        except Exception as e:
            return f"Backtest error: {e}"

        result = {
            "name": sd.name,
            "sharpe": round(summary.get("sharpe", 0), 4),
            "win_rate": round(summary.get("win_rate", 0), 4),
            "profit_factor": round(summary.get("profit_factor", 0), 4),
            "total_pnl": round(summary.get("total_pnl", 0), 2),
            "max_dd": round(summary.get("max_dd", 0), 2),
            "n_trades": summary.get("n_trades", 0),
            "passes_gates": bool(summary.get("passes_gates", False)),
        }

        # Auto-save every tested strategy to leaderboard
        try:
            from src.research.leaderboard import append_result
            row = {
                "run_id": sd.name,
                "generation": 0,
                **result,
                "rules_json": json.dumps(strategy_dict),
                "config_json": json.dumps({"source": "mcp_agent", "auto_saved": True}),
                "elapsed_bt": 0,
            }
            append_result(row, strategy_dict)
        except Exception:
            pass

        return json.dumps(result, indent=2, default=str)

    elif name == "compare_strategies":
        results = []
        for s in args.get("strategies", []):
            r = await _handle_tool_call("test_strategy", {
                "strategy": s,
                "asset_id": args.get("asset_id", "market:binance:btcusdt"),
                "frequency": args.get("frequency", "15m"),
                "days": args.get("days", 180),
            })
            try:
                results.append(json.loads(r))
            except Exception:
                results.append({"name": s.get("name", "?"), "error": r[:200]})
        metric = args.get("metric", "sharpe")
        results.sort(key=lambda x: x.get(metric, -999), reverse=True)
        return json.dumps(results, indent=2, default=str)

    elif name == "search_past_strategies":
        concept = args.get("concept", "").lower()
        top_k = args.get("top_k", 5)
        df = load_leaderboard()
        if df.empty:
            return "No past research data."

        if concept:
            mask = df.get("rules_json", "").str.lower().str.contains(concept, na=False)
            df = df[mask]
        df = df[df["n_trades"] > 0].sort_values("sharpe", ascending=False).head(top_k)
        if df.empty:
            return f"No strategies found for '{concept}'."

        results = []
        for _, r in df.iterrows():
            entry = {
                "run_id": r.get("run_id", "?"),
                "sharpe": round(float(r.get("sharpe", 0)), 4),
                "win_rate": round(float(r.get("win_rate", 0)), 4),
                "n_trades": int(r.get("n_trades", 0)),
            }
            rules = r.get("rules_json", "")
            if isinstance(rules, str) and rules not in ("", "nan"):
                try:
                    entry["rules"] = json.loads(rules)
                except Exception:
                    pass
            results.append(entry)
        return json.dumps(results, indent=2, default=str)

    elif name == "get_leaderboard":
        df = load_leaderboard()
        if df.empty:
            return "No leaderboard data."
        top_n = args.get("top_n", 10)
        min_trades = args.get("min_trades", 10)
        df = df[df["n_trades"] >= min_trades].sort_values("sharpe", ascending=False).head(top_n)
        results = [{
            "run_id": r.get("run_id", "?"),
            "sharpe": round(float(r.get("sharpe", 0)), 4),
            "win_rate": round(float(r.get("win_rate", 0)), 4),
            "profit_factor": round(float(r.get("profit_factor", 0)), 4),
            "n_trades": int(r.get("n_trades", 0)),
        } for _, r in df.iterrows()]
        return json.dumps(results, indent=2, default=str)

    elif name == "save_strategy":
        strategy_dict = args["strategy"]
        metrics = args.get("metrics", {})
        row = {
            "run_id": strategy_dict.get("name", "mcp_strat"),
            "generation": 0,
            "sharpe": metrics.get("sharpe", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "max_dd": metrics.get("max_dd", 0),
            "n_trades": metrics.get("n_trades", 0),
            "passes_gates": metrics.get("passes_gates", False),
            "rules_json": json.dumps(strategy_dict),
            "config_json": json.dumps({"source": "mcp_agent", "desc": args.get("description", "")}),
            "elapsed_bt": 0,
        }
        append_result(row, strategy_dict)
        return json.dumps({"status": "saved", "name": strategy_dict.get("name", "?")}, indent=2)

    else:
        return f"Unknown tool: {name}"


# ─── Agent loop ───────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Eres un investigrador cuantitativo autónomo. Tu objetivo es generar y validar estrategias de trading.

INSTRUCCIONES ESTRICTAS:
1. Primero llama a **list_available_assets** y **get_market_regime** para entender el mercado
2. Luego llama a **test_strategy** para CADA estrategia que quieras probar — USA SIEMPRE valores NUMÉRICOS
3. Después de CADA test_strategy, modifica condiciones y vuelve a probar si el resultado es malo
4. Genera y prueba EXACTAMENTE {n} estrategias diferentes

REGLAS:
- "value" debe ser SIEMPRE un número, NUNCA un string
- Valores típicos: rsi 20-80, atr 0.1-2.0, volume 10-500, adx 15-50, ema 5-100
- Si una estrategia da 0 trades, cambia las condiciones
- El ratio TP/SL debe ser al menos 2:1
- Prefiere 1-2 condiciones, no más

IMPORTANTE: NO te detengas después de analizar el mercado. DEBES generar y probar estrategias.
Usa test_strategy múltiples veces hasta probar {n} estrategias.

Indicadores disponibles: rsi, regime, volume, adx, atr, macd, ema, close, vwap, obi
"""


async def run_round(
    llm_client,
    n_strategies: int = 3,
    max_tool_calls: int = 25,
) -> list[dict]:
    """Run one round of autonomous research using tool_calling.

    The LLM iterates: look at data → generate → test → refine → save.
    """
    from src.research.leaderboard import init_leaderboard

    init_leaderboard()
    system = AGENT_SYSTEM_PROMPT.format(n=n_strategies)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Analiza el mercado y genera {n_strategies} estrategias. OBLIGATORIO: usa test_strategy al menos {n_strategies} veces con diferentes condiciones. NO termines sin probar estrategias."},
    ]

    all_saved = []
    tool_call_count = 0
    test_count = 0

    while tool_call_count < max_tool_calls:
        response = await llm_client._call_model(
            model="deepseek/deepseek-chat",
            system="",  # system is in messages
            user="",
            response_format=None,  # no JSON mode when using tools
            max_tokens=4096,
            temperature=0.7,
            messages=messages,  # pass full history
            tools=TOOLS,
        )

        choice = response["choices"][0]
        msg = choice["message"]

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_call_count += 1
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {}

                logger.info("  Tool call #%d: %s(%s)", tool_call_count, name, json.dumps(args)[:120])
                result = await _handle_tool_call(name, args)
                logger.info("  → %s chars", len(result))

                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

                if name == "test_strategy":
                    test_count += 1

                # If saving, track it
                if name == "save_strategy":
                    try:
                        all_saved.append(json.loads(result))
                    except Exception:
                        pass
        else:
            content = msg.get("content", "")
            logger.info("Agent response: %s", content[:300] if content else "(empty)")
            # If LLM hasn't tested any strategies, force it
            if test_count < n_strategies:
                logger.info("Force: LLM responded without testing — pushing to test strategies")
                messages.append({"role": "user", "content": f"No has probado suficientes estrategias. Usa test_strategy para probar AL MENOS {n_strategies} estrategias diferentes con distintos indicadores y valores. NO respondas sin probar."})
                continue
            break

    return all_saved


# ─── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="MCP Autonomous Research Agent")
    parser.add_argument("--rounds", type=int, default=1, help="Research rounds")
    parser.add_argument("--strategies", type=int, default=3, help="Strategies per round")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")
    args = parser.parse_args()

    from src.brain.llm_client import LLMClient

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("No OPENROUTER_API_KEY. Set env var or pass --api-key")
        return

    llm = LLMClient(api_key=api_key)

    # Market context for Telegram
    market_ctx = ""
    try:
        from src.query import get_regime
        r = get_regime("market:binance:btcusdt", "15m", 200)
        regime_names = {0: "ranging", 1: "uptrend", 2: "downtrend"}
        market_ctx = " · ".join(f"{regime_names.get(int(k), k)}:{v:.0%}" for k, v in r.items())
    except Exception:
        pass

    await _maybe_notify("start", args.rounds, args.strategies, 365, "15m", market_ctx)
    total_start = time.time()
    all_saved = []

    for round_num in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"  Round {round_num + 1}/{args.rounds}")
        print(f"{'='*60}")

        saved = await run_round(llm, n_strategies=args.strategies)

        elapsed = time.time() - total_start
        print(f"\n  Round {round_num + 1}: {len(saved)} strategies saved ({elapsed:.0f}s)")
        for s in saved:
            print(f"    ✅ {s.get('name', '?')} — Sharpe={s.get('sharpe', 0):.2f}")
            # Notify each saved strategy
            await _maybe_notify("strategy",
                s.get("name", "?"), s.get("sharpe", 0), s.get("win_rate", 0),
                s.get("profit_factor", 0), s.get("total_pnl", 0), s.get("n_trades", 0),
                "", json.dumps({"tp_pct": 0.04, "sl_pct": 0.015}),
            )
        all_saved.extend(saved)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Complete: {total:.0f}s | {args.rounds} rounds | OpenRouter cost: ${llm.total_cost:.4f}")
    print(f"{'='*60}")

    # Final Telegram notification
    await _maybe_notify("done",
        total / 3600, args.rounds * args.strategies, len(all_saved),
        max((s.get("sharpe", 0) for s in all_saved), default=0),
        max((s.get("name", "") for s in all_saved), default=""),
        llm.total_cost,
        all_saved if all_saved else None,
    )

    await llm.close()


if __name__ == "__main__":
    import os
    asyncio.run(main())
