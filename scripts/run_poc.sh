#!/bin/bash
# POC: Autonomous Research Agent with Multi-Source Data
# 
# Usage:
#   export OPENROUTER_API_KEY="sk-or-..."
#   bash scripts/run_poc.sh
#
# Or with strategy-only (no API key):
#   bash scripts/run_poc.sh --no-llm

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python3"
export PYTHONPATH="$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   POC: Autonomous Research Agent + Multi-Source Data       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Migrate to Universal TS Store ───────────────────────────────
echo "📦 Step 1: Universal TimeSeries Store..."
$PYTHON scripts/migrate_to_ts.py 2>&1 | tail -10

# ─── Step 2: Align BTC to 15m ────────────────────────────────────────────
echo ""
echo "📐 Step 2: Align data to standard timeframes..."
$PYTHON -c "
from src.ts_aligner import align
n = align('market:binance:btcusdt', to_freq='15m')
print(f'  BTC 15m: {n} rows')
from src.ts_catalog import refresh_catalog
refresh_catalog()
" 2>&1

# ─── Step 3: Test multi-asset query ──────────────────────────────────────
echo ""
echo "🔍 Step 3: Multi-asset query test..."
$PYTHON -c "
from src.query import get, get_regime
import pandas as pd

df = get({
    'btc': 'market:binance:btcusdt',
    'fear': 'sentiment:alternative:fear_greed',
}, frequency='15m', limit=5, with_indicators=True)

print(f'  Columns: {list(df.columns)}')
print(f'  Last rows:')
for _, r in df.iterrows():
    print(f'    {r[\"ts\"]:%H:%M}  BTC=\${r[\"btc_close\"]:,.0f}  RSI={r[\"btc_rsi_14\"]:.0f}  Fear={r[\"fear_fear_greed\"]:.0f}')

regime = get_regime('market:binance:btcusdt', '15m')
regime_names = {0: 'ranging', 1: 'uptrend', 2: 'downtrend'}
regime_str = ', '.join(f'{regime_names.get(int(k), k)}: {v:.0%}' for k, v in regime.items())
print(f'  Market regime: {regime_str}')
" 2>&1

# ─── Step 4: Test strategy via MCP handler ───────────────────────────────
echo ""
echo "🧪 Step 4: Strategy backtest via MCP tools..."
$PYTHON -c "
import asyncio, json
from src.brain.mcp_agent import _handle_tool_call

strategies = [
    {
        'name': 'rsi_oversold_ranging',
        'entry_conditions': [
            {'indicator': 'rsi', 'op': 'lt', 'value': 30, 'params': {'period': 14}},
            {'indicator': 'regime', 'op': 'eq', 'value': 0},
        ],
        'exit': {'tp_pct': 0.04, 'sl_pct': 0.015, 'horizon_bars': 48},
    },
    {
        'name': 'compression_breakout',
        'entry_conditions': [
            {'indicator': 'atr', 'op': 'lt', 'value': 0.3},
            {'indicator': 'volume', 'op': 'gt', 'value': 50},
        ],
        'exit': {'tp_pct': 0.05, 'sl_pct': 0.02, 'horizon_bars': 48},
    },
]

results = []
for s in strategies:
    r = asyncio.run(_handle_tool_call('test_strategy', {
        'strategy': s, 'days': 365,
    }))
    results.append(json.loads(r))

results.sort(key=lambda x: x.get('sharpe', -999), reverse=True)
for r in results:
    tag = '✅' if r.get('passes_gates') else '  '
    print(f'  {tag} {r[\"name\"]:30s}  S={r[\"sharpe\"]:+.2f}  WR={r[\"win_rate\"]:.0%}  PF={r[\"profit_factor\"]:.2f}  T={r[\"n_trades\"]}')
" 2>&1

# ─── Step 5: MCP agent (requires API key) ────────────────────────────────
if [[ "$1" != "--no-llm" && -n "${OPENROUTER_API_KEY}" ]]; then
    echo ""
    echo "🤖 Step 5: MCP Autonomous Research Agent..."
    echo "  (OPENROUTER_API_KEY detected — running 1 round)"
    $PYTHON -m src.brain.mcp_agent --rounds 1 --strategies 2 2>&1
else
    echo ""
    echo "🤖 Step 5: MCP Agent (skipped — set OPENROUTER_API_KEY or pass --no-llm)"
    echo "  To run: OPENROUTER_API_KEY=\"sk-or-...\" bash scripts/run_poc.sh"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   POC Complete!                                             ║"
echo "║                                                              ║"
echo "║   New data layer:   data/ts/                                 ║"
echo "║   Available:        market:binance:btcusdt                   ║"
echo "║                     sentiment:alternative:fear_greed         ║"
echo "║                                                              ║"
echo "║   Next: set OPENROUTER_API_KEY and run the MCP agent         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
