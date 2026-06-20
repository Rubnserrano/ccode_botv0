#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Overnight Runner — Autonomous Research Agent (5 narrativas)
# ═══════════════════════════════════════════════════════════════════════
#
# Pre-requisites:
#   export OPENROUTER_API_KEY="sk-or-..."
#   export CRYPTOPANIC_API_KEY="your_key"     # opcional
#
# Usage:
#   bash scripts/run_overnight.sh
#   bash scripts/run_overnight.sh --no-news    # skip news if no API key
#   bash scripts/run_overnight.sh --dry-run    # setup only, no agent
# ═══════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python3"
export PYTHONPATH="$PROJECT_DIR"
export LOG_FILE="data/overnight_$(date +%Y%m%d_%H%M).log"
SKIP_NEWS=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --no-news) SKIP_NEWS=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      🌙 Overnight Autonomous Research Agent                ║"
echo "║      $(date)                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Log: $LOG_FILE"

# ────────────────────────────────────────────────────────────────────
# STEP 1: Gap-fill BTC data
# ────────────────────────────────────────────────────────────────────
echo ""
echo "📦 [1/6] Gap-filling BTC data..."
$PYTHON -m src.download --symbol BTCUSDT --fill 2>&1 | tail -3

# ────────────────────────────────────────────────────────────────────
# STEP 2: Align all sources to 15m
# ────────────────────────────────────────────────────────────────────
echo ""
echo "📐 [2/6] Aligning all sources to 15m..."
$PYTHON -c "
from src.ts_aligner import align
assets = [
    'market:binance:btcusdt',
    'sentiment:alternative:fear_greed',
    'derivatives:binance:funding_rate',
]
for a in assets:
    try:
        n = align(a, to_freq='15m')
        print(f'  {a}: {n} rows')
    except Exception as e:
        print(f'  {a}: skipped ({e})')
from src.ts_catalog import refresh_catalog
refresh_catalog()
" 2>&1

# ────────────────────────────────────────────────────────────────────
# STEP 3: Compute whale activity proxy (no API key needed)
# ────────────────────────────────────────────────────────────────────
echo ""
echo "🐋 [3/6] Computing whale activity proxy..."
$PYTHON scripts/register_whale_proxy.py 2>&1 | tail -5

# ────────────────────────────────────────────────────────────────────
# STEP 4: Fetch CryptoPanic news (optional)
# ────────────────────────────────────────────────────────────────────
if [ "$SKIP_NEWS" = false ] && [ -n "${CRYPTOPANIC_API_KEY}" ]; then
    echo ""
    echo "📰 [4/6] Fetching CryptoPanic news..."
    CRYPTOPANIC_API_KEY="$CRYPTOPANIC_API_KEY" \
      $PYTHON scripts/register_cryptopanic.py --pages 5 2>&1 | tail -5
elif [ "$SKIP_NEWS" = false ] && [ -z "${CRYPTOPANIC_API_KEY}" ]; then
    echo ""
    echo "📰 [4/6] News: skipped (no CRYPTOPANIC_API_KEY)"
else
    echo ""
    echo "📰 [4/6] News: skipped (--no-news)"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 5: Refresh funding rate
# ────────────────────────────────────────────────────────────────────
echo ""
echo "🔄 [5/6] Refreshing funding rate..."
$PYTHON scripts/register_funding_rate.py 2>&1 | tail -3

# ────────────────────────────────────────────────────────────────────
# STEP 6: Launch MCP Agent (overnight)
# ────────────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🏁 [6/6] DRY RUN — setup complete, agent NOT launched."
    echo ""
    echo "To launch manually:"
    echo "  nohup env OPENROUTER_API_KEY=\"\$OPENROUTER_API_KEY\" \\"
    echo "    PYTHONPATH=\"\$PWD\" .venv/bin/python3 -m src.brain.mcp_agent \\"
    echo "    --rounds 5 --strategies 5 > data/mcp_overnight.log 2>&1 &"
    echo ""
    echo "Status:"
    $PYTHON -c "
from src.ts_catalog import get_catalog
cat = get_catalog()
for _, r in cat.iterrows():
    print(f'  {r[\"asset_id\"]:45s}  {r[\"frequency\"]:6s}  {r[\"rows\"]:>8,} rows')
" 2>&1
    echo ""
    echo "To check agent progress: tail -f data/mcp_overnight.log"
    exit 0
fi

# ────────────────────────────────────────────────────────────────────
# Verify API key
# ────────────────────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY}" ]; then
    echo ""
    echo "❌ OPENROUTER_API_KEY not set. Set it and re-run."
    echo "   export OPENROUTER_API_KEY=\"sk-or-...\""
    exit 1
fi

echo ""
echo "🤖 [6/6] Launching MCP Agent (5 rounds, 5 strategies each)..."
echo "    Logs: data/mcp_overnight.log"
echo "    PID:  $$"
echo ""

# Set up notification — use a simple file-based status tracker
mkdir -p data/overnight
echo "STARTED: $(date)" > data/overnight/status.txt

# Launch agent in background
nohup env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  PYTHONPATH="$PROJECT_DIR" \
  $PYTHON -m src.brain.mcp_agent \
  --rounds 5 --strategies 5 \
  > "$PROJECT_DIR/data/mcp_overnight.log" 2>&1 &

AGENT_PID=$!
echo "PID: $AGENT_PID" >> data/overnight/status.txt
echo "Agent PID: $AGENT_PID"

# ────────────────────────────────────────────────────────────────────
# Health check monitor (runs for 2 min to verify startup)
# ────────────────────────────────────────────────────────────────────
echo ""
echo "⏳ Monitoring startup (60s)..."
for i in $(seq 1 12); do
    sleep 5
    if kill -0 $AGENT_PID 2>/dev/null; then
        LOG_SIZE=$(wc -c < data/mcp_overnight.log 2>/dev/null || echo 0)
        echo "  Running... (${i}x5s, log=${LOG_SIZE}b)"
        if grep -q "Round 1" data/mcp_overnight.log 2>/dev/null; then
            echo "  ✅ Agent started successfully!"
            echo "RUNNING: $(date) | PID=$AGENT_PID" > data/overnight/status.txt
            break
        fi
    else
        echo "  ❌ Agent exited! Check data/mcp_overnight.log"
        echo "FAILED: $(date)" > data/overnight/status.txt
        tail -20 data/mcp_overnight.log
        exit 1
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   🌙 Overnight Research Running!                           ║"
echo "║                                                            ║"
echo "║   PID:       $AGENT_PID                                           ║"
echo "║   Log:       data/mcp_overnight.log                       ║"
echo "║   Status:    tail -f data/mcp_overnight.log               ║"
echo "║                                                            ║"
echo "║   Morning check:                                           ║"
echo "║     grep 'saved' data/mcp_overnight.log                   ║"
echo "║     grep 'Sharpe' data/mcp_overnight.log | sort           ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "   📊 Available data for the agent:"
$PYTHON -c "
from src.ts_catalog import get_catalog
cat = get_catalog()
for _, r in cat.iterrows():
    print(f\"      {r['asset_id']:45s}  {r['frequency']:6s}  {r['rows']:>8,} rows\")
" 2>&1

# ────────────────────────────────────────────────────────────────────
# Wait for agent to finish (keeps script alive)
# ────────────────────────────────────────────────────────────────────
echo ""
echo "   Waiting for agent to complete (Ctrl+C to detach)..."
echo ""
wait $AGENT_PID 2>/dev/null
echo "SAVED: $(date)" > data/overnight/status.txt
echo "✅ Agent completed!"

echo ""
echo "=== RESULTS ==="
grep -i "saved\|sharpe\|round" data/mcp_overnight.log 2>/dev/null | tail -20 || true
echo ""
echo "Full log: data/mcp_overnight.log"
