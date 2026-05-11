#!/bin/bash
# Overnight autonomous research loop — resilient version.
#
# Features:
#   - Auto-restarts on crash (max 3 retries per round)
#   - 10 min per-round timeout (never hangs forever)
#   - Container health check before each round
#   - Sends Telegram notifications on findings
#   - Runs for ~8 hours or until stopped
#
# Usage:
#   TELEGRAM_BOT_TOKEN="xxx" TELEGRAM_CHAT_ID="yyy" \
#   OPENROUTER_API_KEY="sk-or-..." \
#     nohup bash research_overnight.sh &
#
# Logs: /tmp/overnight.log
# To stop: kill $(cat /tmp/overnight.pid)
set -e

API_KEY="${OPENROUTER_API_KEY}"
if [ -z "$API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY no está definida."
    echo "Usa: OPENROUTER_API_KEY='sk-or-...' nohup bash research_overnight.sh &"
    exit 1
fi

MAX_SECONDS=28800       # 8 hours total
ROUND_TIMEOUT=600       # 10 min per round (never hangs longer)
MAX_RETRIES=3           # max retries per round before skipping
START_TS=$(date +%s)
ROUND=1
PID_FILE="/tmp/overnight.pid"
RETRIES=0
echo $$ > "$PID_FILE"

cd /home/rserrano/project/ccode_botv0
export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID
export OPENROUTER_API_KEY

cleanup() {
    echo "[$(date)] === Received signal. Cleaning up... ==="
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    REMAINING=$((MAX_SECONDS - ELAPSED))

    if [ $ELAPSED -ge $MAX_SECONDS ]; then
        echo "[$(date)] === Max time reached ($MAX_SECONDS s). Exiting. ==="
        break
    fi

    echo ""
    echo "[$(date)] === Round $ROUND — ${ELAPSED}s elapsed, ${REMAINING}s remaining ==="
    echo ""

    # Container health check
    if ! docker compose ps app 2>/dev/null | grep -q "healthy"; then
        echo "[$(date)] Container not healthy. Restarting..."
        docker compose restart app 2>/dev/null || docker compose up -d app 2>/dev/null
        sleep 15
    fi

    # Run one round with forced timeout
    ROUND_START=$(date +%s)
    timeout "$ROUND_TIMEOUT" docker compose exec -T \
        -e OPENROUTER_API_KEY="$API_KEY" \
        -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
        -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
        app python -m src.brain.orchestrator \
        --n 15 --days 365 --resample 15m \
        --rounds 1 --api-key "$API_KEY"

    EXIT_CODE=$?
    ROUND_ELAPSED=$(($(date +%s) - ROUND_START))
    echo "[$(date)] Round $ROUND finished in ${ROUND_ELAPSED}s (exit code $EXIT_CODE)"

    ROUND=$((ROUND + 1))

    if [ $EXIT_CODE -eq 0 ]; then
        RETRIES=0
        sleep 3
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] ⚠ Round timed out after ${ROUND_TIMEOUT}s. Retrying..."
        RETRIES=$((RETRIES + 1))
        sleep 5
    else
        echo "[$(date)] ⚠ Round crashed (code $EXIT_CODE). Retrying..."
        RETRIES=$((RETRIES + 1))
        sleep 10
    fi

    # Max retries: skip this round and move on
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "[$(date)] ❌ Max retries reached. Moving on..."
        RETRIES=0
        sleep 30
    fi
done

echo "[$(date)] === Research finished after $ROUND rounds ==="
rm -f "$PID_FILE"
