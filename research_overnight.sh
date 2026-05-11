#!/bin/bash
# Overnight autonomous research loop — resilient v3.
#
# Wraps each round in a subshell to isolate signals.
# Logs signal type for debugging.
#
# Usage:
#   TELEGRAM_BOT_TOKEN="xxx" TELEGRAM_CHAT_ID="yyy" \
#   OPENROUTER_API_KEY="sk-or-..." \
#     nohup bash research_overnight.sh >> /tmp/overnight.log 2>&1 &
#
# To stop: kill $(cat /tmp/overnight.pid)

API_KEY="${OPENROUTER_API_KEY}"
if [ -z "$API_KEY" ]; then
    echo "[$(date)] ERROR: OPENROUTER_API_KEY no definida"
    echo "[$(date)Uso: OPENROUTER_API_KEY='sk-or-...' nohup bash research_overnight.sh &"
    exit 1
fi

MAX_SECONDS=28800       # 8h
ROUND_TIMEOUT=600       # 10min max per round
MAX_RETRIES=3
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
    echo "[$(date)] === CLEANUP exit ==="
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    REMAINING=$((MAX_SECONDS - ELAPSED))

    if [ $ELAPSED -ge $MAX_SECONDS ]; then
        echo "[$(date)] === Max time reached ($MAX_SECONDS s) ==="
        break
    fi

    echo ""
    echo "=== Round $ROUND — ${ELAPSED}s elapsed, ${REMAINING}s remaining ==="
    echo ""

    # Container health check
    if ! docker compose ps app 2>/dev/null | grep -q "healthy"; then
        echo "[$(date)] ⚠ Container not healthy. Restarting..."
        docker compose restart app 2>/dev/null || docker compose up -d app 2>/dev/null
        sleep 15
    fi

    # Run ONE round in a subshell to isolate signals
    ROUND_START=$(date +%s)
    (
        exec timeout "$ROUND_TIMEOUT" docker compose exec -T \
            -e OPENROUTER_API_KEY="$API_KEY" \
            -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
            -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
            app python -m src.brain.orchestrator \
            --n 15 --days 365 --resample 15m \
            --rounds 1 --digest-minutes 5 --api-key "$API_KEY"
    )
    EXIT_CODE=$?
    ROUND_ELAPSED=$(($(date +%s) - ROUND_START))

    echo "[$(date)] Round $ROUND done in ${ROUND_ELAPSED}s (exit=$EXIT_CODE)"

    ROUND=$((ROUND + 1))

    if [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] ⏱ Timeout after ${ROUND_TIMEOUT}s"
        RETRIES=$((RETRIES + 1))
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] ⚠ Crash (code $EXIT_CODE)"
        RETRIES=$((RETRIES + 1))
    else
        RETRIES=0
    fi

    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "[$(date)] ❌ Max retries. Moving to next round."
        RETRIES=0
        sleep 10
    else
        sleep 2
    fi
done

echo "[$(date)] === Research finished ==="
rm -f "$PID_FILE"
