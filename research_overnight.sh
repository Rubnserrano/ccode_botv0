#!/bin/bash
# Overnight autonomous research loop.
#
# Runs the LLM research loop in a resilient loop:
#   - Auto-restarts on crash
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
MAX_SECONDS=28800  # 8 hours
START_TS=$(date +%s)
ROUND=1
PID_FILE="/tmp/overnight.pid"
echo $$ > "$PID_FILE"

cd /home/rserrano/project/ccode_botv0

# Ensure Telegram env vars are passed through
export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID

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

    docker compose exec -T -e OPENROUTER_API_KEY="$API_KEY" \
        -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
        -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
        app timeout "$REMAINING" python -m src.brain.orchestrator \
        --n 15 --days 365 --resample 15m \
        --rounds 1 --api-key "$API_KEY"

    EXIT_CODE=$?
    echo "[$(date)] Round $ROUND finished with exit code $EXIT_CODE"

    ROUND=$((ROUND + 1))

    # If something went very wrong, wait before retrying
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] Crash detected! Restarting in 30s..."
        sleep 30
    else
        # Small pause between rounds
        sleep 5
    fi
done

echo "[$(date)] === Research finished after $ROUND rounds ==="
rm -f "$PID_FILE"
