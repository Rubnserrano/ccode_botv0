#!/bin/bash
# Resilient research loop: runs orchestrator in a loop.
# If it crashes, it restarts automatically.
# Runs for ~1 hour (3600s total).
set -e

API_KEY="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"
MAX_SECONDS=3600
START_TS=$(date +%s)
ROUND=1

cd /home/rserrano/project/ccode_botv0

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    REMAINING=$((MAX_SECONDS - ELAPSED))
    if [ $ELAPSED -ge $MAX_SECONDS ]; then
        echo "=== Total time elapsed ($MAX_SECONDS s). Exiting. ==="
        break
    fi

    echo ""
    echo "============================================"
    echo "  Round $ROUND — ${ELAPSED}s elapsed, ${REMAINING}s remaining"
    echo "============================================"
    echo ""

    docker compose exec -T -e OPENROUTER_API_KEY="$API_KEY" app \
        timeout $REMAINING python -m src.brain.orchestrator \
        --n 10 --days 365 --resample 15m --rounds 1 \
        --api-key "$API_KEY"

    EXIT_CODE=$?
    echo "  Round $ROUND finished with exit code $EXIT_CODE"
    ROUND=$((ROUND + 1))

    # Small pause between rounds to let the container breathe
    sleep 3
done

# Final report
echo ""
echo "============================================"
echo "  RESEARCH COMPLETE — $ROUND rounds"
echo "============================================"
docker compose exec app python -c "
import pandas as pd
df = pd.read_parquet('data/parquet/research/leaderboard.parquet')
brain = df[df['llm_model'].notna() & (df['llm_model'] != '') & (df['n_trades'] > 0)]
print(f'Brain strategies: {len(brain)}')
print(f'Sharpe > 0: {(brain[\"sharpe\"] > 0).sum()}')
print()
for _, r in brain.sort_values('sharpe', ascending=False).head(10).iterrows():
    e = str(r.get('llm_explanation',''))[:120] if str(r.get('llm_explanation','')) != 'nan' else ''
    print(f\"S={r['sharpe']:+.2f}  WR={r['win_rate']:.0%}  PF={r['profit_factor']:.2f}  PnL=\${r['total_pnl']:+.0f}  T={r['n_trades']}  {r['run_id']}\")
    if e: print(f\"  → {e}\")
"
