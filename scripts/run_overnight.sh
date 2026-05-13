#!/usr/bin/env bash
# Overnight research run — Phase 24 Foundation Fix validation
#
# Runs the full orchestrator pipeline with maximum parameters.
# Each round generates hypotheses, filters, backtests, and feeds
# best results into the next round (evolutionary feedback loop).
#
# Usage:
#   nohup bash scripts/run_overnight.sh > results/overnight/20260514/master.log 2>&1 &
#   tail -f results/overnight/20260514/master.log

set -euo pipefail

SESSION_DIR="results/overnight/20260514"
mkdir -p "$SESSION_DIR"

# ── Configuration ──────────────────────────────────────────────────
DAYS=365
RESAMPLE="15m"
N_STRATEGIES=10
ROUNDS=15
SYMBOL="btcusdt"
MAX_RETRIES=3
RETRY_DELAY=120  # seconds between retries on failure

# ── Load env ────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  OVERNIGHT RESEARCH RUN — Phase 24 Foundation Fix"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Config: days=$DAYS resample=$RESAMPLE n=$N_STRATEGIES rounds=$ROUNDS"
echo "  Output: $SESSION_DIR/"
echo "════════════════════════════════════════════════════════════════"

TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')
LOGFILE="$SESSION_DIR/run_${TIMESTAMP}.log"

# ── Pre-flight checks ───────────────────────────────────────────────
echo "[pre-flight] Checking data availability..."
PYTHON=".venv/bin/python3"
ROW_COUNT=$($PYTHON -c "
from src.ts_store import read as ts_read
import pandas as pd
df = ts_read('market:binance:btcusdt', frequency='15m', limit=60000)
cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=$DAYS)
df = df[df['ts'] >= cutoff]
print(len(df))
")
echo "[pre-flight] Data rows available: $ROW_COUNT"

if [ "$ROW_COUNT" -lt 5000 ]; then
    echo "WARNING: Only $ROW_COUNT rows available. Run may be limited."
fi

echo "[pre-flight] Saving git state..."
git log -1 --oneline > "$SESSION_DIR/git_state.txt" 2>/dev/null || true
git diff --stat >> "$SESSION_DIR/git_state.txt" 2>/dev/null || true

# ── Run orchestrator ────────────────────────────────────────────────
echo "[run] Starting orchestrator with $ROUNDS rounds..."
echo "[run] Log: $LOGFILE"

ATTEMPT=0
SUCCESS=false

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[run] Attempt $ATTEMPT/$MAX_RETRIES — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    $PYTHON -u -m src.brain.orchestrator \
        --n "$N_STRATEGIES" \
        --days "$DAYS" \
        --resample "$RESAMPLE" \
        --rounds "$ROUNDS" \
        --symbol "$SYMBOL" \
        2>&1 | tee "$LOGFILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[run] Completed successfully at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        SUCCESS=true
        break
    else
        echo "[run] Failed with exit code $EXIT_CODE at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo "[run] Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    fi
done

# ── Post-run analysis ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  POST-RUN ANALYSIS"
echo "════════════════════════════════════════════════════════════════"

# Copy leaderboard to results
if [ -f "data/parquet/research/leaderboard.parquet" ]; then
    cp "data/parquet/research/leaderboard.parquet" "$SESSION_DIR/leaderboard_snapshot.parquet"
    echo "[post] Leaderboard copied to $SESSION_DIR/leaderboard_snapshot.parquet"
fi

# Copy survival log
if [ -f "data/ts/_survival.jsonl" ]; then
    cp "data/ts/_survival.jsonl" "$SESSION_DIR/survival.jsonl"
    echo "[post] Survival log copied"
fi

# Copy concept stats
if [ -f "data/ts/_concept_stats.jsonl" ]; then
    cp "data/ts/_concept_stats.jsonl" "$SESSION_DIR/concept_stats.jsonl"
    echo "[post] Concept stats copied"
fi

# Generate summary
$PYTHON -u -c "
import json
from pathlib import Path

session = Path('$SESSION_DIR')

# Read concept stats
concept_path = session / 'concept_stats.jsonl'
stats = []
if concept_path.exists():
    with open(concept_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    stats.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

# Read survival
survival_path = session / 'survival.jsonl'
survival = []
if survival_path.exists():
    with open(survival_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    survival.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

# Read leaderboard
try:
    import pandas as pd
    lb_path = session / 'leaderboard_snapshot.parquet'
    if lb_path.exists():
        lb = pd.read_parquet(lb_path)
        top5 = lb.nlargest(5, 'sharpe') if 'sharpe' in lb.columns else lb.head(5)
        print('=== TOP 5 STRATEGIES ===')
        for _, r in top5.iterrows():
            of_tag = ' ⚠️OVERFIT' if r.get('overfit', False) else ''
            wf_tag = f' WF={r.get(\"wf_sharpe\", 0):.2f}' if r.get('wf_sharpe', 0) != 0 else ''
            oos_tag = f' OOS={r.get(\"oos_sharpe\", 0):.2f}' if r.get('oos_trades', 0) >= 10 else ''
            print(f'  {r.get(\"run_id\", \"?\")[:40]:40s}  S={r.get(\"sharpe\", 0):+.2f}  T={r.get(\"n_trades\", 0)}  WR={r.get(\"win_rate\", 0):.0%}{oos_tag}{wf_tag}{of_tag}')
        print()
        print(f'Total strategies tested: {len(lb)}')
        positive = len(lb[lb['sharpe'] > 0]) if 'sharpe' in lb.columns else 0
        print(f'Positive Sharpe: {positive}/{len(lb)}')
        passed_oos = len(lb[(lb.get('oos_trades', 0) >= 10) & (~lb.get('overfit', False))]) if 'oos_trades' in lb.columns else 0
        print(f'Passed OOS: {passed_oos}')
    else:
        print('No leaderboard data found')
except Exception as e:
    print(f'Error reading leaderboard: {e}')

# Survival summary
if survival:
    last = survival[-1]
    print(f'\\nLast survival: generated={last.get(\"generated\", 0)} fast={last.get(\"fast_filter_pass\", 0)} wf={last.get(\"wf_pass\", 0)} oos={last.get(\"oos_pass\", 0)} overfit={last.get(\"overfit\", 0)}')

# Concept stats summary
if stats:
    by_family = {}
    for s in stats:
        fam = s.get('archetype', 'unknown')
        by_family.setdefault(fam, []).append(s.get('reward', 0))
    print('\\n=== CONCEPT STATS ===')
    for fam, rewards in sorted(by_family.items()):
        positive = [r for r in rewards if r > 0]
        avg_pos = sum(positive) / len(positive) if positive else 0
        print(f'  {fam}: {len(rewards)} attempts, avg positive reward={avg_pos:.3f}')
" 2>&1 | tee "$SESSION_DIR/summary.txt"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  COMPLETED: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Success: $SUCCESS"
echo "  Log: $LOGFILE"
echo "  Summary: $SESSION_DIR/summary.txt"
echo "════════════════════════════════════════════════════════════════"