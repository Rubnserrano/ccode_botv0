"""Quick pipeline test — bypasses LLM, tests full flow."""
import pandas as pd
from src.ts_store import read as ts_read
from src.indicators.calculator import calc_all
from src.strategy_engine.search_engine import expand_hypothesis
from src.strategy_engine.schema import validate_strategy
from src.strategy_engine.evaluator import evaluate
from src.backtesting.engine import backtest
from src.backtesting.walk_forward import walk_forward, walk_forward_summary, WalkForwardConfig
from src.backtesting.baseline import compute_baseline
from src.strategy_engine.generic_calculator import register_dynamic_indicator

df = ts_read('market:binance:btcusdt', frequency='15m', limit=50000)
cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=180)
df = df[df['ts'] >= cutoff]
df = calc_all(df)
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_test = df.iloc[split_idx:].reset_index(drop=True)
print(f'Train: {len(df_train):,} rows  Test: {len(df_test):,} rows', flush=True)

baseline = compute_baseline(df_train, n_trials=30, horizon=12, warmup=50, cooldown=2)
threshold = baseline['sharpe_mean'] + baseline['sharpe_std']
print(f'Baseline: S_mean={baseline["sharpe_mean"]:.3f}  S_std={baseline["sharpe_std"]:.3f}  threshold={threshold:.3f}', flush=True)

families = ['breakout_structure', 'volatility_compression_breakout', 'momentum_continuation']
all_strategies = []
for fam in families:
    expanded = expand_hypothesis(fam, n_variants=6)
    all_strategies.extend(expanded)

for s in all_strategies:
    ni = s.pop('new_indicator', None) or s.pop('_new_indicator', None)
    if ni and isinstance(ni, dict):
        try:
            register_dynamic_indicator(ni['name'], ni['formula'], ni.get('params'), ni.get('description', ''))
            print(f'  Reg: {ni["name"]}', flush=True)
        except Exception as e:
            print(f'  Fail: {ni.get("name")}: {e}', flush=True)

survivors = []
for sd in all_strategies:
    try:
        validated = validate_strategy(sd)
    except Exception:
        continue
    def _eval(d, _sd=validated):
        return evaluate(d, _sd)
    _, summary = backtest(df_train, _eval, horizon=validated.exit.horizon_bars, warmup=50, cooldown=2, size_usdc=50, tp_pct=validated.exit.tp_pct, sl_pct=validated.exit.sl_pct)
    sharpe = summary.get('sharpe', -999)
    n_t = summary.get('n_trades', 0)
    if n_t >= 3 and sharpe >= threshold:
        survivors.append((sd, validated, sharpe, n_t))
        print(f'  PASS: {sd["name"][:30]:30s}  S={sharpe:+.3f}  T={n_t}', flush=True)

print(f'\nFast filter: {len(survivors)}/{len(all_strategies)} survived (threshold={threshold:.3f})', flush=True)

wf_cfg = WalkForwardConfig(n_splits=4, horizon=12, warmup=50, cooldown=2, size_usdc=50)
for strategy_dict, sd, sharpe, n_t in survivors[:5]:
    def _eval_full(d, _sd=sd):
        return evaluate(d, _sd)
    wf_results = walk_forward(df_train, _eval_full, wf_cfg)
    wf_summary = walk_forward_summary(wf_results)
    _, test_summary = backtest(df_test, _eval_full, horizon=sd.exit.horizon_bars, warmup=50, cooldown=2, size_usdc=50, tp_pct=sd.exit.tp_pct, sl_pct=sd.exit.sl_pct)
    print(f'  {strategy_dict["name"][:35]:35s}  Train S={sharpe:+.3f}  Test S={test_summary.get("sharpe",0):+.3f}  WF S={wf_summary.get("oos_sharpe_mean",0):+.3f}  folds={wf_summary.get("n_folds",0)}', flush=True)