# VISION — Autonomous Quantitative Research Platform

## Strategic Pivot

Classical indicators (RSI, MACD, EMA, ATR, ADX) on 15m BTCUSDT produce **no edge**.
37 strategies tested, 0 survive OOS. Best Sharpe = -0.146 (baseline random = -0.38).

The system is **correctly rejecting weak hypotheses** — this is a feature, not a bug.
The next leap requires **semantic market features** that classical TA cannot capture.

## Architecture

```
Raw Data → Feature Extraction → Semantic Market State → Hypothesis Generation → Search/Mutation → Robustness Validation → Concept Memory → Adaptive Exploration
```

### Layer 1: Raw Data (exists)
- OHLCV from Binance (1m → 15m/1h/1d)
- Fear & Greed Index (daily, ffill to 15m)

### Layer 2: Feature Extraction (exists + planned)
- Classical: RSI, MACD, EMA, VWAP, OBI, HA, ATR, ADX, regime
- New (planned):
  - **Funding_rate**: perpetual futures funding (sentiment proxy)
  - **Open_interest**: total OI (crowding proxy)
  - **Liquidation_volume**: estimated liquidations (cascade proxy)
  - **Exchange_flow**: net exchange deposits/withdrawals (whale proxy)
  - **Fear_greed_regime**: extreme fear/greed persistence (sentiment regime)

### Layer 3: Semantic Market State (planned)
Each bar gets a vector of semantic features:
- `crowded_longs`: funding_rate > threshold & OI rising
- `volatility_compression`: ATR ratio low + volume declining
- `panic_sentiment`: fear_greed < 20 & price dropping fast
- `liquidity_void`: volume suddenly absent + spread widening
- `squeeze_probability`: OI high + funding extreme + price near range boundary
- `trend_exhaustion`: momentum declining while volume diverges

### Layer 4: Hypothesis Generation (exists, needs expansion)
- LLM Strategist generates hypotheses from families
- New families needed: LIQUIDATION_CASCADE, CROWDED_POSITIONING, FUNDING_REVERSAL
- Families must reference semantic features, not just classical indicators

### Layer 5: Search & Mutation (exists)
- expand_hypothesis() generates 5-15 variants per family
- Vectorized evaluator tests each in ~6ms
- Baseline comparator rejects strategies weaker than random

### Layer 6: Robustness Validation (exists)
- Walk-forward (expanding window, 4 folds)
- OOS holdout test (20% of data, never seen)
- Overfit detection (train/test ratio)
- Max DD gate (OOS)

### Layer 7: Concept Memory (planned)
- Semantic embeddings for strategy similarity (sentence-transformers)
- Episodic memory: which hypotheses survived/died in which regimes
- Reward function: P x R x N (Performance x Robustness x Novelty)

### Layer 8: Adaptive Exploration (planned)
- Explore families with higher expected reward
- Penalize redundant hypotheses (embedding similarity > threshold)
- Focus on regimes where previous hypotheses failed

## Current Results Summary

| Metric | Value |
|--------|-------|
| Total strategies | 37 |
| Best Sharpe (train) | -0.146 |
| Best OOS Sharpe | 0.000 (no trades) |
| WF pass rate | 0/37 (0%) |
| Overfit rate | 37/37 (100%) |
| Baseline random | mean=-0.51, std=0.13 |
| Best vs baseline | Marginal (-0.146 vs -0.380 threshold) |

**Conclusion**: Classical indicators on BTCUSDT 15m cannot produce positive OOS Sharpe.
We need new data dimensions (funding, OI, liquidations) to discover real anomalies.

## Implementation Priority

1. **Funding rates** — most actionable, available from Binance Futures API
2. **Open Interest** — crowding signal, available from Binance
3. **Fear & Greed** — already have data, need to wire into features properly
4. **Liquidation estimates** — derived from OI + funding + volatility
5. **Exchange flows** — CryptoQuant/WhaleAlert (may need API key)

## Reward Function (Target)

```
reward = P(sharpe_oos) × R(1 - overfit_ratio) × N(1 - similarity_to_known)
```

Where:
- P = Performance: OOS Sharpe normalized by baseline
- R = Robustness: 1 - (train_sharpe - oos_sharpe) / train_sharpe
- N = Novelty: 1 - cos_similarity(strategy_embedding, known_best)

Current state: P and R are implemented (concept_stats.py). N requires embeddings (Phase 27).