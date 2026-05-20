"""Tests for walk-forward baseline, breakout, and hypothesis fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.walk_forward import walk_forward, WalkForwardConfig, walk_forward_summary
from src.backtesting.baseline import compute_baseline, random_strategy_fn
from src.strategy_engine.hypothesis import (
    HypothesisFamily, get_template, list_families,
)
from src.strategy_engine.search_engine import expand_hypothesis, _build_strategy
from src.backtesting.engine import backtest


def _make_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    prices = 50000.0 + np.cumsum(rng.randn(n) * 50)
    prices = np.maximum(prices, 1000)
    return pd.DataFrame({
        "ts": dates,
        "open": prices + rng.randn(n) * 10,
        "high": prices + np.abs(rng.randn(n)) * 50,
        "low": prices - np.abs(rng.randn(n)) * 50,
        "close": prices,
        "volume": rng.uniform(10, 500, n),
        "rsi_14": rng.uniform(20, 80, n),
        "macd": rng.randn(n) * 0.5,
        "macd_signal": rng.randn(n) * 0.3,
        "macd_hist": rng.randn(n) * 0.2,
        "ema_9": prices + rng.randn(n) * 20,
        "ema_21": prices + rng.randn(n) * 40,
        "ema_50": prices + rng.randn(n) * 60,
        "vwap": prices + rng.randn(n) * 30,
        "obi": rng.randn(n) * 0.01,
        "atr_14": rng.uniform(200, 600, n),
        "adx_14": rng.uniform(15, 50, n),
        "regime": rng.choice([0, 1, 2, 3], n),
    })


class TestWalkForwardExpandingWindow:
    def test_fold_zero_has_training_data(self):
        df = _make_df(2000)
        cfg = WalkForwardConfig(n_splits=4, horizon=12, warmup=50)

        def _dummy(df_inner):
            return pd.Series(0, index=df_inner.index, dtype=int)

        results = walk_forward(df, _dummy, cfg)
        assert len(results) > 0, "Walk-forward should produce results"

        first_fold = results[0]
        assert first_fold.fold == 0
        assert len(pd.date_range(first_fold.train_start, first_fold.train_end)) > 0
        assert first_fold.train_start < first_fold.test_start

    def test_all_folds_have_training(self):
        df = _make_df(3000)
        cfg = WalkForwardConfig(n_splits=4, horizon=12, warmup=50)

        def _dummy(df_inner):
            return pd.Series(0, index=df_inner.index, dtype=int)

        results = walk_forward(df, _dummy, cfg)
        for r in results:
            assert r.train_start < r.test_start, \
                f"Fold {r.fold}: train must start before test (train_start={r.train_start}, test_start={r.test_start})"

    def test_no_fold_with_zero_rows(self):
        df = _make_df(2000)
        cfg = WalkForwardConfig(n_splits=4, horizon=12, warmup=50)

        def _dummy(df_inner):
            return pd.Series(0, index=df_inner.index, dtype=int)

        results = walk_forward(df, _dummy, cfg)
        for r in results:
            assert r.test_start is not None
            assert r.test_end is not None


class TestBaseline:
    def test_compute_baseline_returns_dict(self):
        df = _make_df(500)
        baseline = compute_baseline(df, n_trials=5, horizon=12)
        assert "sharpe_mean" in baseline
        assert "sharpe_std" in baseline
        assert "sharpe_p95" in baseline
        assert isinstance(baseline["sharpe_mean"], float)

    def test_random_strategy_fn_produces_signals(self):
        df = _make_df(500)
        fn = random_strategy_fn(df, entry_freq=0.05, seed=0)
        signals = fn(df)
        assert len(signals) == len(df)
        assert signals.sum() > 0

    def test_baseline_sharpe_near_zero(self):
        df = _make_df(2000)
        baseline = compute_baseline(df, n_trials=20, horizon=12)
        assert abs(baseline["sharpe_mean"]) < 2.0, \
            f"Random baseline Sharpe should be near 0, got {baseline['sharpe_mean']}"


class TestBreakoutFix:
    def test_breakout_uses_custom_indicator_not_raw_close(self):
        template = get_template(HypothesisFamily.BREAKOUT_STRUCTURE)
        conditions = template.get("conditions_template", [])
        breakout_conds = [c for c in conditions if "breakout" in c.get("indicator", "")]
        assert len(breakout_conds) > 0, "Breakout should use custom indicator, not raw close"

        breakout_cond = breakout_conds[0]
        assert breakout_cond["indicator"] == "breakout_pct_above_high", \
            f"Expected breakout_pct_above_high, got {breakout_cond['indicator']}"

    def test_breakout_has_feature_requirements(self):
        template = get_template(HypothesisFamily.BREAKOUT_STRUCTURE)
        freqs = template.get("feature_requirements", [])
        assert len(freqs) > 0, "Breakout must have feature_requirements"
        assert freqs[0]["name"] == "breakout_pct_above_high"

    def test_expand_breakout_generates_new_indicator(self):
        strategies = expand_hypothesis("breakout_structure", n_variants=4)
        assert len(strategies) > 0
        has_new_indicator = any(s.get("new_indicator") is not None for s in strategies)
        assert has_new_indicator, "Expanded breakout strategies should have new_indicator"

    def test_breakout_condition_value_is_percentage(self):
        strategies = expand_hypothesis("breakout_structure", n_variants=4)
        for s in strategies:
            breakout_conds = [c for c in s.get("entry_conditions", [])
                              if "breakout" in c.get("indicator", "")]
            assert len(breakout_conds) > 0
            val = breakout_conds[0].get("value")
            if isinstance(val, (int, float)):
                assert 0 < val < 1, f"Breakout pct should be a small fraction (0-1), got {val}"


class TestVolatilityCompressionFix:
    def test_has_compression_ratio_indicator(self):
        template = get_template(HypothesisFamily.VOLATILITY_COMPRESSION_BREAKOUT)
        conditions = template.get("conditions_template", [])
        indicators = [c.get("indicator", "") for c in conditions]
        assert "atr_compression_ratio" in indicators, \
            f"Should have atr_compression_ratio, got {indicators}"

    def test_has_expansion_confirmation(self):
        template = get_template(HypothesisFamily.VOLATILITY_COMPRESSION_BREAKOUT)
        conditions = template.get("conditions_template", [])
        roller_conds = [c for c in conditions if c.get("op") in ("gt_rolling", "lt_rolling")
                        and c.get("indicator") == "atr"]
        assert len(roller_conds) > 0, "Should have ATR expansion confirmation condition"

    def test_has_feature_requirements(self):
        template = get_template(HypothesisFamily.VOLATILITY_COMPRESSION_BREAKOUT)
        freqs = template.get("feature_requirements", [])
        assert len(freqs) > 0
        assert "atr_compression_ratio" in freqs[0]["name"]

    def test_expand_generates_indicator(self):
        strategies = expand_hypothesis("volatility_compression_breakout", n_variants=4)
        assert len(strategies) > 0
        has_new_indicator = any(s.get("new_indicator") is not None for s in strategies)
        assert has_new_indicator


class TestStrategyDryRun:
    def test_expand_all_families(self):
        for family in list_families():
            strategies = expand_hypothesis(family, n_variants=4)
            assert len(strategies) > 0, f"Family {family} should generate strategies"

    def test_walk_forward_summary_empty(self):
        result = walk_forward_summary([])
        assert "error" in result