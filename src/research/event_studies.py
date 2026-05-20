"""Event studies — measure predictive signal in features.

Core scientific tool for the Feature Validation Phase.
Tests whether specific market conditions predict future returns.

Event and condition strings are evaluated via pd.DataFrame.eval(),
ensuring reproducibility and serializability.

Usage:
    from src.research.event_studies import run_event_study, print_result

    result = run_event_study(
        df,
        event="funding_rate_zscore > 2",
        condition="regime == 2",
        horizons=[4, 16, 48, 96],
        side="short",
    )
    print_result(result)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUAL_BARS = {"15m": 96 * 365, "1h": 24 * 365, "5m": 288 * 365}


@dataclass
class HorizonResult:
    n_events: int
    mean_return: float
    median_return: float
    std_return: float
    hit_ratio: float
    sharpe_annualized: float
    p_value: float
    ci_lower: float
    ci_upper: float
    skew: float
    kurtosis: float
    min_return: float
    max_return: float
    t_stat: float


@dataclass
class BaselineResult:
    n_samples: int
    horizon_results: dict[int, HorizonResult] = field(default_factory=dict)


@dataclass
class EventStudyResult:
    config: dict[str, Any]
    n_events_train: int
    n_events_test: int
    n_total: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_results: dict[str, dict] = field(default_factory=dict)
    test_results: dict[str, dict] = field(default_factory=dict)
    baseline_results: dict[str, dict] = field(default_factory=dict)
    regime_breakdown: dict[str, dict] = field(default_factory=dict)
    significant_horizons: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    feature: str
    threshold: float
    direction: str
    n_train: int
    n_test: int
    test_mean_return: float
    test_hit_ratio: float
    test_sharpe: float
    test_p_value: float
    significant: bool
    regime: str
    horizon: int


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int],
    price_col: str = "close",
) -> pd.DataFrame:
    """Add fwd_return_{h} columns for each horizon.

    fwd_return_h = close[t+h] / close[t] - 1

    Rows near the end where close[t+h] doesn't exist get NaN.
    """
    result = df.copy()
    for h in horizons:
        result[f"fwd_ret_{h}"] = (
            result[price_col].shift(-h) / result[price_col] - 1
        )
    return result


def _compute_horizon_stats(
    returns: pd.Series,
    baseline_returns: np.ndarray | None = None,
    confidence_level: float = 0.95,
    hourly_bars: int = 96,
    annual_bars: int | None = None,
) -> dict:
    """Compute statistics for a series of forward returns."""
    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return {
            "n_events": n,
            "mean_return": 0.0,
            "median_return": 0.0,
            "std_return": 0.0,
            "hit_ratio": 0.0,
            "sharpe_annualized": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "min_return": 0.0,
            "max_return": 0.0,
            "t_stat": 0.0,
        }

    mean_r = float(returns.mean())
    median_r = float(returns.median())
    std_r = float(returns.std())
    hit = float((returns > 0).mean())

    if annual_bars is None:
        annual_bars = ANNUAL_BARS.get("15m", 96 * 365)
    sharpe = (mean_r / std_r * np.sqrt(annual_bars)) if std_r > 1e-10 else 0.0

    skew = float(returns.skew()) if n >= 3 else 0.0
    kurt = float(returns.kurtosis()) if n >= 3 else 0.0
    min_r = float(returns.min())
    max_r = float(returns.max())

    # Bootstrap CI
    np.random.seed(42)
    n_boot = 1000
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_means[i] = returns.iloc[idx].mean()
    alpha = (1 - confidence_level) / 2
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

    # T-statistic for mean != 0
    t_stat = float(mean_r / (std_r / np.sqrt(n))) if std_r > 1e-10 else 0.0

    # P-value vs baseline (if provided)
    p_value = 1.0
    if baseline_returns is not None and len(baseline_returns) > 1:
        above = float(np.mean(baseline_returns >= mean_r))
        below = float(np.mean(baseline_returns <= mean_r))
        p_value = 2 * min(above, below)
        p_value = min(p_value, 1.0)

    return {
        "n_events": n,
        "mean_return": round(mean_r, 8),
        "median_return": round(median_r, 8),
        "std_return": round(std_r, 8),
        "hit_ratio": round(hit, 4),
        "sharpe_annualized": round(sharpe, 4),
        "p_value": round(p_value, 4),
        "ci_lower": round(ci_lower, 8),
        "ci_upper": round(ci_upper, 8),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "min_return": round(min_r, 8),
        "max_return": round(max_r, 8),
        "t_stat": round(t_stat, 4),
    }


def _prepare_data(
    df: pd.DataFrame,
    horizons: list[int],
    price_col: str = "close",
) -> pd.DataFrame:
    """Prepare data: add forward returns, split train/test."""
    if horizons is None or len(horizons) == 0:
        horizons = [4, 16, 48, 96, 192, 288]
    df = compute_forward_returns(df, horizons, price_col)
    return df


def _compute_event_mask(
    df: pd.DataFrame,
    event: str,
    condition: str | None = None,
) -> pd.Series:
    """Compute boolean mask for events.

    Ignores rows where the event condition depends on NaN values.
    """
    try:
        mask = df.eval(event)
    except Exception as e:
        raise ValueError(f"Failed to evaluate event '{event}': {e}") from e

    if mask.dtype != bool:
        mask = mask.astype(bool)

    if condition is not None and condition.strip():
        try:
            cond_mask = df.eval(condition)
            if cond_mask.dtype != bool:
                cond_mask = cond_mask.astype(bool)
            mask = mask & cond_mask
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate condition '{condition}': {e}"
            ) from e

    return mask.fillna(False)


def _run_baseline(
    df: pd.DataFrame,
    horizons: list[int],
    n_samples: int = 1000,
    condition: str | None = None,
    regime_aware: bool = True,
    price_col: str = "close",
    min_index: int = 50,
) -> dict[int, np.ndarray]:
    """Run randomized baseline to compare against events.

    Returns dict of horizon -> array of mean returns from each baseline trial.
    """
    max_idx = len(df) - max(horizons)
    if max_idx <= min_index:
        raise ValueError(
            f"DataFrame too short ({len(df)} rows) for horizons {horizons}"
        )

    valid_indices = np.arange(min_index, max_idx)

    if condition is not None and condition.strip():
        try:
            cond_mask = df.eval(condition)
            if cond_mask.dtype != bool:
                cond_mask = cond_mask.astype(bool)
            cond_arr = cond_mask.fillna(False).values
            cond_idxs = np.where(cond_arr[min_index:max_idx])[0] + min_index
            if len(cond_idxs) > 0:
                valid_indices = cond_idxs
        except Exception:
            pass

    if regime_aware and "regime" in df.columns:
        unique_regimes = df["regime"].dropna().unique()
        if len(unique_regimes) > 1:
            regime_dist = df["regime"].value_counts(normalize=True).to_dict()
            n_by_regime = {
                r: int(n_samples * p) for r, p in regime_dist.items()
            }
            n_by_regime[list(n_by_regime.keys())[-1]] += (
                n_samples - sum(n_by_regime.values())
            )
            result = {h: [] for h in horizons}
            for regime, n_regime in n_by_regime.items():
                regime_idxs = valid_indices[
                    df.iloc[valid_indices]["regime"].values == regime
                ]
                if len(regime_idxs) < 10:
                    continue
                n_regime = min(n_regime, len(regime_idxs))
                for h in horizons:
                    col = f"fwd_ret_{h}"
                    if col not in df.columns:
                        continue
                    idx = np.random.choice(
                        regime_idxs, size=n_regime, replace=True
                    )
                    result[h].append(
                        df[col].iloc[idx].dropna().values
                    )
            return {
                h: np.concatenate(vals) if vals else np.array([])
                for h, vals in result.items()
            }

    result = {}
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in df.columns:
            continue
        idx = np.random.choice(
            valid_indices, size=n_samples, replace=True
        )
        result[h] = df[col].iloc[idx].dropna().values

    return result


def run_event_study(
    df: pd.DataFrame,
    event: str,
    horizons: list[int] | None = None,
    condition: str | None = None,
    side: str = "both",
    min_samples: int = 30,
    n_baseline: int = 1000,
    train_ratio: float = 0.7,
    confidence_level: float = 0.95,
    price_col: str = "close",
    regime_breakdown: bool = True,
    min_regime_samples: int = 10,
    verbose: bool = False,
) -> EventStudyResult:
    """Run an event study.

    Args:
        df: DataFrame with OHLCV + indicators + derivatives columns.
        event: Event string (e.g. "funding_rate_zscore > 2").
        horizons: Forward return horizons in bars.
        condition: Optional condition string (e.g. "regime == 2").
        side: "long", "short", or "both".
        min_samples: Minimum events required in both train and test.
        n_baseline: Number of random baseline samples.
        train_ratio: Fraction of data for training (temporal split).
        confidence_level: Bootstrap CI level.
        price_col: Column for forward return calculation.
        regime_breakdown: Compute per-regime statistics.
        min_regime_samples: Min events per regime to report.
        verbose: Print progress.

    Returns:
        EventStudyResult with train/test/baseline/regime results.
    """
    if horizons is None:
        horizons = [4, 16, 48, 96, 192, 288]

    df = _prepare_data(df, horizons, price_col)

    # Temporal train/test split
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    train_start = str(df_train["ts"].iloc[0])
    train_end = str(df_train["ts"].iloc[-1])
    test_start = str(df_test["ts"].iloc[0])
    test_end = str(df_test["ts"].iloc[-1])

    if verbose:
        print(
            f"Train: {len(df_train)} rows ({train_start} → {train_end})"
        )
        print(
            f"Test:  {len(df_test)} rows ({test_start} → {test_end})"
        )

    # Event masks
    train_mask = _compute_event_mask(df_train, event, condition)
    test_mask = _compute_event_mask(df_test, event, condition)

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    n_total = n_train + n_test

    if verbose:
        print(
            f"Events: {n_train} train, {n_test} test (total {n_total})"
        )

    config = {
        "event": event,
        "condition": condition or "",
        "side": side,
        "min_samples": min_samples,
        "n_baseline": n_baseline,
        "train_ratio": train_ratio,
        "confidence_level": confidence_level,
        "price_col": price_col,
        "horizons": horizons,
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
    }

    result = EventStudyResult(
        config=config,
        n_events_train=n_train,
        n_events_test=n_test,
        n_total=n_total,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if (n_train < min_samples and n_test < min_samples):
        if verbose:
            print(
                f"  INSUFFICIENT: {n_train}+{n_test} < {min_samples}"
            )
        result.train_results["insufficient"] = {
            "n_events": n_train,
            "reason": "insufficient_samples",
        }
        result.test_results["insufficient"] = {
            "n_events": n_test,
            "reason": "insufficient_samples",
        }
        return result

    # Compute baseline on train data
    baseline_returns = _run_baseline(
        df_train,
        horizons,
        n_samples=n_baseline,
        condition=condition,
        regime_aware=True,
        price_col=price_col,
    )

    # Train results
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in df_train.columns:
            continue
        event_returns = df_train.loc[train_mask, col]
        baseline_vals = baseline_returns.get(h, None)

        stats = _compute_horizon_stats(
            event_returns,
            baseline_vals,
            confidence_level=confidence_level,
        )
        result.train_results[str(h)] = stats

    # Test results
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in df_test.columns:
            continue
        event_returns = df_test.loc[test_mask, col]
        baseline_vals = baseline_returns.get(h, None)

        stats = _compute_horizon_stats(
            event_returns,
            baseline_vals,
            confidence_level=confidence_level,
        )
        result.test_results[str(h)] = stats

    # Baseline results
    for h in horizons:
        vals = baseline_returns.get(h, np.array([]))
        if len(vals) > 0:
            result.baseline_results[str(h)] = {
                "n_samples": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "ci_lower": float(
                    np.percentile(vals, (1 - confidence_level) / 2 * 100)
                ),
                "ci_upper": float(
                    np.percentile(vals, (1 + confidence_level) / 2 * 100)
                ),
            }

    # Significant horizons
    sig_horizons = []
    for h_str, test_r in result.test_results.items():
        if test_r.get("n_events", 0) >= min_samples:
            p = test_r.get("p_value", 1.0)
            hit = test_r.get("hit_ratio", 0.5)
            if p < 0.05:
                sig_horizons.append(
                    f"{h_str}bars(p={p:.3f},hit={hit:.2f})"
                )
    result.significant_horizons = sig_horizons

    # Regime breakdown
    if regime_breakdown and "regime" in df.columns:
        regime_names = {
            0: "ranging",
            1: "trending_up",
            2: "trending_down",
            3: "volatile",
        }
        for regime_val, regime_name in regime_names.items():
            try:
                if condition and condition.strip():
                    rg_condition = (
                        f"({condition}) & (regime == {regime_val})"
                    )
                else:
                    rg_condition = f"regime == {regime_val}"

                rg_train_mask = _compute_event_mask(
                    df_train, event, rg_condition
                )
                rg_test_mask = _compute_event_mask(
                    df_test, event, rg_condition
                )

                if (
                    rg_train_mask.sum() < min_regime_samples
                    and rg_test_mask.sum() < min_regime_samples
                ):
                    continue

                rg_results = {}
                for h in horizons:
                    col = f"fwd_ret_{h}"
                    if col not in df_test.columns:
                        continue
                    rg_event_returns = df_test.loc[rg_test_mask, col]
                    baseline_vals = baseline_returns.get(h, None)

                    rg_stats = _compute_horizon_stats(
                        rg_event_returns,
                        baseline_vals,
                        confidence_level=confidence_level,
                    )
                    rg_results[str(h)] = rg_stats

                result.regime_breakdown[regime_name] = {
                    "n_train": int(rg_train_mask.sum()),
                    "n_test": int(rg_test_mask.sum()),
                    "results": rg_results,
                }
            except Exception as e:
                if verbose:
                    print(
                        f"  Regime {regime_name} breakdown failed: {e}"
                    )

    return result


def print_result(result: EventStudyResult, detailed: bool = False) -> None:
    """Pretty-print an event study result."""
    cfg = result.config
    print(f"\n{'='*65}")
    print(f"  EVENT: {cfg['event']}")
    if cfg.get("condition"):
        print(f"  COND:  {cfg['condition']}")
    print(f"  SIDE:  {cfg['side']}")
    print(f"{'='*65}")

    print(f"\n  Train: {result.train_start[:10]} → {result.train_end[:10]}"
          f"  ({result.n_events_train} events)")
    print(f"  Test:  {result.test_start[:10]} → {result.test_end[:10]}"
          f"  ({result.n_events_test} events)")

    if result.n_events_train < cfg.get("min_samples", 30) and \
       result.n_events_test < cfg.get("min_samples", 30):
        print(f"\n  ⚠ INSUFFICIENT: {result.n_total} total events"
              f" < {cfg.get('min_samples', 30)}")
        return

    print(f"\n  {'Horizon':>8s}  {'n_train':>6s}  {'n_test':>6s}  "
          f"{'Mean_ret':>10s}  {'Hit':>6s}  {'Sharpe':>8s}  "
          f"{'p_val':>6s}  {'Signif':>6s}")
    print(f"  {'-'*62}")

    for h in sorted(result.test_results.keys(), key=int):
        tr = result.train_results.get(h, {})
        tr_n = tr.get("n_events", 0)
        te = result.test_results.get(h, {})
        te_n = te.get("n_events", 0)
        mean_r = te.get("mean_return", 0)
        hit = te.get("hit_ratio", 0)
        sharpe = te.get("sharpe_annualized", 0)
        p_val = te.get("p_value", 1)

        if te_n < cfg.get("min_samples", 30):
            sig = "insuf"
        elif p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.1:
            sig = "*"
        else:
            sig = ""

        print(
            f"  {h:>4s}b  "
            f"{tr_n:>6d}  {te_n:>6d}  "
            f"{mean_r*100:>+9.4f}%  "
            f"{hit:>5.2f}  "
            f"{sharpe:>+8.2f}  "
            f"{p_val:>6.3f}  "
            f"{sig:>6s}"
        )

    if result.significant_horizons:
        print(f"\n  ✅ SIGNIFICANT horizons: "
              f"{', '.join(result.significant_horizons)}")
    else:
        print(f"\n  ❌ No significant horizons (p < 0.05)")

    # Best horizon
    best_h = None
    best_abs_sharpe = 0
    for h_str, te in result.test_results.items():
        if te.get("n_events", 0) >= cfg.get("min_samples", 30):
            s = abs(te.get("sharpe_annualized", 0))
            if s > best_abs_sharpe:
                best_abs_sharpe = s
                best_h = h_str
    if best_h:
        te = result.test_results[best_h]
        print(f"\n  Best horizon: {best_h}b"
              f"  mean={te['mean_return']*100:+.4f}%"
              f"  hit={te['hit_ratio']:.2f}"
              f"  sharpe={te['sharpe_annualized']:+.2f}"
              f"  p={te['p_value']:.3f}")

    # Regime breakdown
    if result.regime_breakdown:
        print(f"\n  {'Regime':>15s}  {'n_test':>6s}  "
              f"{'Mean_ret':>10s}  {'Hit':>6s}  {'Sharpe':>8s}")
        for regime_name, rg in sorted(result.regime_breakdown.items()):
            rg_n = rg["n_test"]
            best_rg = None
            best_rg_sharpe = 0
            for h_str, rg_r in rg["results"].items():
                s = abs(rg_r.get("sharpe_annualized", 0))
                if s > best_rg_sharpe:
                    best_rg_sharpe = s
                    best_rg = (h_str, rg_r)
            if best_rg:
                h_str, rg_r = best_rg
                print(
                    f"  {regime_name:>15s}  {rg_n:>6d}  "
                    f"{rg_r['mean_return']*100:>+9.4f}%  "
                    f"{rg_r['hit_ratio']:>5.2f}  "
                    f"{rg_r['sharpe_annualized']:>+8.2f}"
                )

    if detailed:
        print(f"\n  Full test results:")
        for h_str, te in sorted(result.test_results.items(), key=lambda x: int(x[0])):
            print(f"    {h_str}b: {te}")

    print()


def save_result(result: EventStudyResult, path: str | Path) -> None:
    """Save event study result to JSON."""
    data = asdict(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Event study saved to %s", path)


def print_summary_table(
    results: list[EventStudyResult],
    min_samples: int = 30,
) -> None:
    """Print a summary table comparing multiple event studies."""
    print(f"\n{'='*80}")
    print(f"  EVENT STUDY SUMMARY TABLE")
    print(f"{'='*80}")
    print(
        f"  {'Event':>40s}  {'n_train':>7s}  {'n_test':>7s}  "
        f"{'Best_h':>6s}  {'Mean_ret':>10s}  {'Hit':>5s}  "
        f"{'Sharpe':>7s}  {'p_val':>6s}"
    )
    print(f"  {'-'*72}")

    sig_count = 0
    for r in results:
        cfg = r.config
        event_str = cfg["event"]
        condition_str = cfg.get("condition", "")
        label = event_str
        if condition_str:
            label += f"|{condition_str}"
        label = label[:40]

        best_h = None
        best_abs_sharpe = 0
        for h_str, te in r.test_results.items():
            if te.get("n_events", 0) >= min_samples:
                s = abs(te.get("sharpe_annualized", 0))
                if s > best_abs_sharpe:
                    best_abs_sharpe = s
                    best_h = h_str

        if best_h:
            te = r.test_results[best_h]
            sig_str = "✅" if te.get("p_value", 1) < 0.05 else ""
            if te.get("p_value", 1) < 0.05:
                sig_count += 1
            print(
                f"  {label:>40s}  {r.n_events_train:>7d}  "
                f"{r.n_events_test:>7d}  {best_h:>4s}b  "
                f"{te['mean_return']*100:>+9.4f}%  "
                f"{te['hit_ratio']:>5.2f}  "
                f"{te['sharpe_annualized']:>+7.2f}  "
                f"{te['p_value']:>6.3f}  {sig_str:>2s}"
            )
        else:
            print(
                f"  {label:>40s}  {r.n_events_train:>7d}  "
                f"{r.n_events_test:>7d}  {'N/A':>6s}  "
                f"{'N/A':>10s}  {'N/A':>5s}  {'N/A':>7s}  "
                f"{'N/A':>6s}"
            )

    print(f"\n  Significant: {sig_count}/{len(results)}")


def run_baseline_study(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    n_samples: int = 1000,
    condition: str | None = None,
    price_col: str = "close",
    train_ratio: float = 0.7,
) -> BaselineResult:
    """Run standalone baseline study (no event)."""
    if horizons is None:
        horizons = [4, 16, 48, 96, 192, 288]
    df = _prepare_data(df, horizons, price_col)
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)

    baseline = _run_baseline(
        df_train,
        horizons,
        n_samples=n_samples,
        condition=condition,
        regime_aware=True,
        price_col=price_col,
    )

    result = BaselineResult(n_samples=n_samples)
    for h in horizons:
        vals = baseline.get(h, np.array([]))
        if len(vals) > 0:
            result.horizon_results[h] = HorizonResult(
                n_events=len(vals),
                mean_return=float(np.mean(vals)),
                median_return=float(np.median(vals)),
                std_return=float(np.std(vals)),
                hit_ratio=float(np.mean(vals > 0)),
                sharpe_annualized=(
                    float(np.mean(vals) / np.std(vals) * np.sqrt(96 * 365))
                    if np.std(vals) > 1e-10 else 0.0
                ),
                p_value=1.0,
                ci_lower=float(np.percentile(vals, 2.5)),
                ci_upper=float(np.percentile(vals, 97.5)),
                skew=0.0,
                kurtosis=0.0,
                min_return=float(np.min(vals)),
                max_return=float(np.max(vals)),
                t_stat=0.0,
            )
    return result


def scan_feature_thresholds(
    df: pd.DataFrame,
    feature: str,
    thresholds: list[float],
    horizons: list[int] | None = None,
    condition: str | None = None,
    direction: str = "gt",
    side: str = "both",
    min_samples: int = 30,
    regimes: list[int] | None = None,
    verbose: bool = False,
) -> list[ScanResult]:
    """Scan a feature across thresholds to find optimal cut points.

    WARNING: Overfitting risk is high with scans.
    Use only after manual event study validation shows edge.
    """
    if horizons is None:
        horizons = [48]  # Default: 12h
    if regimes is None:
        regimes = [-1]  # -1 means all

    results = []

    for thresh in thresholds:
        if direction == "gt":
            event_str = f"{feature} > {thresh}"
        else:
            event_str = f"{feature} < {thresh}"

        for regime in regimes:
            if regime >= 0:
                cond = f"regime == {regime}"
                if condition:
                    cond = f"({condition}) & ({cond})"
            else:
                cond = condition

            try:
                r = run_event_study(
                    df,
                    event=event_str,
                    horizons=horizons,
                    condition=cond,
                    side=side,
                    min_samples=min_samples,
                    verbose=False,
                )

                for h in horizons:
                    h_str = str(h)
                    te = r.test_results.get(h_str, {})
                    if te.get("n_events", 0) < min_samples:
                        continue

                    results.append(ScanResult(
                        feature=feature,
                        threshold=thresh,
                        direction=direction,
                        n_train=r.n_events_train,
                        n_test=te.get("n_events", 0),
                        test_mean_return=te.get("mean_return", 0),
                        test_hit_ratio=te.get("hit_ratio", 0),
                        test_sharpe=te.get("sharpe_annualized", 0),
                        test_p_value=te.get("p_value", 1),
                        significant=te.get("p_value", 1) < 0.05,
                        regime=(
                            ["all", "ranging", "trending_up",
                             "trending_down", "volatile"][regime + 1]
                            if regime >= 0 else "all"
                        ),
                        horizon=h,
                    ))
            except Exception as e:
                if verbose:
                    print(f"  Failed {event_str}|{cond}: {e}")

    return results