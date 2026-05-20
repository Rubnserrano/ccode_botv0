"""Run event studies experiments.

Usage:
    python scripts/run_event_studies.py --experiments A1 A2 --output results/event_studies/phase_a/
    python scripts/run_event_studies.py --all --phase a
    python scripts/run_event_studies.py --scan funding_rate_zscore --min -2 --max 2 --step 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.event_studies import (
    run_event_study,
    print_result,
    save_result,
    print_summary_table,
    scan_feature_thresholds,
)

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load complete dataset with indicators + derivatives for event studies."""
    from src.ts_store import read as ts_read
    from src.indicators.calculator import calc_all

    df = ts_read("market:binance:btcusdt", frequency="15m", limit=50000)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    df = df[df["ts"] >= cutoff].reset_index(drop=True)

    # Merge derivatives
    deriv_dir = Path("data/ts/derivatives/aligned/15m")
    if deriv_dir.exists():
        for f in sorted(deriv_dir.glob("*.parquet")):
            try:
                deriv_df = pd.read_parquet(f)
                deriv_df["ts"] = pd.to_datetime(deriv_df["ts"], utc=True)
                df = df.merge(deriv_df, on="ts", how="left")
            except Exception as e:
                logger.warning("Failed to merge %s: %s", f.name, e)

    df = calc_all(df)
    return df


PHASE_A_EXPERIMENTS = {
    "A1": {
        "event": "funding_rate_zscore > 2.0",
        "condition": None,
        "side": "long",
        "description": "Funding Z > 2 (all regimes) — long hypothesis",
    },
    "A2": {
        "event": "funding_rate_zscore > 2.0",
        "condition": "regime == 2",
        "side": "short",
        "description": "Funding Z > 2 & trending_down — short hypothesis",
    },
    "A3": {
        "event": "funding_rate_zscore < -2.0",
        "condition": None,
        "side": "long",
        "description": "Funding Z < -2 (all regimes) — long hypothesis",
    },
    "A4": {
        "event": "funding_rate_zscore < -2.0",
        "condition": "regime == 1",
        "side": "long",
        "description": "Funding Z < -2 & trending_up — long in uptrend",
    },
    "A5": {
        "event": "funding_rate_zscore > 1.5",
        "condition": "regime == 2",
        "side": "short",
        "description": "Funding Z > 1.5 & trending_down — short, less extreme",
    },
    "A6": {
        "event": "funding_rate_zscore > 1.5",
        "condition": "regime == 0",
        "side": "long",
        "description": "Funding Z > 1.5 & ranging — neutral regime",
    },
    "A7": {
        "event": "funding_rate_zscore < -1.5",
        "condition": None,
        "side": "long",
        "description": "Funding Z < -1.5 — less extreme short side",
    },
    "A8": {
        "event": "funding_rate_zscore < -1.0",
        "condition": "regime == 2",
        "side": "long",
        "description": "Funding Z < -1 & trending_down — contrarian long",
    },
    "A9": {
        "event": "funding_rate_zscore > 1.0",
        "condition": "regime == 1",
        "side": "short",
        "description": "Funding Z > 1 & trending_up — pullback in uptrend",
    },
}

PHASE_B_EXPERIMENTS = {
    "B1": {
        "event": "(fear_greed < 20) & (funding_rate_zscore > 1.5)",
        "condition": None,
        "side": "short",
        "description": "Extreme fear + high funding — short (prelim: -0.38% to -0.70%)",
    },
    "B2": {
        "event": "(fear_greed < 25) & (funding_rate_zscore > 2.0)",
        "condition": None,
        "side": "long",
        "description": "Fear + high funding — potential short squeeze (prelim hit 55%)",
    },
    "B3": {
        "event": "fear_greed < 15",
        "condition": None,
        "side": "long",
        "description": "Pure extreme fear — capitulation buy",
    },
    "B4": {
        "event": "fear_greed < 20",
        "condition": "regime == 2",
        "side": "short",
        "description": "Extreme fear in downtrend — continuation down",
    },
    "B5": {
        "event": "fear_greed > 75",
        "condition": "regime == 1",
        "side": "short",
        "description": "Extreme greed in uptrend — potential top",
    },
    "B6": {
        "event": "(fear_greed < 25) & (funding_rate_zscore < -1.5)",
        "condition": None,
        "side": "long",
        "description": "Fear + low funding — double bearish, potential reversal",
    },
    "B7": {
        "event": "(fear_greed < 15) & (funding_rate_zscore > 1.5)",
        "condition": None,
        "side": "short",
        "description": "Extreme fear + high funding (tighter) — capitulation short",
    },
}

DEFAULT_HORIZONS = [4, 16, 48, 96, 192, 288]  # 1h, 4h, 12h, 24h, 48h, 72h


def run_single_experiment(experiment_id: str, config: dict, df: pd.DataFrame,
                          horizons: list[int], output_dir: Path,
                          min_samples: int = 30, verbose: bool = True) -> None:
    """Run a single experiment and save results."""
    print(f"\n{'#'*65}")
    print(f"  Experiment {experiment_id}: {config.get('description', '')}")
    print(f"  Event: {config['event']}")
    if config.get("condition"):
        print(f"  Cond:  {config['condition']}")
    print(f"  Side:  {config['side']}")
    print(f"{'#'*65}")

    result = run_event_study(
        df,
        event=config["event"],
        horizons=horizons,
        condition=config.get("condition"),
        side=config.get("side", "both"),
        min_samples=min_samples,
        verbose=verbose,
    )

    print_result(result)

    save_result(result, output_dir / f"{experiment_id}.json")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run event studies experiments"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        help="Experiment IDs to run (e.g. A1 A2 A3)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments in specified phase"
    )
    parser.add_argument(
        "--phase", choices=["a", "b", "ab"], default="a",
        help="Phase: a=funding-only, b=funding+fear_greed"
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+",
        default=DEFAULT_HORIZONS,
        help=f"Horizons in bars (default: {DEFAULT_HORIZONS})"
    )
    parser.add_argument(
        "--min-samples", type=int, default=30,
        help="Minimum event samples required"
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output directory (e.g. results/event_studies/phase_a/)"
    )
    parser.add_argument(
        "--scan", type=str, default=None,
        help="Scan feature thresholds (feature name)"
    )
    parser.add_argument(
        "--scan-min", type=float, default=-3.0,
        help="Min threshold for scan"
    )
    parser.add_argument(
        "--scan-max", type=float, default=3.0,
        help="Max threshold for scan"
    )
    parser.add_argument(
        "--scan-step", type=float, default=0.5,
        help="Threshold step for scan"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Determine experiments
    if args.phase == "a":
        experiments = PHASE_A_EXPERIMENTS
    elif args.phase == "b":
        experiments = PHASE_B_EXPERIMENTS
    else:
        experiments = {**PHASE_A_EXPERIMENTS, **PHASE_B_EXPERIMENTS}

    if args.experiments:
        selected = {
            k: v for k, v in experiments.items() if k in args.experiments
        }
        if not selected:
            print(f"Unknown experiments: {args.experiments}")
            print(f"Available: {list(experiments.keys())}")
            return
    elif args.all:
        selected = experiments
    else:
        selected = experiments

    # Output dir
    if args.output:
        output_dir = args.output
    else:
        phase_label = {"a": "phase_a", "b": "phase_b", "ab": "phase_ab"}
        output_dir = (
            Path("results") / "event_studies" / phase_label[args.phase]
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan mode
    if args.scan:
        df = load_data()
        thresholds = [
            args.scan_min + i * args.scan_step
            for i in range(
                int((args.scan_max - args.scan_min) / args.scan_step) + 1
            )
        ]
        results = scan_feature_thresholds(
            df,
            feature=args.scan,
            thresholds=thresholds,
            horizons=[48],
            min_samples=args.min_samples,
            verbose=args.verbose,
        )
        print(f"\nScan results for {args.scan}:")
        print(f"{'Threshold':>10s}  {'n_test':>6s}  {'Mean_ret':>10s}  "
              f"{'Hit':>5s}  {'Sharpe':>7s}  {'p_val':>6s}  {'Sig':>4s}")
        print("-" * 55)
        for r in results:
            print(
                f"{r.threshold:>10.2f}  {r.n_test:>6d}  "
                f"{r.test_mean_return*100:>+9.4f}%  "
                f"{r.test_hit_ratio:>5.2f}  "
                f"{r.test_sharpe:>+7.2f}  "
                f"{r.test_p_value:>6.3f}  "
                f"{'✅' if r.significant else ''}"
            )
        return

    # Load data
    df = load_data()
    print(f"Data: {len(df)} rows, {len(df.columns)} cols")

    # Run experiments
    results = []
    for exp_id, config in sorted(selected.items()):
        r = run_single_experiment(
            exp_id, config, df, args.horizons, output_dir,
            min_samples=args.min_samples, verbose=args.verbose,
        )
        results.append(r)

    # Summary
    print(f"\n{'='*65}")
    print(f"  SUMMARY — {len(results)} experiments")
    print(f"{'='*65}")
    print_summary_table(results, min_samples=args.min_samples)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()