"""Concept statistics — tracks reward and survival metrics by family + regime.

Provides:
  - expected_sharpe(regime): baseline Sharpe for a given regime
  - reward_simple(metrics): unifies Sharpe, robustness, novelty into one scalar
  - record_outcome(): stores results per concept
  - get_concept_stats(): aggregated view
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CONCEPT_PATH = Path(__file__).resolve().parents[2] / "data" / "ts" / "_concept_stats.jsonl"

# Default expected Sharpe per regime (updated from data over time)
_default_expected_sharpe = {0: 0.3, 1: 0.6, 2: 0.4, 3: 0.5}

# In-memory cache
_expected_sharpe = dict(_default_expected_sharpe)


def expected_sharpe(regime: int) -> float:
    """Return expected Sharpe for a given regime.

    Used to normalize regime-dependent Sharpe inflation/deflation.
    """
    return _expected_sharpe.get(regime, 0.3)


def reward_simple(
    sharpe_oos: float,
    sharpe_train: float,
    wf_sharpe: float,
    wf_cv: float,
    max_dd: float,
    total_pnl: float,
    n_trades: int,
    regime: int = -1,
    novelty_score: float = 1.0,
    overfit: bool = False,
) -> float:
    """Compute unified reward scalar.

    reward = P × R × N
    where:
      P = performance (regime-normalized Sharpe)
      R = robustness (CV + drawdown + decay penalties)
      N = novelty penalization

    Fase 1: version simple, multiplicativa directa.
    """
    if overfit or n_trades < 10:
        return 0.0

    # --- P: Performance (regime-normalized) ---
    regime_norm = expected_sharpe(regime) if regime >= 0 else 0.5
    sharpe_norm = sharpe_oos / regime_norm if regime_norm > 0 else sharpe_oos
    P = (
        max(0, min(sharpe_norm, 5)) * 0.6
        + max(0, min(sharpe_train, 5)) * 0.3
        + max(0, min(wf_sharpe, 5)) * 0.1
    )

    # --- R: Robustness ---
    # CV penalty
    cv_penalty = 1.0 if wf_cv <= 1.0 else math.exp(-0.5 * (wf_cv - 1.0))

    # Drawdown penalty
    dd_ratio = abs(max_dd) / (abs(total_pnl) + 1)
    dd_penalty = 1.0 if dd_ratio <= 0.10 else math.exp(-3.0 * (dd_ratio - 0.10))

    # Decay penalty (OOS vs train)
    if sharpe_train > 0:
        decay = sharpe_oos / sharpe_train
        decay_penalty = 1.0 if decay >= 0.5 else math.exp(-3.0 * (0.5 - decay))
    else:
        decay_penalty = 1.0

    R = cv_penalty * dd_penalty * decay_penalty

    # --- N: Novelty ---
    N = max(0.1, novelty_score)

    # --- Reward ---
    reward = P * R * N
    return round(reward, 4)


def record_outcome(
    archetype: str,
    regime: int,
    reward: float,
    passed_fast: bool,
    passed_oos: bool,
    sharpe_oos: float,
    sharpe_train: float,
    n_trades: int,
) -> None:
    """Store one outcome record."""
    _CONCEPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "archetype": archetype,
        "regime": regime,
        "reward": reward,
        "passed_fast": passed_fast,
        "passed_oos": passed_oos,
        "sharpe_oos": round(sharpe_oos, 4),
        "sharpe_train": round(sharpe_train, 4),
        "n_trades": n_trades,
    }
    try:
        with open(_CONCEPT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("concept_stats: failed to record: %s", e)

    # Update expected Sharpe per regime (online update)
    if regime >= 0 and n_trades >= 10:
        current = _expected_sharpe.get(regime, 0.3)
        _expected_sharpe[regime] = 0.95 * current + 0.05 * sharpe_oos


def get_concept_stats() -> list[dict]:
    """Load all concept records."""
    if not _CONCEPT_PATH.exists():
        return []
    records = []
    with open(_CONCEPT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_avg_reward(archetype: str | None = None, regime: int | None = None) -> float:
    """Average reward for a given archetype + regime combo."""
    records = get_concept_stats()
    if not records:
        return 0.0
    filtered = records
    if archetype:
        filtered = [r for r in filtered if r.get("archetype") == archetype]
    if regime is not None:
        filtered = [r for r in filtered if r.get("regime") == regime]
    if not filtered:
        return 0.0
    rewards = [r.get("reward", 0) for r in filtered if r.get("reward", 0) > 0]
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)
