"""Strategy survival tracker — logs pipeline stage pass rates.

Each round logs how many strategies survive each stage:
  generated → fast filter → walk-forward → OOS → not overfit

Useful for detecting LLM degradation and filter effectiveness.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SURVIVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "ts" / "_survival.jsonl"


def log_survival(
    round_num: int,
    generated: int,
    fast_pass: int,
    wf_pass: int,
    oos_pass: int,
    overfit_count: int,
) -> None:
    """Append one survival record for a round."""
    _SURVIVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "round": round_num,
        "generated": generated,
        "fast_filter_pass": fast_pass,
        "wf_pass": wf_pass,
        "oos_pass": oos_pass,
        "overfit": overfit_count,
        "fast_rate": round(fast_pass / generated, 3) if generated else 0,
        "wf_rate": round(wf_pass / generated, 3) if generated else 0,
        "oos_rate": round(oos_pass / generated, 3) if generated else 0,
    }
    try:
        with open(_SURVIVAL_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(
            "survival: round=%d  gen=%d  fast=%d (%.0f%%)  wf=%d (%.0f%%)  oos=%d (%.0f%%)  overfit=%d",
            round_num, generated,
            fast_pass, 100 * fast_pass / generated,
            wf_pass, 100 * wf_pass / generated,
            oos_pass, 100 * oos_pass / generated,
            overfit_count,
        )
    except Exception as e:
        logger.warning("survival: failed to log: %s", e)


def load_survival() -> list[dict]:
    """Load all survival records."""
    if not _SURVIVAL_PATH.exists():
        return []
    records = []
    with open(_SURVIVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
