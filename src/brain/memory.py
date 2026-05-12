"""Strategy Memory — deduplication + novelty detection.

Each strategy produces a **fingerprint** based on its indicators and ops.
Before a strategy is backtested, we check if a similar one already exists.
This prevents the LLM from testing the same idea 30 times.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

_MEMORY_PATH = Path(__file__).resolve().parents[2] / "data" / "parquet" / "research" / "fingerprints.jsonl"


def fingerprint(strategy_dict: dict) -> str:
    """Create a unique hash from the indicators and operators used.

    Two strategies that use the same indicators with the same operators
    (even with different numeric values) get the SAME fingerprint.
    This catches structural duplicates like "estrat_1" vs "estrat_1_v2".
    """
    conditions = strategy_dict.get("entry_conditions", [])
    sig = sorted([
        (c.get("indicator", "?"), c.get("op", "?"))
        for c in conditions
    ])
    raw = json.dumps(sig, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def is_redundant(fp: str, seen: set[str] | None = None) -> bool:
    """Check if a fingerprint has been seen before.

    If ``seen`` is provided (in-memory cache), checks there first.
    Otherwise checks the persistent JSONL log.
    """
    if seen is not None and fp in seen:
        return True

    # Check persistent store
    if not _MEMORY_PATH.exists():
        return False
    with open(_MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line).get("fp") == fp:
                return True
    return False


def record(fp: str, strategy_dict: dict) -> None:
    """Record a fingerprint + strategy name in the persistent store."""
    _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({"fp": fp, "name": strategy_dict.get("name", "?"), "ts": str(pd.Timestamp.now())})
    with open(_MEMORY_PATH, "a") as f:
        f.write(entry + "\n")


def get_seen_set() -> set[str]:
    """Load all known fingerprints into memory (for fast checking)."""
    seen = set()
    if not _MEMORY_PATH.exists():
        return seen
    with open(_MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["fp"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen
