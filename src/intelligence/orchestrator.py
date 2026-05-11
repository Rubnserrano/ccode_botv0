"""Orchestrator — manages fetch schedules for external data sources.

Only sources that are actively being used (referenced by a strategy)
or explicitly requested are fetched.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.intelligence.registry import load_source, save_source
from src.intelligence.fetcher import fetch_source
from src.intelligence.aligner import align_source

logger = logging.getLogger(__name__)


async def run_source_schedule(source_name: str, interval_seconds: int = 3600) -> None:
    """Fetch a single source on a schedule.

    Runs forever in a background task.
    Each iteration: fetch → align → update metadata.
    """
    while True:
        try:
            ds = load_source(source_name)
            if ds is None or not ds.enabled:
                logger.info("schedule: '%s' disabled or removed, stopping", source_name)
                break

            logger.info("schedule: fetching '%s'...", source_name)
            df = fetch_source(ds)
            if not df.empty:
                for tf in ds.align.target_timeframes:
                    align_source(df, ds, tf)
                ds.total_rows = len(df)
                ds.last_fetch_at = datetime.now(timezone.utc).isoformat()
                save_source(ds)
                logger.info("schedule: '%s' done (%d rows)", source_name, len(df))
            else:
                logger.warning("schedule: '%s' returned no data", source_name)

        except Exception as e:
            logger.error("schedule: '%s' error: %s", source_name, e)

        await asyncio.sleep(interval_seconds)


async def run_all_schedules() -> None:
    """Run schedules for all enabled sources (background task)."""
    from src.intelligence.registry import list_sources
    sources = list_sources()
    tasks = []
    for src in sources:
        if src.get("enabled"):
            # Parse schedule string to seconds
            sched = src.get("schedule", "1h")
            seconds = _parse_schedule(sched)
            task = asyncio.create_task(run_source_schedule(src["name"], seconds))
            tasks.append(task)
            logger.info("schedule: registered '%s' (every %s)", src["name"], sched)
    if tasks:
        await asyncio.gather(*tasks)


def _parse_schedule(schedule: str) -> int:
    """Convert '1h', '30m', '1d' to seconds."""
    unit = schedule[-1]
    value = int(schedule[:-1])
    if unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    return 3600
