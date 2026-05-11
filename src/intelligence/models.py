"""Models for external data sources — validation and serialization.

Each data source is defined as a JSON document describing:
  - Where to fetch data from (URL, schedule, auth)
  - How to parse it (JSON path, CSV columns)
  - How to align it to OHLCV timeframes (ffill, interpolate)
  - What columns to produce

Registered sources are stored in data/sources/{name}.json
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


VALID_METHODS = {"ffill", "interpolate", "sum", "avg"}
VALID_TYPES = {"json", "csv"}
VALID_TIMEFRAMES = {"15m", "1h", "4h", "1d"}
MIN_SCHEDULE_SECONDS = 300  # 5 min


@dataclass
class ParseConfig:
    type: str = "json"                # "json" or "csv"
    timestamp_field: str = "ts"       # JSON path to timestamp
    value_field: str = "value"        # JSON path to value
    value_transform: str = "float"    # "float", "int", "str"


@dataclass
class AlignConfig:
    method: str = "ffill"             # "ffill", "interpolate", "sum", "avg"
    decay_periods: int = 0            # 0 = no decay, >0 = linear decay over N periods
    target_timeframes: list[str] = field(default_factory=lambda: ["15m"])


@dataclass
class RateLimit:
    max_calls_per_hour: int = 60
    cooldown_seconds: int = 60


@dataclass
class DataSource:
    """Complete definition of an external data source.

    Serialized to/from JSON in data/sources/{name}.json
    """
    name: str
    url: str
    description: str = ""
    params: dict = field(default_factory=dict)          # URL query params
    headers: dict = field(default_factory=dict)         # HTTP headers
    schedule: str = "1h"                                # fetch interval
    parse: ParseConfig = field(default_factory=ParseConfig)
    columns: dict = field(default_factory=lambda: {"value": "value"})
    align: AlignConfig = field(default_factory=AlignConfig)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    storage_quota_mb: int = 100
    api_key: str = ""                                   # stored for POC, no encryption
    enabled: bool = True
    last_fetch_at: str = ""
    total_rows: int = 0

    def validate(self) -> list[str]:
        """Validate the data source definition. Returns list of errors."""
        errors = []
        if not self.name or not self.name.replace("_", "").isalnum():
            errors.append("name must be alphanumeric + underscores only")
        if not self.url.startswith("https://"):
            errors.append("url must use HTTPS")
        if self.parse.type not in VALID_TYPES:
            errors.append(f"parse.type must be one of {VALID_TYPES}")
        if self.align.method not in VALID_METHODS:
            errors.append(f"align.method must be one of {VALID_METHODS}")
        for tf in self.align.target_timeframes:
            if tf not in VALID_TIMEFRAMES:
                errors.append(f"Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")
        return errors

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "params": dict(self.params),
            "headers": dict(self.headers),
            "schedule": self.schedule,
            "parse": {"type": self.parse.type, "timestamp_field": self.parse.timestamp_field,
                       "value_field": self.parse.value_field, "value_transform": self.parse.value_transform},
            "columns": dict(self.columns),
            "align": {"method": self.align.method, "decay_periods": self.align.decay_periods,
                      "target_timeframes": list(self.align.target_timeframes)},
            "rate_limit": {"max_calls_per_hour": self.rate_limit.max_calls_per_hour,
                          "cooldown_seconds": self.rate_limit.cooldown_seconds},
            "storage_quota_mb": self.storage_quota_mb,
            "api_key": "****" if self.api_key else "",
            "enabled": self.enabled,
            "last_fetch_at": self.last_fetch_at,
            "total_rows": self.total_rows,
        }
