"""PuppyGraph Element Cache config."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TimeUnit(Enum):
    """Enum for time units in cache TTL."""

    YEAR = 0
    MONTH = 1
    DAY = 2
    HOUR = 3
    MINUTE = 4
    PARTITION = 5


@dataclass(frozen=True)
class TimeToLive:
    """Time-to-live configuration for cache."""

    amount: int
    unit: TimeUnit


@dataclass(frozen=True)
class RefreshStrategy:
    """Refresh strategy configuration for cache."""

    parallelism: int


@dataclass(frozen=True)
class GraphElementCacheConfig:
    """Cache configuration for elements."""

    partition_key: str
    partition_ttl: Optional[TimeToLive] = None
    refresh_strategy: Optional[RefreshStrategy] = None
