"""Per-tenant spend budgets.

A tenant has a budget in USD per period (``daily``, ``monthly`` or ``total``;
0 = unlimited). The tracker keeps the current period's spend in memory, adds
every request's cost as it happens, and is re-seeded from the usage table at
startup and every ``refresh_s`` seconds, so several replicas converge on the
database's view of spend.

A request is admitted while ``spent < budget``; the call that crosses the line
is allowed to finish (its cost is unknown until it returns), and the next one
is rejected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

PERIODS = ("daily", "monthly", "total")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def period_start(now: datetime, period: str) -> datetime:
    """Start (UTC) of the budget period that contains ``now``."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "monthly":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "total":
        return _EPOCH
    raise ValueError(f"unknown budget period {period!r}; expected one of {PERIODS}")


@dataclass(frozen=True)
class Budget:
    limit_usd: float = 0.0
    period: str = "monthly"

    def __post_init__(self) -> None:
        if self.limit_usd < 0:
            raise ValueError("budget must be >= 0")
        if self.period not in PERIODS:
            raise ValueError(f"unknown budget period {self.period!r}")

    @property
    def unlimited(self) -> bool:
        return self.limit_usd == 0


@dataclass
class BudgetStatus:
    allowed: bool
    spent_usd: float
    limit_usd: float
    remaining_usd: float


class BudgetTracker:
    """In-memory spend per (tenant, period start)."""

    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._clock = clock
        self._spent: Dict[Tuple[str, datetime], float] = {}
        self._lock = threading.Lock()

    def _key(self, tenant: str, budget: Budget) -> Tuple[str, datetime]:
        return tenant, period_start(self._clock(), budget.period)

    def spent(self, tenant: str, budget: Budget) -> float:
        with self._lock:
            return self._spent.get(self._key(tenant, budget), 0.0)

    def check(self, tenant: str, budget: Budget) -> BudgetStatus:
        spent = self.spent(tenant, budget)
        if budget.unlimited:
            return BudgetStatus(True, spent, 0.0, float("inf"))
        remaining = budget.limit_usd - spent
        return BudgetStatus(spent < budget.limit_usd, spent, budget.limit_usd, max(0.0, remaining))

    def add(self, tenant: str, budget: Budget, cost: float) -> float:
        """Record ``cost`` against the current period. Returns the new period spend."""
        if cost < 0:
            raise ValueError("cost must be >= 0")
        key = self._key(tenant, budget)
        with self._lock:
            self._spent[key] = self._spent.get(key, 0.0) + cost
            return self._spent[key]

    def seed(self, tenant: str, budget: Budget, spent_from_db: float, pending: float = 0.0) -> None:
        """Replace the current period's spend with the database total plus not-yet-flushed cost."""
        if spent_from_db < 0 or pending < 0:
            raise ValueError("spend must be >= 0")
        with self._lock:
            self._spent[self._key(tenant, budget)] = spent_from_db + pending

    def since(self, budget: Budget) -> datetime:
        return period_start(self._clock(), budget.period)

    def prune(self, keep_after: Optional[datetime] = None) -> int:
        """Drop counters for periods that started before ``keep_after`` (default: this month)."""
        cutoff = keep_after or period_start(self._clock(), "monthly")
        with self._lock:
            old = [k for k in self._spent if _EPOCH < k[1] < cutoff]
            for k in old:
                del self._spent[k]
        return len(old)
