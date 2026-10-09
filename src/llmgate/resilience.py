"""Circuit breaker and retry policy for upstream calls."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"
STATE_VALUE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitBreaker:
    """Classic three-state breaker.

    * **closed**: calls flow; ``failure_threshold`` consecutive failures open it.
    * **open**: calls are refused until ``recovery_s`` has passed since it opened.
    * **half_open**: up to ``half_open_max`` trial calls are let through. A success
      closes the breaker; a failure re-opens it for another ``recovery_s``.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_s: float = 30.0,
        half_open_max: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_s < 0:
            raise ValueError("recovery_s must be >= 0")
        if half_open_max < 1:
            raise ValueError("half_open_max must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self.half_open_max = half_open_max
        self._clock = clock
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trials = 0
        self._lock = threading.Lock()

    def _current(self) -> str:
        if self._state == OPEN and self._clock() - self._opened_at >= self.recovery_s:
            self._state = HALF_OPEN
            self._trials = 0
        return self._state

    @property
    def state(self) -> str:
        with self._lock:
            return self._current()

    @property
    def failures(self) -> int:
        return self._failures

    def allow(self) -> bool:
        """May a call go through now? In half-open this reserves a trial slot."""
        with self._lock:
            st = self._current()
            if st == CLOSED:
                return True
            if st == HALF_OPEN and self._trials < self.half_open_max:
                self._trials += 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = CLOSED
            self._failures = 0
            self._trials = 0

    def record_failure(self) -> None:
        with self._lock:
            st = self._current()
            if st == HALF_OPEN:
                self._trip()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()
        self._trials = 0


RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def is_retryable_status(status: int) -> bool:
    """Upstream statuses worth retrying / failing over on (overload, timeouts, 5xx)."""
    return status in RETRYABLE_STATUS or status >= 500


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff: ``base_s * 2**attempt``, capped at ``max_s``, with full jitter."""

    max_retries: int = 2
    base_s: float = 0.2
    max_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_s < 0 or self.max_s < 0:
            raise ValueError("delays must be >= 0")

    def delay(self, attempt: int, rand: Optional[Callable[[], float]] = None) -> float:
        """Sleep before retry number ``attempt`` (0-based). ``rand`` returns a value in [0, 1)."""
        cap = min(self.max_s, self.base_s * (2**attempt))
        r = random.random() if rand is None else rand()
        return cap * r

    def delays(self, rand: Optional[Callable[[], float]] = None) -> List[float]:
        return [self.delay(i, rand) for i in range(self.max_retries)]


def parse_retry_after(value: Optional[str], max_s: float = 30.0) -> Optional[float]:
    """Seconds from a ``Retry-After`` header (delta-seconds form only), capped at ``max_s``."""
    if value is None:
        return None
    try:
        secs = float(value.strip())
    except ValueError:
        return None
    if secs < 0 or secs != secs:  # negative or NaN
        return None
    return min(secs, max_s)
