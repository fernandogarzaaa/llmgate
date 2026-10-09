"""Token-bucket rate limiting, in memory or in Redis.

Each tenant gets two buckets: requests per minute (RPM) and tokens per minute
(TPM). A bucket holds at most ``capacity`` tokens and refills continuously at
``capacity / 60`` tokens per second. A limit of 0 means "unlimited".

The TPM bucket is charged an estimate before the call (``try_acquire``) and
then corrected with the real count afterwards (``consume`` may drive it
negative, which delays the tenant's next request instead of rejecting a call
that already happened).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class Decision:
    allowed: bool
    retry_after_s: float = 0.0
    remaining: float = 0.0


class TokenBucket:
    """A single token bucket. Thread-safe. ``clock`` is injectable for tests."""

    def __init__(self, capacity: float, refill_per_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_per_s <= 0:
            raise ValueError("refill_per_s must be > 0")
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_s)
        self._updated = now

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def try_acquire(self, n: float = 1.0) -> Decision:
        """Take ``n`` tokens if available; otherwise say how long to wait.

        A request larger than the capacity can never succeed, so it is
        rejected with ``retry_after_s = inf``.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        with self._lock:
            self._refill()
            if n > self.capacity:
                return Decision(False, math.inf, self._tokens)
            if self._tokens >= n:
                self._tokens -= n
                return Decision(True, 0.0, self._tokens)
            return Decision(False, (n - self._tokens) / self.refill_per_s, self._tokens)

    def consume(self, n: float) -> None:
        """Unconditionally take ``n`` tokens (balance may go negative); ``n < 0`` refunds."""
        with self._lock:
            self._refill()
            self._tokens = min(self.capacity, self._tokens - n)


@dataclass(frozen=True)
class Limits:
    rpm: int = 0
    tpm: int = 0


class RateLimiter:
    """Per-key RPM + TPM buckets held in process memory."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: Dict[Tuple[str, str, int], TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str, kind: str, per_minute: int) -> TokenBucket:
        k = (key, kind, per_minute)
        with self._lock:
            b = self._buckets.get(k)
            if b is None:
                b = TokenBucket(per_minute, per_minute / 60.0, self._clock)
                self._buckets[k] = b
            return b

    async def check(self, key: str, limits: Limits, est_tokens: int) -> Tuple[Decision, str]:
        """Admit one request estimated at ``est_tokens``. Returns (decision, which-limit)."""
        if limits.rpm > 0:
            d = self._bucket(key, "rpm", limits.rpm).try_acquire(1)
            if not d.allowed:
                return d, "rpm"
        if limits.tpm > 0 and est_tokens > 0:
            d = self._bucket(key, "tpm", limits.tpm).try_acquire(min(est_tokens, limits.tpm))
            if not d.allowed:
                if limits.rpm > 0:  # give the request slot back: it was not served
                    self._bucket(key, "rpm", limits.rpm).consume(-1)
                return d, "tpm"
        return Decision(True), ""

    async def settle(self, key: str, limits: Limits, est_tokens: int, actual_tokens: int) -> None:
        """Correct the TPM bucket once the real token count is known."""
        if limits.tpm > 0:
            charged = min(max(est_tokens, 0), limits.tpm)
            self._bucket(key, "tpm", limits.tpm).consume(actual_tokens - charged)


# KEYS[1] = bucket key; ARGV = capacity, refill_per_s, now_s, n, force(0/1)
# Returns {allowed(0/1), tokens_after*1000, retry_after_ms}
_LUA_BUCKET = """
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local n = tonumber(ARGV[4])
local force = tonumber(ARGV[5])
local s = redis.call('HMGET', KEYS[1], 't', 'u')
local tokens = tonumber(s[1])
local updated = tonumber(s[2])
if tokens == nil then tokens = cap; updated = now end
local elapsed = math.max(0, now - updated)
tokens = math.min(cap, tokens + elapsed * rate)
local allowed = 0
local retry = 0
if force == 1 then
  tokens = math.min(cap, tokens - n)
  allowed = 1
elseif n > cap then
  retry = -1
elseif tokens >= n then
  tokens = tokens - n
  allowed = 1
else
  retry = math.ceil((n - tokens) / rate * 1000)
end
redis.call('HSET', KEYS[1], 't', tostring(tokens), 'u', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(cap / rate) + 60)
return {allowed, math.floor(tokens * 1000), retry}
"""


class RedisRateLimiter:
    """Same semantics as :class:`RateLimiter`, shared across replicas via a Redis Lua script."""

    def __init__(self, redis: Any, prefix: str = "llmgate:rl", clock: Callable[[], float] = time.time) -> None:
        self._r = redis
        self._prefix = prefix
        self._clock = clock
        self._script = redis.register_script(_LUA_BUCKET)

    async def _call(self, key: str, kind: str, per_minute: int, n: float, force: bool) -> Decision:
        res = await self._script(
            keys=[f"{self._prefix}:{key}:{kind}:{per_minute}"],
            args=[per_minute, per_minute / 60.0, self._clock(), n, 1 if force else 0],
        )
        allowed, tokens_milli, retry_ms = (int(x) for x in res)
        remaining = tokens_milli / 1000.0
        if allowed:
            return Decision(True, 0.0, remaining)
        return Decision(False, math.inf if retry_ms == -1 else retry_ms / 1000.0, remaining)

    async def check(self, key: str, limits: Limits, est_tokens: int) -> Tuple[Decision, str]:
        if limits.rpm > 0:
            d = await self._call(key, "rpm", limits.rpm, 1, False)
            if not d.allowed:
                return d, "rpm"
        if limits.tpm > 0 and est_tokens > 0:
            d = await self._call(key, "tpm", limits.tpm, min(est_tokens, limits.tpm), False)
            if not d.allowed:
                if limits.rpm > 0:
                    await self._call(key, "rpm", limits.rpm, -1, True)
                return d, "tpm"
        return Decision(True), ""

    async def settle(self, key: str, limits: Limits, est_tokens: int, actual_tokens: int) -> None:
        if limits.tpm > 0:
            charged = min(max(est_tokens, 0), limits.tpm)
            await self._call(key, "tpm", limits.tpm, actual_tokens - charged, True)


def make_limiter(redis: Optional[Any]) -> Any:
    return RedisRateLimiter(redis) if redis is not None else RateLimiter()
