"""Call upstream models with retries, circuit breakers and fallbacks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from .cost import Price
from .providers.base import Provider, ProviderError
from .resilience import parse_retry_after

# The request itself is at fault: retrying or failing over cannot help.
CLIENT_ERRORS = frozenset({400, 413, 422})


@dataclass(frozen=True)
class ModelTarget:
    name: str
    provider: str
    upstream_model: str
    price: Price = field(default_factory=Price)


@dataclass
class Attempt:
    model: str
    provider: str
    ok: bool
    status: int = 200
    kind: str = "ok"
    latency_s: float = 0.0


@dataclass
class DispatchResult:
    model: str
    provider: str
    attempts: List[Attempt] = field(default_factory=list)
    payload: Optional[Dict[str, Any]] = None
    stream: Optional[AsyncIterator[bytes]] = None
    upstream_s: float = 0.0

    @property
    def fallback_used(self) -> bool:
        return bool(self.attempts) and self.attempts[0].model != self.model


class Dispatcher:
    def __init__(
        self,
        targets: Dict[str, ModelTarget],
        providers: Dict[str, Provider],
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        on_attempt: Optional[Callable[[Attempt], None]] = None,
    ) -> None:
        self.targets = targets
        self.providers = providers
        self._sleep = sleep
        self._on_attempt = on_attempt

    def _note(self, result: List[Attempt], a: Attempt) -> None:
        result.append(a)
        if self._on_attempt is not None:
            self._on_attempt(a)

    async def _run(self, candidates: List[str], body: Dict[str, Any], stream: bool) -> DispatchResult:
        attempts: List[Attempt] = []
        last: Optional[ProviderError] = None
        for model in candidates:
            target = self.targets[model]
            prov = self.providers[target.provider]
            payload = {**body, "model": target.upstream_model}
            for i in range(prov.retry.max_retries + 1):
                if not prov.breaker.allow():
                    last = ProviderError(503, f"circuit open for provider {prov.name!r}", True, "circuit_open")
                    self._note(attempts, Attempt(model, prov.name, False, 503, "circuit_open"))
                    break
                t0 = time.perf_counter()
                try:
                    if stream:
                        it = await prov.open_stream(payload)
                        dt = time.perf_counter() - t0
                        prov.breaker.record_success()
                        self._note(attempts, Attempt(model, prov.name, True, 200, "ok", dt))
                        return DispatchResult(model, prov.name, attempts, stream=it, upstream_s=dt)
                    data = await prov.complete(payload)
                    dt = time.perf_counter() - t0
                    prov.breaker.record_success()
                    self._note(attempts, Attempt(model, prov.name, True, 200, "ok", dt))
                    return DispatchResult(model, prov.name, attempts, payload=data, upstream_s=dt)
                except ProviderError as e:
                    dt = time.perf_counter() - t0
                    self._note(attempts, Attempt(model, prov.name, False, e.status, e.kind, dt))
                    if e.status in CLIENT_ERRORS:
                        prov.breaker.record_success()  # the upstream answered; the request was bad
                        raise
                    prov.breaker.record_failure()
                    last = e
                    if not e.retryable:
                        break  # e.g. 401/403/404 from this provider: fail over, don't retry
                    if i < prov.retry.max_retries:
                        wait = parse_retry_after(e.retry_after, prov.retry.max_s)
                        await self._sleep(prov.retry.delay(i) if wait is None else wait)
        if last is None:
            raise ProviderError(503, "no upstream model available", False, "no_candidates")
        raise last

    async def complete(self, candidates: List[str], body: Dict[str, Any]) -> DispatchResult:
        return await self._run(candidates, body, stream=False)

    async def stream(self, candidates: List[str], body: Dict[str, Any]) -> DispatchResult:
        """Retries and fallbacks apply until the first byte; after that the stream is committed."""
        return await self._run(candidates, body, stream=True)
