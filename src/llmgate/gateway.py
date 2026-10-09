"""Gateway state: everything a request needs, built once from the config."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .budget import Budget, BudgetTracker
from .cache import MemoryCache, RedisCache, SemanticCache
from .config import GatewayConfig
from .cost import Price, cost_usd
from .costroute_adapter import load_scorer
from .dispatch import Attempt, Dispatcher, ModelTarget
from .embeddings import make_embedder
from .guard import GuardBackend, load_guard
from .metrics import Metrics
from .providers import Provider, make_provider
from .ratelimit import Limits, make_limiter
from .resilience import STATE_VALUE
from .routing import CostRouteConfig, Router, rules_from_config
from .store import KeyRecord, Store, Tenant
from .tracing import setup_tracing

log = logging.getLogger("llmgate")

ANONYMOUS = Tenant(id="anonymous", name="anonymous")


@dataclass
class AuthContext:
    tenant: Tenant
    key_id: Optional[str]

    @property
    def budget(self) -> Budget:
        return Budget(self.tenant.budget_usd, self.tenant.budget_period)


class UsageWriter:
    """Batches usage rows and writes them off the event loop."""

    def __init__(self, store: Store, flush_s: float = 0.5, max_batch: int = 500) -> None:
        self.store = store
        self.flush_s = flush_s
        self.max_batch = max_batch
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.pending: Dict[str, float] = {}  # tenant id -> cost queued but not yet written
        self._task: Optional[asyncio.Task[None]] = None

    def put(self, row: Dict[str, Any]) -> None:
        self.pending[row["tenant_id"]] = self.pending.get(row["tenant_id"], 0.0) + row["cost_usd"]
        self.queue.put_nowait(row)

    async def flush(self) -> int:
        rows: List[Dict[str, Any]] = []
        while not self.queue.empty() and len(rows) < self.max_batch:
            rows.append(self.queue.get_nowait())
        if rows:
            try:
                await asyncio.to_thread(self.store.record_usage, rows)
            except Exception:  # never lose the request path over accounting; retry next tick
                log.exception("usage write failed; re-queueing %d rows", len(rows))
                for r in rows:
                    self.queue.put_nowait(r)
                return 0
            for r in rows:
                self.pending[r["tenant_id"]] = max(0.0, self.pending.get(r["tenant_id"], 0.0) - r["cost_usd"])
        return len(rows)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.flush_s)
            while await self.flush() >= self.max_batch:
                pass

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        while await self.flush():
            pass


class Gateway:
    def __init__(
        self,
        cfg: GatewayConfig,
        store: Optional[Store] = None,
        redis: Any = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.cfg = cfg
        self.metrics = Metrics()
        self.store = store or Store(cfg.database_url)
        self.redis = redis
        if self.redis is None and cfg.redis_url:
            import redis.asyncio as aioredis

            self.redis = aioredis.from_url(cfg.redis_url)
        self.http = http_client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200)
        )
        self.providers: Dict[str, Provider] = {p.name: make_provider(p, self.http) for p in cfg.providers}
        self.targets: Dict[str, ModelTarget] = {
            m.name: ModelTarget(m.name, m.provider, m.upstream_model or m.name, Price(m.input_per_1m, m.output_per_1m))
            for m in cfg.models
        }
        self.dispatcher = Dispatcher(self.targets, self.providers, on_attempt=self._on_attempt)

        cr = cfg.routing.costroute
        self.scorer: Any = None
        self.scorer_source = ""
        crc: Optional[CostRouteConfig] = None
        if cr.enabled:
            self.scorer, self.scorer_source = load_scorer(cr.router)
            crc = CostRouteConfig(cr.alias, cr.strong, cr.weak, cr.threshold)
        self.router = Router(
            {m.name: m.fallbacks for m in cfg.models},
            rules_from_config(cfg.routing.rules),
            crc,
            self.scorer,
            cfg.routing.default_model,
        )

        self.guard: Optional[GuardBackend] = load_guard(cfg.guard)
        self.limiter = make_limiter(self.redis)
        self.budgets = BudgetTracker()

        ec = cfg.cache.exact
        self.exact: Any = None
        if ec.enabled:
            self.exact = RedisCache(self.redis) if ec.backend == "redis" else MemoryCache(ec.max_entries)
        sc = cfg.cache.semantic
        self.semantic: Optional[SemanticCache] = None
        self.embedder_name = ""
        if sc.enabled:
            emb = make_embedder(
                sc.embedder,
                self.http,
                model=sc.embedder_model,
                base_url=sc.embedder_base_url,
                api_key=sc.embedder_api_key,
            )
            self.embedder_name = emb.name
            self.semantic = SemanticCache(
                emb,
                sc.threshold,
                sc.max_entries_per_namespace,
                sc.ttl_s,
                sc.sqlite_path if sc.backend == "sqlite" else None,
            )

        self.writer = UsageWriter(self.store)
        self._keys: Dict[str, Tuple[float, Optional[Tuple[KeyRecord, Tenant]]]] = {}
        self._known_tenants: Dict[str, Tenant] = {}
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._bg: set[asyncio.Future[None]] = set()
        self.tracing = setup_tracing(cfg.tracing)

    # ------------------------------------------------------------------ lifecycle
    def bootstrap(self) -> None:
        """Create the tenants/keys listed under ``tenancy.bootstrap`` if they don't exist."""
        for b in self.cfg.tenancy.bootstrap:
            t = self.store.get_tenant(b.tenant)
            if t is None:
                t = self.store.create_tenant(b.tenant, b.rpm, b.tpm, b.budget_usd, b.budget_period)
            if not self.store.key_exists(b.api_key):
                self.store.create_key(t.id, "bootstrap", key=b.api_key)

    async def start(self) -> None:
        await asyncio.to_thread(self.bootstrap)
        self.writer.start()
        self._refresh_task = asyncio.create_task(self._refresh_budgets_loop())

    async def stop(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        await self.writer.stop()
        await self.http.aclose()
        if self.semantic is not None:
            self.semantic.close()
        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception:
                pass

    async def refresh_budget(self, tenant: Tenant) -> None:
        b = Budget(tenant.budget_usd, tenant.budget_period)
        if b.unlimited:
            return
        spent = await asyncio.to_thread(self.store.tenant_spend, tenant.id, self.budgets.since(b))
        self.budgets.seed(tenant.id, b, spent, self.writer.pending.get(tenant.id, 0.0))

    async def _refresh_budgets_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.tenancy.budget_refresh_s)
            for t in list(self._known_tenants.values()):
                try:
                    await self.refresh_budget(t)
                except Exception:
                    log.exception("budget refresh failed for tenant %s", t.name)

    # ----------------------------------------------------------------------- auth
    async def authenticate(self, key: Optional[str]) -> Optional[AuthContext]:
        if not key:
            return AuthContext(ANONYMOUS, None) if not self.cfg.tenancy.require_auth else None
        now = time.monotonic()
        cached = self._keys.get(key)
        if cached is not None and cached[0] > now:
            found = cached[1]
        else:
            found = await asyncio.to_thread(self.store.resolve_key, key)
            ttl = self.cfg.tenancy.key_cache_s if found is not None else min(5.0, self.cfg.tenancy.key_cache_s)
            if len(self._keys) > 50_000:
                self._keys.clear()
            self._keys[key] = (now + ttl, found)
        if found is None:
            return None
        rec, tenant = found
        if tenant.id not in self._known_tenants:
            self._known_tenants[tenant.id] = tenant
            await self.refresh_budget(tenant)
        return AuthContext(tenant, rec.id)

    def limits_for(self, tenant: Tenant) -> Limits:
        t = self.cfg.tenancy
        return Limits(tenant.rpm or t.default_rpm, tenant.tpm or t.default_tpm)

    # -------------------------------------------------------------- accounting
    def _on_attempt(self, a: Attempt) -> None:
        self.metrics.upstream_latency.labels(a.provider, "ok" if a.ok else a.kind).observe(a.latency_s)
        if not a.ok:
            self.metrics.upstream_errors.labels(a.provider, a.kind).inc()
        prov = self.providers.get(a.provider)
        if prov is not None:
            self.metrics.breaker.labels(a.provider).set(STATE_VALUE[prov.breaker.state])

    def account(
        self,
        *,
        auth: AuthContext,
        request_id: str,
        model: str,
        provider: str,
        route: str,
        cache: str,
        status: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        est_tokens: int,
    ) -> Tuple[float, float]:
        """Record cost, tokens, budget spend, TPM correction and the usage row. Returns (cost, saved)."""
        price = self.targets[model].price if model in self.targets else Price()
        full = cost_usd(prompt_tokens, completion_tokens, price)
        cost, saved = (0.0, full) if cache != "miss" else (full, 0.0)
        tname = auth.tenant.name
        m = self.metrics
        m.requests.labels(tname, model, str(status), cache).inc()
        if cache == "miss":
            m.tokens.labels(tname, model, "prompt").inc(prompt_tokens)
            m.tokens.labels(tname, model, "completion").inc(completion_tokens)
            m.cost.labels(tname, model).inc(cost)
            if cost > 0:
                self.budgets.add(auth.tenant.id, auth.budget, cost)
            task = asyncio.ensure_future(
                self.limiter.settle(
                    auth.tenant.id, self.limits_for(auth.tenant), est_tokens, prompt_tokens + completion_tokens
                )
            )
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        else:
            m.saved.labels(tname, model).inc(saved)
        self.writer.put(
            {
                "ts": datetime.now(timezone.utc),
                "request_id": request_id,
                "tenant_id": auth.tenant.id,
                "key_id": auth.key_id,
                "model": model,
                "provider": provider,
                "route": route,
                "cache": cache,
                "status": status,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
                "saved_usd": saved,
                "latency_ms": latency_s * 1000,
            }
        )
        return cost, saved

    # ---------------------------------------------------------------- health
    async def readiness(self) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}
        try:
            await asyncio.wait_for(asyncio.to_thread(self.store.ping), 3)
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc.__class__.__name__}"
        if self.redis is not None:
            try:
                await asyncio.wait_for(self.redis.ping(), 3)
                checks["redis"] = "ok"
            except Exception as exc:
                checks["redis"] = f"error: {exc.__class__.__name__}"
        states = {n: p.breaker.state for n, p in self.providers.items()}
        checks["providers"] = states
        ready = all(v == "ok" for k, v in checks.items() if k != "providers") and any(
            s != "open" for s in states.values()
        )
        return {"ready": ready, "checks": checks}

    def components(self) -> Dict[str, Any]:
        cr = self.cfg.routing.costroute
        return {
            "guard": self.guard.name if self.guard is not None else "disabled",
            "router": {
                "rules": len(self.router.rules),
                "costroute": f"{cr.router} via {self.scorer_source}" if cr.enabled else "disabled",
            },
            "cache": {
                "exact": self.cfg.cache.exact.backend if self.exact is not None else "disabled",
                "semantic": f"{self.cfg.cache.semantic.backend}/{self.embedder_name}"
                if self.semantic is not None
                else "disabled",
            },
            "rate_limiter": "redis" if self.redis is not None else "memory",
            "database": self.store.url.split(":", 1)[0],
            "tracing": self.tracing,
        }
