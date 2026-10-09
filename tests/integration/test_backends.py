"""Redis and Postgres backends. Real services in CI (LLMGATE_TEST_REDIS_URL / LLMGATE_TEST_DATABASE_URL);
locally the Redis tests fall back to fakeredis (with Lua support) when available."""

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from llmgate.cache import RedisCache
from llmgate.ratelimit import Limits, RedisRateLimiter
from llmgate.store import Store

REDIS_URL = os.environ.get("LLMGATE_TEST_REDIS_URL")
PG_URL = os.environ.get("LLMGATE_TEST_DATABASE_URL")


@pytest.fixture
async def redis():
    if REDIS_URL:
        import redis.asyncio as aioredis

        r = aioredis.from_url(REDIS_URL)
    else:
        fakeredis = pytest.importorskip("fakeredis")
        pytest.importorskip("lupa")
        r = fakeredis.FakeAsyncRedis()
    yield r
    await r.aclose()


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


async def test_redis_rate_limiter_rpm(redis):
    c = Clock()
    rl = RedisRateLimiter(redis, prefix=f"t:{uuid.uuid4().hex}", clock=c)
    lim = Limits(rpm=2)
    assert (await rl.check("a", lim, 0))[0].allowed
    assert (await rl.check("a", lim, 0))[0].allowed
    d, which = await rl.check("a", lim, 0)
    assert not d.allowed and which == "rpm"
    assert d.retry_after_s == pytest.approx(30.0, abs=0.01)
    c.t += 30
    assert (await rl.check("a", lim, 0))[0].allowed


async def test_redis_rate_limiter_tpm_and_settle(redis):
    c = Clock()
    rl = RedisRateLimiter(redis, prefix=f"t:{uuid.uuid4().hex}", clock=c)
    lim = Limits(tpm=100)
    assert (await rl.check("a", lim, 60))[0].allowed
    await rl.settle("a", lim, 60, 100)  # really used 100: bucket is now at 0
    d, which = await rl.check("a", lim, 1)
    assert not d.allowed and which == "tpm"
    d, _ = await rl.check("a", lim, 1000)  # larger than capacity: clamped to capacity, still waits
    assert not d.allowed


async def test_redis_exact_cache(redis):
    cache = RedisCache(redis)
    k = f"t:{uuid.uuid4().hex}"
    assert await cache.get(k) is None
    await cache.set(k, {"a": [1, 2]}, ttl_s=60)
    assert await cache.get(k) == {"a": [1, 2]}
    assert await cache.ping()


@pytest.mark.skipif(not PG_URL, reason="LLMGATE_TEST_DATABASE_URL not set")
def test_postgres_store_roundtrip():
    st = Store(PG_URL)
    name = f"t-{uuid.uuid4().hex[:8]}"
    t = st.create_tenant(name, rpm=5, budget_usd=2.0, allowed_models=["a"])
    plain, rec = st.create_key(name)
    found = st.resolve_key(plain)
    assert found is not None and found[1].name == name and found[1].allowed_models == ["a"]
    now = datetime.now(timezone.utc)
    st.record_usage(
        [
            {
                "ts": now,
                "request_id": "r",
                "tenant_id": t.id,
                "key_id": rec.id,
                "model": "m",
                "provider": "p",
                "route": "direct",
                "cache": "miss",
                "status": 200,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_usd": 0.25,
                "saved_usd": 0.0,
                "latency_ms": 12.0,
            }
        ]
    )
    assert st.tenant_spend(t.id, now - timedelta(minutes=1)) == pytest.approx(0.25)
    rows = st.usage_summary(now - timedelta(minutes=1), ["tenant", "model"], tenant=name)
    assert rows == [
        {
            "tenant": name,
            "model": "m",
            "requests": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_usd": 0.25,
            "saved_usd": 0.0,
        }
    ]
    assert st.revoke_key(rec.id) and st.resolve_key(plain) is None
