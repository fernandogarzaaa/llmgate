import numpy as np
import pytest

from llmgate.cache import (
    KEY_FIELDS,
    MemoryCache,
    RedisCache,
    SemanticCache,
    _last_user_index,
    cache_key,
    is_cacheable,
    normalize,
    semantic_parts,
)


def body(**kw):
    b = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


# ------------------------------------------------------------------------- exact keys
def test_key_is_stable_across_releases():
    # Changing these breaks every shared (Redis) cache on upgrade: bump deliberately.
    assert (
        cache_key(body(), "m", "t") == "llmgate:exact:6b6d86e6a0867c0e9fd9c7eb424401fde6f80f3b3ef6c1daf3be5b8d057fcd8c"
    )
    assert (
        cache_key({"messages": [{"role": "user", "content": "héllo"}], "temperature": 0}, "m", "t")
        == "llmgate:exact:49bc2c7d5ea6ac7e2fa8eb96ffcb263697282f4c1dfb85ef7b4892b3245e9c71"
    )


def test_key_ignores_non_semantic_fields_and_key_order():
    k = cache_key(body(), "m", "t")
    assert k == cache_key(body(stream=True, user="u", metadata={"a": 1}, stream_options={}), "m", "t")
    assert k == cache_key({"messages": [{"content": "hi", "role": "user"}], "model": "m"}, "m", "t")
    assert k == cache_key(body(temperature=None), "m", "t")


def test_key_uses_routed_model_not_requested():
    assert cache_key(body(model="auto"), "m", "t") == cache_key(body(model="other"), "m", "t")


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("max_tokens", 5),
        ("seed", 1),
        ("tools", [{"type": "function"}]),
        ("response_format", {"type": "json"}),
        ("stop", ["x"]),
        ("n", 1),
    ],
)
def test_key_depends_on_sampling_fields(field, value):
    assert cache_key(body(), "m", "t") != cache_key(body(**{field: value}), "m", "t")


def test_key_depends_on_scope_model_and_messages():
    k = cache_key(body(), "m", "t")
    assert k != cache_key(body(), "m", "other")
    assert k != cache_key(body(), "m2", "t")
    assert k != cache_key(body(messages=[{"role": "user", "content": "ho"}]), "m", "t")
    assert "model" in KEY_FIELDS and "stream" not in KEY_FIELDS


def test_is_cacheable():
    assert is_cacheable(body(), False)
    assert is_cacheable(body(n=1), False)
    assert is_cacheable(body(n=None), False)
    assert is_cacheable(body(n=2), False) is False
    assert is_cacheable(body(), True) is False
    assert is_cacheable(body(temperature=0), True)
    assert is_cacheable(body(temperature=0.0), True)
    assert is_cacheable(body(temperature=0.1), True) is False
    assert is_cacheable(body(temperature=0, n=3), True) is False


# --------------------------------------------------------------------- semantic parts
def test_last_user_index():
    assert _last_user_index([]) == -1
    assert _last_user_index([{"role": "system"}]) == -1
    assert _last_user_index([{"role": "user"}]) == 0
    assert _last_user_index([{"role": "user"}, {"role": "user"}, {"role": "assistant"}]) == 1


def test_semantic_parts_basic():
    ns, text = semantic_parts(body(messages=[{"role": "user", "content": "  hi  "}]), "m", "t")
    assert text == "hi" and len(ns) == 32
    assert ns == "bd1592ab1fd61c1bb8a6ce94788c7349"


def test_semantic_namespace_separates_context():
    q = [{"role": "user", "content": "q"}]
    base = semantic_parts({"messages": q}, "m", "t")[0]
    assert base == semantic_parts({"messages": q, "stream": True, "user": "x"}, "m", "t")[0]
    assert base == semantic_parts({"messages": q, "temperature": None}, "m", "t")[0]
    assert base != semantic_parts({"messages": q, "temperature": 0.7}, "m", "t")[0]
    assert base != semantic_parts({"messages": q}, "m2", "t")[0]
    assert base != semantic_parts({"messages": q}, "m", "t2")[0]
    sys = [{"role": "system", "content": "pirate"}] + q
    assert base != semantic_parts({"messages": sys}, "m", "t")[0]


@pytest.mark.parametrize(
    "b",
    [
        {},
        {"messages": "hi"},
        {"messages": []},
        {"messages": [{"role": "system", "content": "s"}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]},
        {"messages": [{"role": "user", "content": "   "}]},
        {"messages": [{"role": "user", "content": "q"}], "tools": [{"type": "function"}]},
        {"messages": [{"role": "user", "content": "q"}], "functions": [{"name": "f"}]},
    ],
)
def test_semantic_parts_rejects_unusable_requests(b):
    assert semantic_parts(b, "m", "t") is None


def test_semantic_parts_empty_tools_is_fine():
    assert semantic_parts({"messages": [{"role": "user", "content": "q"}], "tools": []}, "m", "t") is not None


# ----------------------------------------------------------------------- MemoryCache
async def test_memory_cache_defaults_and_validation():
    assert MemoryCache().max_entries == 10_000
    MemoryCache(1)
    with pytest.raises(ValueError):
        MemoryCache(0)
    assert await MemoryCache().ping() is True


async def test_memory_cache_ttl_boundary():
    t = [0.0]
    c = MemoryCache(10, clock=lambda: t[0])
    await c.set("k", {"a": 1}, ttl_s=5)
    assert await c.get("k") == {"a": 1}
    t[0] = 4.99
    assert await c.get("k") == {"a": 1}
    t[0] = 5.0
    assert await c.get("k") is None
    assert len(c) == 0  # expired entries are dropped on read


async def test_memory_cache_no_ttl_never_expires():
    t = [0.0]
    c = MemoryCache(10, clock=lambda: t[0])
    await c.set("k", 1)
    await c.set("z", 2, ttl_s=0)
    t[0] = 1e9
    assert await c.get("k") == 1 and await c.get("z") == 2
    assert await c.get("missing") is None


async def test_memory_cache_lru_eviction_respects_reads_and_writes():
    c = MemoryCache(2)
    await c.set("a", 1)
    await c.set("b", 2)
    assert len(c) == 2
    await c.get("a")  # a is now most recent
    await c.set("c", 3)  # evicts b
    assert await c.get("b") is None and await c.get("a") == 1 and await c.get("c") == 3
    await c.set("a", 10)  # overwrite refreshes recency of a
    await c.set("d", 4)  # evicts c
    assert await c.get("c") is None and await c.get("a") == 10
    assert len(c) == 2


class FakeRedis:
    def __init__(self):
        self.data, self.ex = {}, {}

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v, ex=None):
        self.data[k] = v
        self.ex[k] = ex

    async def ping(self):
        return True


async def test_redis_cache_json_and_expiry():
    r = FakeRedis()
    c = RedisCache(r)
    assert await c.get("k") is None
    await c.set("k", {"a": [1]}, ttl_s=60)
    assert await c.get("k") == {"a": [1]} and r.ex["k"] == 60
    await c.set("short", 1, ttl_s=0.2)
    assert r.ex["short"] == 1  # Redis needs whole seconds >= 1
    await c.set("forever", 1)
    assert r.ex["forever"] is None
    await c.set("zero", 1, ttl_s=0)
    assert r.ex["zero"] is None
    assert await c.ping() is True


# --------------------------------------------------------------------- SemanticCache
VECS = {"a": [1, 0], "b": [0.99, 0.141], "c": [0, 1], "z": [0, 0], "d": [0.6, 0.8]}


async def embed(texts):
    return [np.array(VECS[t], dtype=np.float32) for t in texts]


def test_normalize():
    assert np.allclose(normalize(np.array([3.0, 4.0])), [0.6, 0.8])
    z = normalize(np.array([0.0, 0.0]))
    assert np.all(z == 0) and z.dtype == np.float32
    assert normalize(np.array([[1.0, 0.0]])).shape == (2,)


def test_semantic_cache_validation_and_defaults():
    sc = SemanticCache(embed)
    assert (sc.threshold, sc.max_entries, sc.ttl_s) == (0.9, 5_000, 0)
    SemanticCache(embed, threshold=1.0, max_entries_per_namespace=1)
    for kw in ({"threshold": 0.0}, {"threshold": 1.01}, {"max_entries_per_namespace": 0}):
        with pytest.raises(ValueError):
            SemanticCache(embed, **kw)


async def test_semantic_cache_hit_and_miss():
    sc = SemanticCache(embed, threshold=0.95)
    await sc.store("ns", "a", {"answer": 1})
    hit = await sc.lookup("ns", "b")
    assert hit.hit and hit.payload == {"answer": 1} and hit.matched_text == "a"
    assert hit.similarity == pytest.approx(0.9902, abs=1e-3)
    miss = await sc.lookup("ns", "c")
    assert not miss.hit and miss.payload is None and miss.similarity == pytest.approx(0.0, abs=1e-6)
    assert miss.vector is not None and np.allclose(miss.vector, [0, 1])
    assert not (await sc.lookup("other", "b")).hit
    assert (await sc.lookup("empty", "a")).similarity == -1.0


async def test_semantic_threshold_is_inclusive_and_overridable():
    sc = SemanticCache(embed, threshold=0.6)
    await sc.store("ns", "a", 1)
    assert (await sc.lookup("ns", "d")).hit  # cos = 0.6 exactly
    assert not (await sc.lookup("ns", "d", threshold=0.61)).hit
    assert (await sc.lookup("ns", "c", threshold=0.0 + 1e-9)).hit is False


async def test_semantic_best_match_wins():
    sc = SemanticCache(embed, threshold=0.5)
    await sc.store("ns", "c", "C")
    await sc.store("ns", "a", "A")
    await sc.store("ns", "d", "D")
    assert (await sc.lookup("ns", "b")).payload == "A"
    assert sc.size() == 3


async def test_semantic_store_uses_given_vector():
    sc = SemanticCache(embed, threshold=0.99)
    await sc.store("ns", "unknown-text", "X", vector=np.array([0.0, 2.0]))
    assert (await sc.lookup("ns", "c")).payload == "X"


async def test_semantic_ttl_expiry():
    t = [100.0]
    sc = SemanticCache(embed, threshold=0.9, ttl_s=10, clock=lambda: t[0])
    await sc.store("ns", "a", "A")
    t[0] = 109.9
    assert (await sc.lookup("ns", "a")).hit
    t[0] = 110.0
    assert not (await sc.lookup("ns", "a")).hit
    assert sc.size() == 0
    await sc.store("ns", "c", "C")  # matrix rebuilt after expiry
    assert (await sc.lookup("ns", "c")).payload == "C"


async def test_semantic_max_entries_keeps_newest():
    sc = SemanticCache(embed, threshold=0.99, max_entries_per_namespace=2)
    await sc.store("ns", "a", "A")
    await sc.store("ns", "c", "C")
    assert sc.size() == 2
    await sc.store("ns", "d", "D")
    assert sc.size() == 2
    assert not (await sc.lookup("ns", "a")).hit
    assert (await sc.lookup("ns", "c")).hit and (await sc.lookup("ns", "d")).hit


async def test_semantic_sqlite_persistence(tmp_path):
    path = str(tmp_path / "sem.db")
    t = [0.0]
    sc = SemanticCache(embed, threshold=0.95, ttl_s=50, sqlite_path=path, clock=lambda: t[0])
    await sc.store("ns", "a", {"answer": 1})
    await sc.store("ns", "c", {"answer": 2})
    sc.close()
    sc.close()  # idempotent
    sc2 = SemanticCache(
        embed, threshold=0.95, max_entries_per_namespace=1, ttl_s=50, sqlite_path=path, clock=lambda: t[0]
    )
    assert (await sc2.lookup("ns", "c")).payload == {"answer": 2}
    assert not (await sc2.lookup("ns", "a")).hit  # only the newest max_entries are loaded
    t[0] = 49.0
    assert (await sc2.lookup("ns", "c")).hit
    t[0] = 50.0
    assert not (await sc2.lookup("ns", "c")).hit
    assert not (await sc2.lookup("other", "c")).hit
    sc2.close()


async def test_memory_cache_sub_second_ttl():
    t = [0.0]
    c = MemoryCache(10, clock=lambda: t[0])
    await c.set("k", 1, ttl_s=0.5)
    t[0] = 0.6
    assert await c.get("k") is None


async def test_semantic_exact_match_at_threshold_one():
    sc = SemanticCache(embed, threshold=1.0)
    await sc.store("ns", "a", "A")
    hit = await sc.lookup("ns", "a")
    assert hit.hit and hit.similarity == 1.0


async def test_semantic_sub_second_ttl():
    t = [0.0]
    sc = SemanticCache(embed, threshold=0.9, ttl_s=0.5, clock=lambda: t[0])
    await sc.store("ns", "a", "A")
    t[0] = 0.6
    assert not (await sc.lookup("ns", "a")).hit


async def test_semantic_sqlite_schema_threads_and_close(tmp_path):
    import asyncio
    import sqlite3

    path = str(tmp_path / "s.db")
    sc = SemanticCache(embed, threshold=0.9, sqlite_path=path)
    idx = (
        sqlite3.connect(path)
        .execute("SELECT name FROM sqlite_master WHERE type='index' AND name='ix_semantic_ns'")
        .fetchall()
    )
    assert idx == [("ix_semantic_ns",)]
    # the gateway stores from worker threads: the connection must allow it
    await asyncio.to_thread(asyncio.run, sc.store("ns", "a", "A"))
    assert (await sc.lookup("ns", "a")).hit
    conn = sc._db
    sc.close()
    assert sc._db is None
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    SemanticCache(embed).close()  # memory-only close is a no-op
