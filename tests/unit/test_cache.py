import numpy as np
import pytest

from llmgate.cache import MemoryCache, SemanticCache, cache_key, is_cacheable, semantic_parts


def body(**kw):
    b = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


def test_key_ignores_stream_and_user():
    assert cache_key(body(), "m", "t") == cache_key(body(stream=True, user="u"), "m", "t")


def test_key_depends_on_temperature_scope_and_model():
    k = cache_key(body(), "m", "t")
    assert k != cache_key(body(temperature=0.5), "m", "t")
    assert k != cache_key(body(), "m", "other")
    assert k != cache_key(body(), "m2", "t")


def test_is_cacheable():
    assert is_cacheable(body(), False)
    assert not is_cacheable(body(n=2), False)
    assert not is_cacheable(body(), True)
    assert is_cacheable(body(temperature=0), True)


async def test_memory_cache_ttl():
    t = [0.0]
    c = MemoryCache(10, clock=lambda: t[0])
    await c.set("k", {"a": 1}, ttl_s=5)
    assert await c.get("k") == {"a": 1}
    t[0] = 6
    assert await c.get("k") is None


async def test_semantic_cache_hit_and_miss():
    vecs = {"a": [1, 0], "b": [0.99, 0.141], "c": [0, 1]}

    async def embed(texts):
        return [np.array(vecs[t], dtype=np.float32) for t in texts]

    sc = SemanticCache(embed, threshold=0.95)
    await sc.store("ns", "a", {"answer": 1})
    assert (await sc.lookup("ns", "b")).hit
    assert not (await sc.lookup("ns", "c")).hit
    assert not (await sc.lookup("other", "b")).hit


def test_semantic_parts_requires_last_user():
    assert semantic_parts(body(), "m", "t") is not None
    b = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert semantic_parts(b, "m", "t") is None
