"""The in-memory semantic namespace against a brute-force reference.

The namespace keeps a row-aligned embedding matrix that grows in place, shifts on
eviction and is rebuilt lazily when entries expire. A randomized interleaving of
inserts, lookups, evictions and expiries must always return the same best entry
and similarity as a plain linear scan over the live entries.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pytest

from llmgate.cache import _Entry, _Namespace, normalize


def _vec(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    return normalize(rng.standard_normal(dim).astype(np.float32))


@pytest.mark.parametrize("seed", range(40))
def test_namespace_matches_linear_scan(seed: int) -> None:
    r = random.Random(seed)
    rng = np.random.default_rng(seed)
    ns = _Namespace()
    ref: List[_Entry] = []
    max_entries = r.randint(1, 12)
    now = 0.0
    lookups = 0
    for step in range(80):
        now += 1.0
        if r.random() < 0.6:
            expires = now + r.randint(1, 25) if r.random() < 0.4 else 0.0
            e = _Entry(_vec(rng), f"t{step}", step, expires)
            ns.add(e, max_entries)
            ref.append(e)
            del ref[:-max_entries]
        else:
            q = _vec(rng)
            ref = [e for e in ref if not e.expires or e.expires > now]
            i, sim = ns.best(q, now)
            lookups += 1
            if not ref:
                assert i is None and sim == -1.0
                continue
            sims = [float(e.vec @ q) for e in ref]
            j = int(np.argmax(sims))
            assert i is not None
            assert ns.entries[i] is ref[j]
            assert sim == pytest.approx(sims[j], abs=1e-5)
            assert len(ns.entries) == len(ref)
    assert lookups > 0


def test_matrix_grows_past_initial_capacity_after_a_lookup() -> None:
    rng = np.random.default_rng(1)
    ns = _Namespace()
    first = _Entry(_vec(rng), "a", 0, 0.0)
    ns.add(first, 100)
    assert ns.best(first.vec, 0.0)[0] == 0  # materialises a 1-row matrix
    entries = [first]
    for k in range(9):  # forces several doublings of the in-place matrix
        e = _Entry(_vec(rng), str(k), k, 0.0)
        ns.add(e, 100)
        entries.append(e)
    for idx, e in enumerate(entries):
        i, sim = ns.best(e.vec, 0.0)
        assert i == idx
        assert sim == pytest.approx(1.0, abs=1e-5)


def test_eviction_shifts_matrix_rows() -> None:
    rng = np.random.default_rng(2)
    ns = _Namespace()
    es = [_Entry(_vec(rng), str(k), k, 0.0) for k in range(5)]
    ns.add(es[0], 3)
    ns.best(es[0].vec, 0.0)
    for e in es[1:]:
        ns.add(e, 3)
    assert [e.text for e in ns.entries] == ["2", "3", "4"]
    for idx, e in enumerate(es[2:]):
        assert ns.best(e.vec, 0.0) == (idx, pytest.approx(1.0, abs=1e-5))
    i, sim = ns.best(es[0].vec, 0.0)
    assert sim < 0.999  # the evicted vector is gone from the matrix too


def test_expired_entries_are_swept_only_after_earliest_expiry() -> None:
    rng = np.random.default_rng(3)
    ns = _Namespace()
    a = _Entry(_vec(rng), "a", 0, 10.0)
    b = _Entry(_vec(rng), "b", 1, 0.0)
    ns.add(a, 10)
    ns.add(b, 10)
    assert ns.best(a.vec, 9.0)[0] == 0
    assert len(ns.entries) == 2
    i, sim = ns.best(a.vec, 10.0)
    assert [e.text for e in ns.entries] == ["b"]
    assert i == 0 and sim < 0.999
