"""Response caching: an exact cache and a semantic (embedding-similarity) cache.

**Exact cache.** The key is a SHA-256 over a canonical JSON of every request
field that can change the answer (model, messages, sampling parameters, tools,
response format, seed ...) plus the scope (the tenant, unless the cache is
configured as global). Fields that cannot change the answer (``stream``,
``user``, ``metadata``, ``stream_options``) are left out, so a streamed and a
non-streamed request share an entry.

**Semantic cache.** The last user message is embedded and compared by cosine
similarity with earlier requests in the same *namespace*: same scope, same
model, same sampling parameters, and the same conversation before the last user
turn (system prompt and history are hashed into the namespace). The best match
is a hit when its similarity is >= the threshold.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

# Request fields that can change the response. Everything else is ignored for keying.
KEY_FIELDS = (
    "model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens", "n", "stop",
    "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "seed",
    "tools", "tool_choice", "functions", "function_call", "response_format", "parallel_tool_calls",
)
_PARAM_FIELDS = tuple(f for f in KEY_FIELDS if f not in ("model", "messages"))


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(body: Mapping[str, Any], model: str, scope: str) -> str:
    """Exact-cache key for a chat request routed to ``model`` within ``scope``."""
    picked = {k: body[k] for k in KEY_FIELDS if k in body and body[k] is not None}
    picked["model"] = model
    return "llmgate:exact:" + hashlib.sha256(_canon({"scope": scope, "req": picked}).encode()).hexdigest()


def is_cacheable(body: Mapping[str, Any], only_deterministic: bool) -> bool:
    """Requests with n > 1 are never cached; with ``only_deterministic`` only temperature 0 is."""
    n = body.get("n")
    if n is not None and n != 1:
        return False
    if only_deterministic:
        t = body.get("temperature")
        return t is not None and float(t) == 0.0
    return True


def _last_user_index(messages: List[Mapping[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def semantic_parts(body: Mapping[str, Any], model: str, scope: str) -> Optional[Tuple[str, str]]:
    """``(namespace, text)`` for the semantic cache, or ``None`` if the request has no usable
    last user turn (no user message, a non-text last user message, or tool calls involved)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    if body.get("tools") or body.get("functions"):
        return None
    i = _last_user_index(messages)
    if i < 0 or i != len(messages) - 1:
        return None
    content = messages[i].get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    params = {k: body[k] for k in _PARAM_FIELDS if k in body and body[k] is not None}
    ns_src = _canon({"scope": scope, "model": model, "params": params, "history": messages[:i]})
    return hashlib.sha256(ns_src.encode()).hexdigest()[:32], content.strip()


# --------------------------------------------------------------------------- exact cache backends


class MemoryCache:
    """LRU + TTL cache in process memory."""

    def __init__(self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._clock = clock
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    async def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires and self._clock() >= expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl_s: float = 0) -> None:
        """Store ``value``; ``ttl_s`` of 0 means no expiry."""
        expires = self._clock() + ttl_s if ttl_s > 0 else 0.0
        with self._lock:
            self._data[key] = (expires, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    async def ping(self) -> bool:
        return True


class RedisCache:
    """Exact cache in Redis (values are JSON)."""

    def __init__(self, redis: Any) -> None:
        self._r = redis

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._r.get(key)
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, ttl_s: float = 0) -> None:
        data = json.dumps(value)
        if ttl_s > 0:
            await self._r.set(key, data, ex=max(1, int(ttl_s)))
        else:
            await self._r.set(key, data)

    async def ping(self) -> bool:
        return bool(await self._r.ping())


# --------------------------------------------------------------------------- semantic cache


@dataclass
class SemanticHit:
    payload: Optional[Any]
    similarity: float
    vector: Optional[np.ndarray]
    matched_text: str = ""

    @property
    def hit(self) -> bool:
        return self.payload is not None


@dataclass
class _Entry:
    vec: np.ndarray
    text: str
    payload: Any
    expires: float


class _Namespace:
    def __init__(self) -> None:
        self.entries: List[_Entry] = []
        self.matrix: Optional[np.ndarray] = None

    def add(self, e: _Entry, max_entries: int) -> None:
        self.entries.append(e)
        if len(self.entries) > max_entries:
            del self.entries[: len(self.entries) - max_entries]
        self.matrix = None

    def best(self, vec: np.ndarray, now: float) -> Tuple[int, float]:
        live = [e for e in self.entries if not e.expires or e.expires > now]
        if len(live) != len(self.entries):
            self.entries = live
            self.matrix = None
        if not self.entries:
            return -1, -1.0
        if self.matrix is None:
            self.matrix = np.stack([e.vec for e in self.entries])
        sims = self.matrix @ vec
        i = int(np.argmax(sims))
        return i, float(sims[i])


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n


class SemanticCache:
    """Nearest-neighbour cache over normalised embeddings (brute-force cosine per namespace).

    ``embed`` is an async callable mapping a list of texts to vectors. With
    ``sqlite_path`` set, entries are persisted and reloaded on first use of a
    namespace; otherwise they live in memory only.
    """

    def __init__(
        self,
        embed: Callable[[List[str]], Any],
        threshold: float = 0.9,
        max_entries_per_namespace: int = 5_000,
        ttl_s: float = 0,
        sqlite_path: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries_per_namespace < 1:
            raise ValueError("max_entries_per_namespace must be >= 1")
        self.embed = embed
        self.threshold = threshold
        self.max_entries = max_entries_per_namespace
        self.ttl_s = ttl_s
        self._clock = clock
        self._ns: Dict[str, _Namespace] = {}
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        if sqlite_path:
            self._db = sqlite3.connect(sqlite_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache (namespace TEXT NOT NULL, text TEXT NOT NULL, "
                "vec BLOB NOT NULL, payload TEXT NOT NULL, expires REAL NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_semantic_ns ON semantic_cache(namespace)")
            self._db.commit()

    def _namespace(self, ns: str) -> _Namespace:
        with self._lock:
            space = self._ns.get(ns)
            if space is None:
                space = _Namespace()
                if self._db is not None:
                    rows = self._db.execute(
                        "SELECT text, vec, payload, expires FROM semantic_cache WHERE namespace = ? ORDER BY rowid",
                        (ns,),
                    ).fetchall()
                    for text, blob, payload, expires in rows[-self.max_entries:]:
                        space.entries.append(
                            _Entry(np.frombuffer(blob, dtype=np.float32).copy(), text, json.loads(payload), expires))
                self._ns[ns] = space
            return space

    async def _vector(self, text: str) -> np.ndarray:
        vecs = await self.embed([text])
        return normalize(np.asarray(vecs[0]))

    async def lookup(self, ns: str, text: str, threshold: Optional[float] = None) -> SemanticHit:
        vec = await self._vector(text)
        space = self._namespace(ns)
        with self._lock:
            i, sim = space.best(vec, self._clock())
            thr = self.threshold if threshold is None else threshold
            if i >= 0 and sim >= thr:
                e = space.entries[i]
                return SemanticHit(e.payload, sim, vec, e.text)
        return SemanticHit(None, sim, vec)

    async def store(self, ns: str, text: str, payload: Any, vector: Optional[np.ndarray] = None) -> None:
        vec = normalize(vector) if vector is not None else await self._vector(text)
        expires = self._clock() + self.ttl_s if self.ttl_s > 0 else 0.0
        space = self._namespace(ns)
        with self._lock:
            space.add(_Entry(vec, text, payload, expires), self.max_entries)
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO semantic_cache(namespace, text, vec, payload, expires) VALUES (?, ?, ?, ?, ?)",
                    (ns, text, vec.astype(np.float32).tobytes(), json.dumps(payload), expires),
                )
                self._db.commit()

    def size(self) -> int:
        return sum(len(s.entries) for s in self._ns.values())

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
