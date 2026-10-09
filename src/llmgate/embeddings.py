"""Text embedders for the semantic cache.

* ``hashing``: a dependency-free hashed bag of words + word bigrams + character
  trigrams (signed feature hashing, sublinear TF, L2-normalised). Fast and
  deterministic, but lexical: it cannot tell paraphrases from look-alikes as
  well as a neural model (see the README's measured false-hit rates).
* ``fastembed``: ONNX sentence embeddings via the ``fastembed`` package
  (default ``BAAI/bge-small-en-v1.5``), CPU-only, no torch.
* ``openai``: any OpenAI-compatible ``/v1/embeddings`` endpoint.
"""

from __future__ import annotations

import asyncio
import math
import re
import zlib
from typing import Any, List, Optional, Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        words = _WORD.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed_one(self, text: str) -> np.ndarray:
        counts: dict = {}
        for f in self._features(text):
            h = zlib.crc32(f.encode())
            idx, sign = h % self.dim, (1.0 if (h >> 31) & 1 else -1.0)
            weight = 1.0 if f[0] == "c" else 2.0  # words/bigrams count more than char n-grams
            counts[idx] = counts.get(idx, 0.0) + sign * weight
        v = np.zeros(self.dim, dtype=np.float32)
        for idx, c in counts.items():
            v[idx] = math.copysign(1.0 + math.log(abs(c)), c) if c else 0.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    async def __call__(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self.embed_one(t) for t in texts]


class FastEmbedEmbedder:
    """Sentence embeddings via fastembed (ONNX Runtime). Runs in a worker thread."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", threads: Optional[int] = 1) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("the fastembed embedder needs: pip install 'llmgate[embeddings]'") from None
        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model, threads=threads)

    def embed_sync(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(list(texts))]

    async def __call__(self, texts: Sequence[str]) -> List[np.ndarray]:
        return await asyncio.to_thread(self.embed_sync, texts)


class OpenAIEmbedder:
    """Calls an OpenAI-compatible ``POST {base_url}/embeddings``."""

    def __init__(self, client: Any, base_url: str, model: str, api_key: str = "") -> None:
        self.name = f"openai:{model}"
        self._client = client
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

    async def __call__(self, texts: Sequence[str]) -> List[np.ndarray]:
        r = await self._client.post(self._url, json={"model": self._model, "input": list(texts)},
                                    headers=self._headers)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]


def make_embedder(kind: str, client: Any = None, **kw: Any) -> Any:
    if kind == "hashing":
        return HashingEmbedder(int(kw.get("dim", 1024)))
    if kind == "fastembed":
        return FastEmbedEmbedder(kw.get("model") or "BAAI/bge-small-en-v1.5")
    if kind == "openai":
        if not kw.get("base_url") or not kw.get("model"):
            raise ValueError("openai embedder needs base_url and model")
        return OpenAIEmbedder(client, kw["base_url"], kw["model"], kw.get("api_key") or "")
    raise ValueError(f"unknown embedder {kind!r}; expected hashing | fastembed | openai")
