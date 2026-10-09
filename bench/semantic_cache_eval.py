"""Semantic-cache quality on Quora Question Pairs (QQP, GLUE validation split).

Simulation: for each evaluated pair (q1, q2, is_duplicate), q1 is stored in the
cache with its "answer"; then q2 is looked up against *all* stored questions
(one shared namespace, as for a single tenant + model). A lookup is

* a **correct hit** if the nearest stored question is q2's own partner and the
  pair is labelled duplicate (serving q1's answer for q2 is right),
* a **false hit** otherwise (a non-duplicate partner, or a different question
  altogether: the user gets an answer to another question),
* a **miss** when the best similarity is below the threshold.

The threshold is chosen on a dev half (highest hit rate with hit precision >=
the target, default 0.95) and reported on the held-out test half, together with
a sweep. Dollar savings are computed for stated prices and token counts.

    python bench/semantic_cache_eval.py --embedders hashing,fastembed --pairs 4000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from llmgate.embeddings import make_embedder  # noqa: E402

ROWS_API = "https://datasets-server.huggingface.co/rows"

# Stated pricing (USD per 1M tokens, list prices used throughout the README) and request size.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "claude-sonnet": (3.00, 15.00)}
PROMPT_TOKENS, COMPLETION_TOKENS = 500, 250


def fetch_qqp(n: int, path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        if len(rows) >= n:
            return rows[:n]
    rows: List[Dict[str, Any]] = []
    with httpx.Client(timeout=60) as c:
        for off in range(0, n, 100):
            r = c.get(
                ROWS_API,
                params={
                    "dataset": "nyu-mll/glue",
                    "config": "qqp",
                    "split": "validation",
                    "offset": off,
                    "length": min(100, n - off),
                },
            )
            r.raise_for_status()
            for item in r.json()["rows"]:
                row = item["row"]
                rows.append({"q1": row["question1"], "q2": row["question2"], "label": int(row["label"])})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def embed_all(emb: Any, texts: List[str], batch: int = 64) -> Tuple[np.ndarray, float]:
    out: List[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        out.extend(asyncio.run(emb(texts[i : i + batch])))
    dt = time.perf_counter() - t0
    m = np.stack([np.asarray(v, dtype=np.float32) for v in out])
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return m, dt / len(texts) * 1000


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def embed_cached(emb: Any, name: str, split: str, side: str, texts: List[str]) -> Tuple[np.ndarray, float]:
    """Embeddings are cached under bench/data (not committed); the timing is from the first run."""
    base = os.path.join(HERE, "data", f"emb_{name}_{split}_{side}_{len(texts)}")
    if os.path.exists(base + ".npy") and os.path.exists(base + ".json"):
        return np.load(base + ".npy"), json.load(open(base + ".json"))["ms_per_text"]
    m, ms = embed_all(emb, texts)
    np.save(base + ".npy", m)
    json.dump({"ms_per_text": ms}, open(base + ".json", "w"))
    return m, ms


def simulate(
    store: np.ndarray, queries: np.ndarray, labels: np.ndarray, q1: List[str], q2: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (best similarity, whether serving the best match is correct) per query.

    Correct = the matched stored question is the query's own labelled-duplicate partner (or has
    exactly the same text as that partner, since QQP repeats questions), or the matched stored
    question is textually identical to the query itself.
    """
    sims = queries @ store.T
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(queries)), best]
    n1 = [_norm(t) for t in q1]
    correct = np.array(
        [(labels[i] == 1 and (b == i or n1[b] == n1[i])) or n1[b] == _norm(q2[i]) for i, b in enumerate(best)]
    )
    return best_sim, correct


def metrics_at(best_sim: np.ndarray, correct: np.ndarray, labels: np.ndarray, t: float) -> Dict[str, float]:
    hit = best_sim >= t
    n, nh = len(hit), int(hit.sum())
    ch = int((hit & correct).sum())
    return {
        "threshold": round(float(t), 3),
        "hit_rate": round(nh / n, 4),
        "correct_hit_rate": round(ch / n, 4),
        "false_hit_rate": round((nh - ch) / n, 4),
        "hit_precision": round(ch / nh, 4) if nh else None,
        "duplicate_recall": round(ch / max(1, int(labels.sum())), 4),
        "false_hit_rate_on_non_duplicates": round(
            int((hit & (labels == 0)).sum()) / max(1, int((labels == 0).sum())), 4
        ),
    }


def choose_threshold(best_sim: np.ndarray, correct: np.ndarray, labels: np.ndarray, target: float) -> float:
    for t in np.round(np.arange(0.50, 1.0001, 0.005), 3):
        m = metrics_at(best_sim, correct, labels, t)
        if m["hit_precision"] is not None and m["hit_precision"] >= target:
            return float(t)
    return 1.0


def dollars(hit_rate: float, false_rate: float) -> Dict[str, Any]:
    out = {}
    for model, (pin, pout) in PRICES.items():
        per_req = (PROMPT_TOKENS * pin + COMPLETION_TOKENS * pout) / 1e6
        out[model] = {
            "cost_per_request_usd": round(per_req, 6),
            "saved_per_1M_requests_usd": round(hit_rate * per_req * 1e6, 2),
            "wrong_answers_per_1M_requests": int(round(false_rate * 1e6)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--embedders", default="hashing,fastembed")
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "semantic_cache.json"))
    args = ap.parse_args()

    rows = fetch_qqp(args.pairs, os.path.join(HERE, "data", "qqp_validation.jsonl"))
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    half = len(idx) // 2
    splits = {"dev": [rows[i] for i in idx[:half]], "test": [rows[i] for i in idx[half:]]}
    report: Dict[str, Any] = {
        "dataset": f"GLUE QQP validation (nyu-mll/glue, config qqp), first {len(rows)} pairs, "
        f"shuffled seed {args.seed}, split 50/50",
        "n_dev": half,
        "n_test": len(rows) - half,
        "duplicate_share_test": round(sum(r["label"] for r in splits["test"]) / (len(rows) - half), 4),
        "pricing_assumption": {
            "prices_usd_per_1M_tokens": PRICES,
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
        },
        "target_hit_precision_on_dev": args.target_precision,
        "embedders": {},
    }
    for name in args.embedders.split(","):
        try:
            emb = make_embedder(name)
        except ImportError as e:
            report["embedders"][name] = {"error": str(e)}
            continue
        res: Dict[str, Any] = {"name": emb.name}
        per_split = {}
        for split, data in splits.items():
            t1, t2 = [r["q1"] for r in data], [r["q2"] for r in data]
            q1, q1_ms = embed_cached(emb, name, split, "q1", t1)
            q2, _ = embed_cached(emb, name, split, "q2", t2)
            labels = np.array([r["label"] for r in data])
            per_split[split] = (*simulate(q1, q2, labels, t1, t2), labels)
            res.setdefault("embed_ms_per_text", round(q1_ms, 3))
        thr = choose_threshold(*per_split["dev"], args.target_precision)
        res["chosen_threshold"] = thr
        res["test_at_chosen"] = metrics_at(*per_split["test"], thr)
        res["test_sweep"] = [metrics_at(*per_split["test"], t) for t in (0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99)]
        m = res["test_at_chosen"]
        res["dollars_at_chosen"] = dollars(m["hit_rate"], m["false_hit_rate"])
        report["embedders"][name] = res
        print(name, json.dumps(res["test_at_chosen"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
