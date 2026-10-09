"""Measure llmgate's latency overhead against a fixed-latency mock upstream.

For each gateway configuration this script starts a real gateway process
(uvicorn, 1 worker) in front of a real mock-upstream process, drives it with a
closed-loop asyncio load generator at several concurrency levels, and compares
the client-observed latency with the same load sent to the mock upstream
directly. Overhead = gateway percentile - direct percentile at the same
concurrency.

    python bench/loadtest.py --out bench/results/loadtest.json

Everything runs on one machine, so the load generator, the gateway and the mock
compete for the same CPUs; see the README for the conditions of each run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import httpx
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "lg-bench-key-000001"

TOPICS = [
    "sorting algorithms",
    "the French revolution",
    "photosynthesis",
    "TCP congestion control",
    "sourdough",
    "black holes",
    "tax brackets",
    "the Krebs cycle",
    "Kubernetes operators",
    "Roman aqueducts",
]


def prompt(i: int, unique: bool) -> str:
    t = TOPICS[i % len(TOPICS)]
    base = f"Give me a short, friendly explanation of {t} for a curious teenager."
    return f"{base} (request {i})" if unique else base


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return int(p)


def wait_ready(url: str, timeout: float = 120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{url} did not become ready")


def gateway_config(mock_url: str, db: str, guard: bool, router: bool, cache: bool) -> Dict[str, Any]:
    return {
        "server": {"admin_key": "bench-admin", "log_level": "warning"},
        "database_url": f"sqlite:///{db}",
        "providers": [
            {
                "name": "mock",
                "type": "openai",
                "base_url": mock_url + "/v1",
                "api_key": "x",
                "timeout_s": 30,
                "max_retries": 0,
            }
        ],
        "models": [
            {"name": "strong", "provider": "mock", "input_per_1m": 2.5, "output_per_1m": 10},
            {"name": "weak", "provider": "mock", "input_per_1m": 0.15, "output_per_1m": 0.6},
        ],
        "routing": {
            "costroute": {
                "enabled": router,
                "alias": "auto",
                "router": "heuristic",
                "threshold": 0.5,
                "strong": "strong",
                "weak": "weak",
            }
        },
        "guard": {"enabled": guard, "input": "block", "output": "redact"},
        "cache": {
            "exact": {"enabled": cache, "backend": "memory"},
            "semantic": {"enabled": cache, "backend": "memory", "embedder": "hashing", "threshold": 0.95},
        },
        "tenancy": {"bootstrap": [{"tenant": "bench", "api_key": KEY}], "budget_refresh_s": 3600},
    }


async def run_load(
    url: str, headers: Dict[str, str], model: str, concurrency: int, total: int, unique: bool, offset: int
) -> Dict[str, Any]:
    lat: List[float] = []
    errors = 0
    counter = iter(range(total))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=60, trust_env=False) as client:

        async def worker() -> None:
            nonlocal errors
            for i in counter:
                body = {"model": model, "messages": [{"role": "user", "content": prompt(offset + i, unique)}]}
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=body, headers=headers)
                    ok = r.status_code == 200
                except httpx.HTTPError:
                    ok = False
                dt = time.perf_counter() - t0
                if ok:
                    lat.append(dt)
                else:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    lat.sort()

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))] * 1000 if lat else float("nan")

    return {
        "n": len(lat),
        "errors": errors,
        "rps": round(len(lat) / wall, 1),
        "p50_ms": round(pct(50), 2),
        "p95_ms": round(pct(95), 2),
        "p99_ms": round(pct(99), 2),
        "mean_ms": round(statistics.fmean(lat) * 1000, 2) if lat else None,
    }


def start(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    e = {**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", **(env or {})}
    return subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)


def stop(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(10)
    except Exception:
        p.kill()


def server_overhead(metrics_text: str) -> Optional[float]:
    s = c = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("llmgate_gateway_overhead_seconds_sum"):
            s += float(line.split()[-1])
        elif line.startswith("llmgate_gateway_overhead_seconds_count"):
            c += float(line.split()[-1])
    return round(s / c * 1000, 3) if c else None


VARIANTS = {
    # name: (guard, router, cache, model, unique prompts)
    "passthrough": (False, False, False, "weak", True),
    "guard": (True, False, False, "weak", True),
    "router": (False, True, False, "auto", True),
    "cache_miss": (False, False, True, "weak", True),
    "all_on_miss": (True, True, True, "auto", True),
    "cache_hit": (False, False, True, "weak", False),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-ms", type=float, default=50.0)
    ap.add_argument("--concurrency", default="1,10,50")
    ap.add_argument("--requests", default="200,600,1500", help="requests per concurrency level")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "loadtest.json"))
    args = ap.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]
    totals = [int(x) for x in args.requests.split(",")]
    py = sys.executable

    mock_port = free_port()
    mock = start([py, "-m", "llmgate", "mock-upstream", "--port", str(mock_port), "--latency-ms", str(args.latency_ms)])
    mock_url = f"http://127.0.0.1:{mock_port}"
    results: Dict[str, Any] = {
        "conditions": {
            "mock_latency_ms": args.latency_ms,
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "note": "gateway, mock upstream and load generator share the same host; 1 uvicorn worker each",
        },
        "direct": {},
        "gateway": {},
    }
    try:
        wait_ready(mock_url + "/healthz")
        asyncio.run(run_load(mock_url + "/v1/chat/completions", {}, "weak", 10, args.warmup, True, 0))
        for c, n in zip(levels, totals):
            results["direct"][str(c)] = asyncio.run(
                run_load(mock_url + "/v1/chat/completions", {}, "weak", c, n, True, 10_000)
            )
            print("direct", c, results["direct"][str(c)], flush=True)
        for name in args.variants.split(","):
            guard, router, cache, model, unique = VARIANTS[name]
            tmp = tempfile.mkdtemp(prefix="llmgate-bench-")
            cfg_path = os.path.join(tmp, "config.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(gateway_config(mock_url, os.path.join(tmp, "bench.db"), guard, router, cache), f)
            gport = free_port()
            gw = start(
                [py, "-m", "llmgate", "serve", "--config", cfg_path, "--port", str(gport), "--host", "127.0.0.1"]
            )
            gurl = f"http://127.0.0.1:{gport}"
            try:
                wait_ready(gurl + "/healthz")
                h = {"authorization": f"Bearer {KEY}"}
                asyncio.run(run_load(gurl + "/v1/chat/completions", h, model, 10, args.warmup, unique, 0))
                results["gateway"][name] = {}
                for i, (c, n) in enumerate(zip(levels, totals)):
                    r = asyncio.run(run_load(gurl + "/v1/chat/completions", h, model, c, n, unique, 100_000 * (i + 1)))
                    d = results["direct"][str(c)]
                    r["overhead_p50_ms"] = round(r["p50_ms"] - d["p50_ms"], 2)
                    r["overhead_p95_ms"] = round(r["p95_ms"] - d["p95_ms"], 2)
                    results["gateway"][name][str(c)] = r
                    print(name, c, r, flush=True)
                mt = httpx.get(gurl + "/metrics", trust_env=False).text
                results["gateway"][name]["server_side_overhead_mean_ms"] = server_overhead(mt)
            finally:
                stop(gw)
    finally:
        stop(mock)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
