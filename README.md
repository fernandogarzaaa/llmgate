# llmgate

An OpenAI-compatible LLM gateway: one endpoint in front of OpenAI, Anthropic and local
(llama.cpp / vLLM / Ollama) models, with per-tenant keys, rate limits and budgets, cost
accounting, guardrails, cost-aware routing, fallbacks and circuit breakers, exact and semantic
caching, Prometheus metrics and OpenTelemetry tracing.

## The problem

Once more than one team or product calls LLM APIs, the same plumbing gets rebuilt in every
service: provider keys copied around, no per-team spend limit, nobody able to answer "who spent
what on which model", retries and fallbacks written five different ways, prompt-injection and PII
checks applied inconsistently, and identical (or nearly identical) questions paid for again and
again. A gateway centralises that. Clients keep using the OpenAI SDK and point `base_url` at
llmgate; the gateway authenticates the tenant, enforces limits and budgets, scans input and
output, picks the model, serves from cache when it safely can, calls the upstream with retries and
fallbacks, and records tokens and cost per tenant and model.

## Quickstart (no API keys, mock upstream)

Everything below runs against the built-in mock upstream (`llmgate mock-upstream`), which answers
the OpenAI and Anthropic APIs with canned responses after a fixed delay. No keys are needed.
Requires Python 3.10+ and `curl`.

```bash
git clone https://github.com/fernandogarzaaa/llmgate.git
cd llmgate
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
# If your shell has an HTTP proxy configured, keep local calls off it:
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
```

### CLI and `serve`

```bash
llmgate check-config -c examples/config.mock.yaml

llmgate mock-upstream --port 9000 > mock.log 2>&1 &
MOCK_PID=$!
llmgate serve -c examples/config.mock.yaml > gateway.log 2>&1 &
GW_PID=$!
until curl -sf localhost:8080/healthz > /dev/null; do sleep 1; done
curl -s localhost:8080/healthz; echo

# model "auto" lets the cost router choose between gpt-4o and gpt-4o-mini (both mocked)
curl -s -D - localhost:8080/v1/chat/completions \
  -H "authorization: Bearer lg-demo-key-change-me" -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is a token bucket?"}]}'; echo

# the same request again is served from the exact cache (x-llmgate-cache: exact)
curl -s -D - -o /dev/null localhost:8080/v1/chat/completions \
  -H "authorization: Bearer lg-demo-key-change-me" -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is a token bucket?"}]}' | grep -i x-llmgate

# tenants, keys and spend
llmgate keys create --tenant acme --create-tenant -c examples/config.mock.yaml
llmgate usage -c examples/config.mock.yaml --by tenant,model
curl -s localhost:8080/metrics | grep '^llmgate_requests' | head

kill $GW_PID $MOCK_PID
```

The demo tenant key `lg-demo-key-change-me` and the admin key `dev-admin-key` come from
`examples/config.mock.yaml`; spend is written to `llmgate-demo.db` (SQLite) in the current
directory. The OpenAI Python SDK works unchanged:
`OpenAI(base_url="http://localhost:8080/v1", api_key="lg-demo-key-change-me")`.

Guardrails and the full cost router come from two optional libraries, `injectguard` and
`costroute`, which are not on PyPI yet. Without them `/healthz` reports guardrails as disabled
and routing uses a vendored heuristic, so a prompt-injection request is forwarded instead of
being blocked with `guardrail_input_blocked`. With them installed (`pip install ./path/to/injectguard
./path/to/costroute`) the same config blocks it.

`python scripts/smoke_test.py` runs the same flow (plus Anthropic-format streaming, the guard
check when injectguard is installed, metrics and usage) on throwaway ports and fails on the first
broken check.

### docker-compose (gateway + Redis + Postgres + Prometheus + Grafana + mock)

```bash
docker compose up --build
# gateway http://localhost:8080, Prometheus http://localhost:9090, Grafana http://localhost:3000
curl localhost:8080/v1/chat/completions -H "authorization: Bearer lg-demo-key-change-me" \
  -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
docker compose down -v
```

Not run by me: the build host had no Docker. See [What is untested](#limitations-and-what-is-untested).

### Helm

```bash
helm lint deploy/helm/llmgate
helm template llmgate deploy/helm/llmgate --set serviceMonitor.enabled=true > /tmp/llmgate.yaml
# on a real cluster, with an image you built and pushed yourself:
# docker build -t <registry>/llmgate:0.1.0 . && docker push <registry>/llmgate:0.1.0
# helm install llmgate deploy/helm/llmgate --set image.repository=<registry>/llmgate --set image.tag=0.1.0 \
#   --set secrets.LLMGATE_ADMIN_KEY=$(openssl rand -hex 16)
```

`helm lint`, `helm template`, kubeconform on the rendered manifests (6 valid; the ServiceMonitor is
skipped for lack of a CRD schema) and `terraform validate` on `deploy/terraform/examples/kubernetes`
pass. Nothing has been installed on a cluster, and no image
is published (`ghcr.io/fernandogarzaaa/llmgate` in `values.yaml` does not exist yet).

## Architecture

```mermaid
flowchart LR
    C[Client<br/>OpenAI SDK] -->|POST /v1/chat/completions| A[Auth<br/>hashed virtual keys]
    A --> RL[Rate limit<br/>RPM/TPM token buckets<br/>memory or Redis Lua]
    RL --> B[Budget check<br/>daily / monthly / total]
    B --> GI[Input guard<br/>injectguard: block / flag]
    GI --> R[Router<br/>rules + costroute<br/>strong vs weak]
    R --> EC{Exact cache<br/>memory / Redis}
    EC -->|hit| RESP[Response]
    EC -->|miss| SC{Semantic cache<br/>embeddings + cosine}
    SC -->|hit| RESP
    SC -->|miss| D[Dispatcher<br/>timeouts, retries, Retry-After,<br/>circuit breakers, fallbacks]
    D --> P1[OpenAI-compatible]
    D --> P2[Anthropic Messages]
    D --> P3[Local: llama.cpp / vLLM / Ollama]
    D --> GO[Output guard<br/>redact / block / flag]
    GO --> ST[Cache store]
    ST --> RESP
    GO --> ACC[(Usage & cost<br/>SQLite / Postgres)]
    A -.-> M[Prometheus /metrics<br/>OpenTelemetry spans]
```

Source map: `app.py` (FastAPI pipeline), `gateway.py` (wiring), `ratelimit.py`, `budget.py`,
`cost.py`, `guard.py`, `routing.py` + `costroute_adapter.py`, `cache.py` + `embeddings.py`,
`dispatch.py` + `resilience.py`, `providers/`, `store.py`, `metrics.py`, `tracing.py`, `cli.py`,
`mock_upstream.py`.

## Measured results

All numbers come from `bench/results/*.json` (rendered into
[bench/results/RESULTS.md](bench/results/RESULTS.md) by `bench/render_results.py`).

### Gateway overhead (load test)

`python bench/loadtest.py` starts a real mock upstream with a fixed 50 ms latency and one real
gateway process per configuration, then drives it with a closed-loop asyncio client at
concurrency 1 and 10 and compares against calling the mock directly.

Conditions, and why the absolute numbers should be read loosely:

- One 2-vCPU aarch64 box, Python 3.12, one uvicorn worker. The load generator, the gateway and
  the mock share those two CPUs.
- The host was contended by other jobs: 1-minute load average 6.06 at the start, 7.64 at the end,
  6.2 to 7.7 during the runs. On 2 vCPUs that is 3 to 4 runnable processes per core.
- c=50 was dropped. In an earlier run on the same host it was saturated (direct-to-mock throughput
  fell from 161 req/s at c=10 to 93 req/s at c=50, p95 above 1.3 s for every configuration), so it
  measured the box, not the gateway.
- Cache-miss variants use prompts built from seeded random tokens. The highest hashing-embedder
  similarity between any two of them is 0.2346, far below the 0.95 semantic threshold, and the
  gateway reported 0 cache hits in every miss run. (The previous version appended "(request i)"
  to one sentence; those prompts scored about 0.956 against each other, so the "miss" runs were
  partly served from cache.)
- Guard and router rows ran with `injectguard` (rules + PII layers, no classifier) and `costroute`
  (heuristic router) installed.

| configuration | c=1 p50 / p95 (ms) | c=10 p50 / p95 (ms) | c=1 overhead p50 / p95 | c=10 overhead p50 / p95 | req/s at c=10 | server-side mean overhead (ms) |
|---|---|---|---|---|---|---|
| direct to mock (baseline) | 54.6 / 60.5 | 58.1 / 73.7 | 0 / 0 | 0 / 0 | 165.4 | n/a |
| auth + rate limit + accounting | 59.0 / 64.9 | 64.2 / 88.2 | +4.4 / +4.4 | +6.1 / +14.5 | 146.1 | 1.73 |
| + guardrails | 59.6 / 66.0 | 72.6 / 113.6 | +5.0 / +5.4 | +14.5 / +39.9 | 127.5 | 1.95 |
| + router (`model: auto`) | 60.1 / 66.2 | 78.8 / 109.4 | +5.5 / +5.7 | +20.7 / +35.7 | 121.3 | 5.25 |
| + exact & semantic cache, all misses | 60.7 / 69.3 | 141.8 / 285.8 | +6.2 / +8.8 | +83.7 / +212.1 | 64.5 | 5.47 |
| guard + router + caches, all misses | 63.2 / 69.0 | 158.0 / 316.2 | +8.6 / +8.5 | +99.8 / +242.5 | 59.4 | 18.36 |
| exact-cache hits | 3.4 / 7.3 | 25.4 / 71.7 | -51.2 / -53.2 | -32.8 / -1.9 | 312.8 | 2.97 |

0 failed requests across all runs.

What this says, with the caveats above:

- At c=1 the gateway adds roughly 4 to 9 ms at p50 in every configuration.
- At c=10 the caches on the miss path are the expensive part: throughput halves (146 to 64
  req/s) and p50 overhead goes from +6 ms to +84 ms, while the gateway's own server-side overhead
  histogram reports only about 5.5 ms. The time is being lost outside the measured section.
  One diagnostic re-run of the cache-miss variant with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`
  (`bench/results/diag_cache_miss_openblas1.json`, load average about 9.5) gave +23.7 / +48.5 ms
  overhead and 107 req/s at c=10, which points at NumPy's BLAS thread pool fighting for the two
  contended cores. That is one run, not a fix; setting those variables for the gateway process is
  the first thing to try, and the cause is not confirmed.
- Exact-cache hits skip the upstream entirely, so they come back faster than the upstream itself.

### Semantic cache quality (QQP)

`bench/semantic_cache_eval.py` uses the GLUE QQP validation set (first 4000 pairs, shuffled with
seed 13, split into dev and test halves of 2000). For each pair, the first question is stored and
the second is looked up; a hit on a pair labelled duplicate is a correct hit, a hit on a
non-duplicate is a wrong answer served from cache. 35.2% of test pairs are duplicates.

| embedder | ms / text | threshold | hit rate | correct hits | false hits (of all lookups) | hit precision | duplicates caught |
|---|---|---|---|---|---|---|---|
| hashing (built-in, no deps) | 0.20 | 0.965 | 0.5% | 0.4% | 0.1% | 81.8% | 1.3% |
| fastembed `BAAI/bge-small-en-v1.5` | 66.21 | 0.970 | 6.7% | 5.9% | 0.8% | 88.1% | 16.7% |
| fastembed `BAAI/bge-small-en-v1.5` | 66.21 | 0.98 | 4.4% | 4.0% | 0.45% | 89.8% | 11.2% |

The first two rows use the threshold picked on the dev half (lowest threshold with at least 90%
hit precision); the third is the test-half sweep at 0.98. The full sweep from 0.80 to 0.99 is in
RESULTS.md.

Caveats:

- **QQP negatives are close negatives.** QQP pairs were collected as candidate duplicates, so
  most non-duplicate pairs are on the same topic and share much of their wording. That makes
  false hits more likely than on typical traffic, and the 35% duplicate share is not a real
  cache-hit rate either. Treat the precision numbers as a stress test, not a
  forecast.
- **Why 0.98 in the production example.** On the dev half 0.97 was the lowest threshold that
  reached 90% hit precision, but on the test half it reached only 88.1%. 0.98 gives 89.8% on test
  and halves the false-hit rate (0.8% to 0.45% of lookups) at the cost of catching 11.2% of
  duplicates instead of 16.7%. `examples/config.production.yaml` ships 0.98 with fastembed for
  that reason. Nobody should take it as a universal value: tune it on your own traffic.
- **The hashing embedder is for demos.** It is free and fast but nearly useless as a semantic
  cache on QQP (1.3% of duplicates caught), and at the quickstart's 0.95 its test-half precision is
  66.7%. The mock config uses it only because it needs no model download.

Dollar estimate. Assumptions: every request is 500 prompt + 250 completion tokens; list prices in
USD per 1M input / output tokens of gpt-4o-mini $0.15 / $0.60, gpt-4o $2.50 / $10.00,
claude-sonnet $3.00 / $15.00; every hit (correct or not) saves one full request; traffic looks like
QQP. Embedding cost and latency are not subtracted.

| embedder @ threshold | model | saved per 1M requests | wrong answers served per 1M requests |
|---|---|---|---|
| hashing @ 0.965 | gpt-4o-mini / gpt-4o / claude-sonnet | $1 / $21 / $29 | 1,000 |
| bge-small @ 0.97 | gpt-4o-mini / gpt-4o / claude-sonnet | $15 / $251 / $352 | 8,000 |
| bge-small @ 0.98 | gpt-4o-mini / gpt-4o / claude-sonnet | $10 / $165 / $231 | 4,500 |

The savings are small and the wrong answers are not: at 0.98, every million requests saves about
$165 on gpt-4o and returns about 4,500 cached answers to questions that were not the same question.
Whether that trade is acceptable depends on the application; for most it argues for the exact
cache only, or a semantic cache scoped narrowly (per tenant, per prompt template) with a high
threshold.

### Test quality (mutation testing)

[testteeth](https://pypi.org/project/testteeth/) 0.2.0 mutates the dependency-light core
(`ratelimit.py`, `budget.py`, `cost.py`, `cache.py`, `resilience.py`, `routing.py`) and runs
`tests/unit` against each mutant.

| run | mutants | killed | survived | not covered | timeouts | score |
|---|---|---|---|---|---|---|
| before (initial test suite) | 543 | 230 | 143 | 170 | 0 | 42.4% |
| after (current code) | 557 | 536 | 15 | 6 | 1 | 96.2% |

Full reports: [docs/mutation/baseline.txt](docs/mutation/baseline.txt) and
[docs/mutation/after.txt](docs/mutation/after.txt). The "after" run was taken on the final source,
including the in-place semantic-cache matrix (which added mutants; on the code before that change
the same tests scored 532 of 537, 99.1%). Half of the 14 survivors listed in the report and all 6
uncovered mutants are in that new `_Namespace` code; the rest are default constants in the Redis
rate limiter, the circuit breaker and the routing module. `tests/unit/test_cache_namespace.py` was added afterwards to cover the matrix growth,
eviction and expiry paths against a brute-force reference; it passes, but the score above has not
been re-measured with it.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q -m "not slow"   # unit, component and integration tests
python -m pytest -q -m slow         # end-to-end: real mock + gateway processes
ruff check . && ruff format --check src tests bench && mypy
testteeth run --workers 2           # mutation testing
python bench/loadtest.py && python bench/render_results.py > bench/results/RESULTS.md
```

Redis and Postgres tests run when `LLMGATE_TEST_REDIS_URL` / `LLMGATE_TEST_DATABASE_URL` are set;
Redis rate limiting is otherwise exercised through `fakeredis` with Lua. The guardrail tests and the
costroute and injectguard adapter tests are skipped when those libraries are not installed; a
separate test then checks that the gateway runs with guardrails disabled.

## Limitations and what is untested

- **Docker image and docker-compose:** the Dockerfile and compose stack have never been built or
  started. The build host had no Docker.
- **Cluster deploy:** the Helm chart and Terraform module are linted, rendered and validated only.
  Nothing has been installed on Kubernetes, and no image is published.
- **GitHub workflows:** `.github/workflows/ci.yml` and `release.yml` have never run. The repository
  is not on GitHub yet.
- **Private integrations in CI:** `costroute` and `injectguard` are installed in CI from git URLs
  with `continue-on-error`. While those repositories are private that step will fail quietly and
  the integration-specific tests will be skipped, not run.
- **Postgres:** the Postgres store test is skipped locally (no `LLMGATE_TEST_DATABASE_URL`); only
  SQLite has been exercised. Redis is exercised through fakeredis, not a real server.
- **Real providers:** every test and benchmark used the mock upstream. The OpenAI, Anthropic and
  local adapters are tested against mocked responses, not live APIs.
- **Benchmarks:** one contended 2-vCPU host, no c=50 data, a single run per configuration, and an
  unexplained slowdown in the semantic-cache miss path at c=10 (see above). The semantic-cache
  numbers come from one dataset with deliberately hard negatives.
- **Semantic cache scale:** brute-force cosine search per namespace (default cap 5,000 entries);
  there is no ANN index.

## License

MIT. See [LICENSE](LICENSE).
