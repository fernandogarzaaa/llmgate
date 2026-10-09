## Load test

Mock upstream latency 50 ms; 2 vCPU aarch64, Python 3.12.15; gateway, mock upstream and load generator share the same host; 1 uvicorn worker each.

Host 1-minute load average: 6.06 at start, 7.64 at end, 6.2-7.69 across the runs (other jobs were running on the same 2 vCPUs, so absolute latencies are inflated and noisy).

Miss variants use prompts built from seeded random tokens; the highest hashing-embedder similarity between any two of them is 0.2346, far below the 0.95 semantic threshold.

| configuration | c=1 p50 / p95 (ms) | c=10 p50 / p95 (ms) |
|---|---|---|
| direct to mock (baseline) | 54.6 / 60.5 | 58.1 / 73.7 |
| auth + rate limit + accounting only | 59.0 / 64.9 | 64.2 / 88.2 |
| + guardrails (injectguard rules + PII) | 59.6 / 66.0 | 72.6 / 113.6 |
| + costroute router (`model: auto`) | 60.1 / 66.2 | 78.8 / 109.4 |
| + exact & semantic cache, all misses | 60.7 / 69.3 | 141.8 / 285.8 |
| guard + router + caches (misses) | 63.2 / 69.0 | 158.0 / 316.2 |
| exact-cache hits (upstream skipped) | 3.4 / 7.3 | 25.4 / 71.7 |

Gateway overhead = gateway percentile minus direct percentile at the same concurrency (negative = faster than calling the upstream):

| configuration | c=1 p50 / p95 overhead (ms) | c=10 p50 / p95 overhead (ms) | throughput at c=10 (req/s) | server-side mean overhead (ms) |
|---|---|---|---|---|
| direct to mock | 0 / 0 | 0 / 0 | 165.4 | n/a |
| auth + rate limit + accounting only | +4.4 / +4.4 | +6.1 / +14.5 | 146.1 | 1.73 |
| + guardrails (injectguard rules + PII) | +5.0 / +5.4 | +14.5 / +39.9 | 127.5 | 1.95 |
| + costroute router (`model: auto`) | +5.5 / +5.7 | +20.7 / +35.7 | 121.3 | 5.25 |
| + exact & semantic cache, all misses | +6.2 / +8.8 | +83.7 / +212.1 | 64.5 | 5.47 |
| guard + router + caches (misses) | +8.6 / +8.5 | +99.8 / +242.5 | 59.4 | 18.36 |
| exact-cache hits (upstream skipped) | -51.2 / -53.2 | -32.8 / -1.9 | 312.8 | 2.97 |

Failed requests across all gateway runs: 0.

Cache hits served during the measured runs: auth + rate limit + accounting only: 0, + guardrails (injectguard rules + PII): 0, + costroute router (`model: auto`): 0, + exact & semantic cache, all misses: 0, guard + router + caches (misses): 0, exact-cache hits (upstream skipped): 800.

## Semantic cache (QQP)

GLUE QQP validation (nyu-mll/glue, config qqp), first 4000 pairs, shuffled seed 13, split 50/50; dev n=2000, test n=2000 (35.2% of test pairs are labelled duplicates). Threshold chosen on dev as the lowest with hit precision >= 90%.

| embedder | ms / text | threshold | hit rate | correct hits | false hits (of all lookups) | hit precision | duplicates caught |
|---|---|---|---|---|---|---|---|
| hashing | 0.20 | 0.965 | 0.5% | 0.4% | 0.1% | 81.8% | 1.3% |
| fastembed:BAAI/bge-small-en-v1.5 | 66.21 | 0.970 | 6.7% | 5.9% | 0.8% | 88.1% | 16.7% |

Threshold sweep on the test half:

| embedder | threshold | hit rate | false hits | hit precision | duplicates caught |
|---|---|---|---|---|---|
| hashing | 0.80 | 11.6% | 6.5% | 44.2% | 14.5% |
| hashing | 0.85 | 8.0% | 4.7% | 40.9% | 9.2% |
| hashing | 0.90 | 4.3% | 2.6% | 39.5% | 4.8% |
| hashing | 0.95 | 0.9% | 0.3% | 66.7% | 1.7% |
| hashing | 0.97 | 0.5% | 0.1% | 90.0% | 1.3% |
| hashing | 0.98 | 0.4% | 0.1% | 88.9% | 1.1% |
| hashing | 0.99 | 0.4% | 0.1% | 88.9% | 1.1% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.80 | 59.1% | 28.6% | 51.5% | 86.4% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.85 | 44.5% | 17.8% | 60.0% | 75.7% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.90 | 29.6% | 8.6% | 70.8% | 59.4% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.95 | 12.5% | 2.4% | 81.2% | 28.8% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.97 | 6.7% | 0.8% | 88.1% | 16.7% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.98 | 4.4% | 0.4% | 89.8% | 11.2% |
| fastembed:BAAI/bge-small-en-v1.5 | 0.99 | 2.4% | 0.1% | 93.8% | 6.4% |

Savings at the chosen threshold, assuming 500 prompt + 250 completion tokens per request and list prices (USD per 1M input / output tokens):

| embedder | model (price) | cost / request | saved per 1M requests | wrong answers per 1M requests |
|---|---|---|---|---|
| hashing | gpt-4o-mini ($0.15 / $0.6) | $0.000225 | $1 | 1,000 |
| hashing | gpt-4o ($2.5 / $10.0) | $0.003750 | $21 | 1,000 |
| hashing | claude-sonnet ($3.0 / $15.0) | $0.005250 | $29 | 1,000 |
| fastembed:BAAI/bge-small-en-v1.5 | gpt-4o-mini ($0.15 / $0.6) | $0.000225 | $15 | 8,000 |
| fastembed:BAAI/bge-small-en-v1.5 | gpt-4o ($2.5 / $10.0) | $0.003750 | $251 | 8,000 |
| fastembed:BAAI/bge-small-en-v1.5 | claude-sonnet ($3.0 / $15.0) | $0.005250 | $352 | 8,000 |

The same estimate at threshold 0.98 (the value in examples/config.production.yaml), derived from the test-half sweep above (saved = hit rate x cost per request; wrong answers = false-hit rate x 1M):

| embedder | model | hit precision | saved per 1M requests | wrong answers per 1M requests |
|---|---|---|---|---|
| hashing | gpt-4o-mini | 88.9% | $1 | 500 |
| hashing | gpt-4o | 88.9% | $17 | 500 |
| hashing | claude-sonnet | 88.9% | $24 | 500 |
| fastembed:BAAI/bge-small-en-v1.5 | gpt-4o-mini | 89.8% | $10 | 4,500 |
| fastembed:BAAI/bge-small-en-v1.5 | gpt-4o | 89.8% | $165 | 4,500 |
| fastembed:BAAI/bge-small-en-v1.5 | claude-sonnet | 89.8% | $231 | 4,500 |

