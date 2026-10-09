## Load test

_loadtest.json not found_

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

