# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Helm chart defaults to `replicaCount: 1` with autoscaling off, and the Terraform module defaults
  to `replicas_min = replicas_max = 1`, because budgets are tracked in memory per replica and can be
  overspent by up to ~replicas x one `budget_refresh_s` window when scaled out. Documented in the
  README ("Budgets and multiple replicas").

## [0.1.0] - 2026-10-09

### Added
- OpenAI-compatible `POST /v1/chat/completions` (JSON and SSE streaming) and `GET /v1/models`.
- Providers: OpenAI-compatible, Anthropic Messages (request/response and stream translation),
  local llama.cpp / vLLM / Ollama.
- Resilience: per-provider timeouts, retries with exponential backoff and `Retry-After`,
  circuit breakers, ordered model fallbacks (until the first streamed byte).
- Routing: rule-based routing (model, regex, prompt length, tenant, header) and cost-aware
  routing through costroute (vendored heuristic fallback when costroute is not installed).
- Guardrails through injectguard on input (block/flag) and output (redact/block/flag, buffered
  or post-scan for streams).
- Exact cache (memory or Redis) and semantic cache (memory or SQLite; hashing, fastembed or
  OpenAI-compatible embeddings).
- Multi-tenancy: hashed virtual API keys, per-tenant token-bucket RPM/TPM limits (memory or
  Redis Lua), budgets (daily/monthly/total), per-tenant and per-model cost attribution in
  SQLite or Postgres.
- Observability: Prometheus `/metrics`, OpenTelemetry tracing, Grafana dashboard,
  `/healthz` and `/readyz`.
- Admin CLI (`llmgate serve | tenants | keys | usage | mock-upstream | check-config`) and admin API.
- Deployment: Dockerfile (non-root, healthcheck), docker-compose stack, Helm chart, Terraform module.
- Benchmarks: load test against a fixed-latency mock upstream, semantic-cache evaluation on QQP.
