"""Prometheus metrics. Each app instance owns its own registry (tests create many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, disable_created_metrics

disable_created_metrics()

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
OVERHEAD_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("llmgate_requests_total", "Chat requests handled",
                                ["tenant", "model", "status", "cache"], registry=r)
        self.latency = Histogram("llmgate_request_duration_seconds", "End-to-end request latency (to last byte)",
                                 ["model", "cache", "stream"], buckets=LATENCY_BUCKETS, registry=r)
        self.ttfb = Histogram("llmgate_time_to_first_byte_seconds", "Streaming: time to first upstream byte",
                              ["model"], buckets=LATENCY_BUCKETS, registry=r)
        self.upstream_latency = Histogram("llmgate_upstream_duration_seconds", "Upstream call latency per attempt",
                                          ["provider", "outcome"], buckets=LATENCY_BUCKETS, registry=r)
        self.overhead = Histogram("llmgate_gateway_overhead_seconds",
                                  "Time spent in the gateway itself (non-streaming: total minus upstream)",
                                  ["cache"], buckets=OVERHEAD_BUCKETS, registry=r)
        self.tokens = Counter("llmgate_tokens_total", "Tokens processed", ["tenant", "model", "type"], registry=r)
        self.cost = Counter("llmgate_cost_usd_total", "Upstream cost attributed to tenants (USD)",
                            ["tenant", "model"], registry=r)
        self.saved = Counter("llmgate_cost_saved_usd_total", "Cost avoided by cache hits (USD)",
                             ["tenant", "model"], registry=r)
        self.cache = Counter("llmgate_cache_requests_total", "Cache lookups", ["layer", "result"], registry=r)
        self.guard = Counter("llmgate_guard_events_total", "Guardrail detections",
                             ["direction", "action", "category"], registry=r)
        self.ratelimited = Counter("llmgate_ratelimit_rejections_total", "Requests rejected by rate limits",
                                   ["tenant", "limit"], registry=r)
        self.budget = Counter("llmgate_budget_rejections_total", "Requests rejected by budgets", ["tenant"],
                              registry=r)
        self.upstream_errors = Counter("llmgate_upstream_errors_total", "Failed upstream attempts",
                                       ["provider", "kind"], registry=r)
        self.fallbacks = Counter("llmgate_fallbacks_total", "Requests served by a fallback model",
                                 ["requested", "served"], registry=r)
        self.routes = Counter("llmgate_route_decisions_total", "Routing decisions", ["strategy", "model"],
                              registry=r)
        self.breaker = Gauge("llmgate_circuit_state", "Circuit breaker state (0 closed, 1 half-open, 2 open)",
                             ["provider"], registry=r)
        self.inflight = Gauge("llmgate_inflight_requests", "Requests in flight", registry=r)
