"""OpenTelemetry tracing.

``opentelemetry-api`` is a core dependency, so spans are always created; without
the SDK (``pip install 'llmgate[otel]'``) they are no-ops. Incoming W3C
``traceparent`` headers are honoured so the gateway joins the caller's trace.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from .config import TracingConfig

log = logging.getLogger("llmgate.tracing")
tracer = trace.get_tracer("llmgate")


def setup_tracing(cfg: TracingConfig) -> bool:
    """Install an SDK tracer provider if tracing is enabled and the SDK is present."""
    if not cfg.enabled or cfg.exporter == "none":
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        log.warning("tracing.enabled but opentelemetry-sdk is missing: pip install 'llmgate[otel]'")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
    exporter: Any
    if cfg.exporter == "console":
        exporter = ConsoleSpanExporter()
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("OTLP exporter missing: pip install 'llmgate[otel]'")
            return False
        exporter = OTLPSpanExporter(endpoint=cfg.endpoint) if cfg.endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    global tracer
    tracer = trace.get_tracer("llmgate")
    return True


def extract(headers: Mapping[str, str]) -> otel_context.Context:
    return propagate.extract(dict(headers))


@contextmanager
def span(name: str, ctx: Optional[otel_context.Context] = None, **attrs: Any) -> Iterator[Any]:
    with tracer.start_as_current_span(name, context=ctx) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s
