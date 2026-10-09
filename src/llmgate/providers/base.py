"""Provider interface and shared helpers."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..config import ProviderConfig
from ..resilience import CircuitBreaker, RetryPolicy, is_retryable_status


class ProviderError(Exception):
    """An upstream failure. ``retryable`` errors are retried and failed over."""

    def __init__(self, status: int, message: str, retryable: bool, kind: str = "http",
                 retry_after: Optional[str] = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retryable = retryable
        self.kind = kind
        self.retry_after = retry_after
        self.body = body


def error_from_response(status: int, content: bytes, headers: Optional[httpx.Headers] = None) -> ProviderError:
    try:
        body: Any = json.loads(content)
    except ValueError:
        body = {"error": {"message": content.decode("utf-8", "replace")[:500], "type": "upstream_error"}}
    msg = ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        msg = str(body["error"].get("message", ""))
    return ProviderError(status, msg or f"upstream returned HTTP {status}", is_retryable_status(status),
                         "http", headers.get("retry-after") if headers is not None else None, body)


def error_from_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(504, f"upstream timeout ({exc.__class__.__name__})", True, "timeout")
    return ProviderError(502, f"upstream connection error ({exc.__class__.__name__})", True, "connection")


class SSEParser:
    """Incremental parser for ``text/event-stream`` bytes; yields (event, data) pairs."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> List[Tuple[str, str]]:
        self._buf += chunk.replace(b"\r\n", b"\n")
        out: List[Tuple[str, str]] = []
        while b"\n\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n\n", 1)
            event, data = "message", []
            for line in raw.decode("utf-8", "replace").split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
            if data:
                out.append((event, "\n".join(data)))
        return out


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


class Provider:
    """Base class. Subclasses translate OpenAI chat requests to their wire format."""

    type = "base"

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.client = client
        self.timeout = httpx.Timeout(cfg.timeout_s, connect=cfg.connect_timeout_s)
        self.breaker = CircuitBreaker(cfg.breaker.failure_threshold, cfg.breaker.recovery_s,
                                      cfg.breaker.half_open_max)
        self.retry = RetryPolicy(cfg.max_retries, cfg.retry_base_s, cfg.retry_max_s)

    async def complete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def open_stream(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Connect and validate the status; raise :class:`ProviderError` before any byte is
        returned so the caller can still retry or fail over. Yields OpenAI-format SSE bytes."""
        raise NotImplementedError

    async def ping(self) -> bool:
        return True
