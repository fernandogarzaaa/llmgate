"""Anthropic Messages API, translated to and from the OpenAI chat format."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .base import (
    SSE_DONE,
    Provider,
    ProviderError,
    SSEParser,
    error_from_exception,
    error_from_response,
    sse,
)

STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
DEFAULT_MAX_TOKENS = 1024


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def to_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI chat request -> Anthropic Messages request.

    System messages are concatenated into ``system``; consecutive messages with
    the same role are merged (Anthropic requires alternation); ``max_tokens``
    defaults to 1024 because Anthropic requires it.
    """
    system: List[str] = []
    msgs: List[Dict[str, Any]] = []
    for m in body.get("messages", []):
        role = m.get("role")
        text = _text(m.get("content"))
        if role in ("system", "developer"):
            system.append(text)
            continue
        role = "assistant" if role == "assistant" else "user"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": role, "content": text})
    out: Dict[str, Any] = {
        "model": body["model"],
        "messages": msgs,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS),
    }
    if system:
        out["system"] = "\n\n".join(system)
    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            out[k] = body[k]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return out


def from_anthropic(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Anthropic Messages response -> OpenAI chat completion."""
    usage = data.get("usage") or {}
    pt, ct = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return {
        "id": "chatcmpl-" + str(data.get("id", uuid.uuid4().hex)),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": _text(data.get("content"))},
            "finish_reason": STOP_REASONS.get(str(data.get("stop_reason")), "stop"),
        }],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


class StreamTranslator:
    """Anthropic stream events -> OpenAI ``chat.completion.chunk`` SSE bytes."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish: Optional[str] = None

    def _chunk(self, delta: Dict[str, Any], finish: Optional[str] = None) -> bytes:
        return sse({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def handle(self, event: str, data: str) -> List[bytes]:
        try:
            ev = json.loads(data)
        except ValueError:
            return []
        typ = ev.get("type", event)
        if typ == "message_start":
            self.prompt_tokens = int(((ev.get("message") or {}).get("usage") or {}).get("input_tokens", 0))
            return [self._chunk({"role": "assistant", "content": ""})]
        if typ == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                return [self._chunk({"content": delta.get("text", "")})]
            return []
        if typ == "message_delta":
            self.completion_tokens = int((ev.get("usage") or {}).get("output_tokens", self.completion_tokens))
            self.finish = STOP_REASONS.get(str((ev.get("delta") or {}).get("stop_reason")), "stop")
            return []
        if typ == "message_stop":
            usage = {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                     "total_tokens": self.prompt_tokens + self.completion_tokens}
            final = sse({"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                         "model": self.model, "choices": [], "usage": usage})
            return [self._chunk({}, self.finish or "stop"), final, SSE_DONE]
        if typ == "error":
            err = ev.get("error") or {}
            return [sse({"error": {"message": err.get("message", "upstream error"),
                                   "type": err.get("type", "upstream_error")}}), SSE_DONE]
        return []


class AnthropicProvider(Provider):
    type = "anthropic"

    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json", "anthropic-version": self.cfg.anthropic_version, **self.cfg.headers}
        if self.cfg.api_key:
            h["x-api-key"] = self.cfg.api_key
        return h

    def _url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/messages"

    async def complete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = await self.client.post(self._url(), json=to_anthropic(body), headers=self._headers(),
                                       timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise error_from_exception(exc) from None
        if r.status_code >= 400:
            raise error_from_response(r.status_code, r.content, r.headers)
        try:
            data = r.json()
        except ValueError:
            raise ProviderError(502, "upstream returned invalid JSON", True) from None
        return from_anthropic(data, body["model"])

    async def open_stream(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        payload = {**to_anthropic(body), "stream": True}
        req = self.client.build_request("POST", self._url(), json=payload, headers=self._headers(),
                                        timeout=self.timeout)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise error_from_exception(exc) from None
        if resp.status_code >= 400:
            content = await resp.aread()
            await resp.aclose()
            raise error_from_response(resp.status_code, content, resp.headers)
        return self._translate(resp, body["model"])

    async def _translate(self, resp: httpx.Response, model: str) -> AsyncIterator[bytes]:
        tr, parser = StreamTranslator(model), SSEParser()
        try:
            async for chunk in resp.aiter_raw():
                for event, data in parser.feed(chunk):
                    for out in tr.handle(event, data):
                        yield out
        finally:
            await resp.aclose()


def split_events(raw: bytes) -> List[Tuple[str, str]]:
    return SSEParser().feed(raw)
