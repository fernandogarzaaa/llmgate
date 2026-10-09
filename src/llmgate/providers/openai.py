"""OpenAI-compatible upstreams: OpenAI, Azure-style proxies, vLLM, llama.cpp server, Ollama."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx

from .base import Provider, error_from_exception, error_from_response


class OpenAIProvider(Provider):
    type = "openai"

    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json", **self.cfg.headers}
        if self.cfg.api_key:
            h["authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/chat/completions"

    async def complete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload = {**body, "stream": False}
        payload.pop("stream_options", None)
        try:
            r = await self.client.post(self._url(), json=payload, headers=self._headers(), timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise error_from_exception(exc) from None
        if r.status_code >= 400:
            raise error_from_response(r.status_code, r.content, r.headers)
        try:
            data: Dict[str, Any] = r.json()
        except ValueError:
            raise error_from_response(502, b'{"error":{"message":"upstream returned invalid JSON"}}') from None
        return data

    def _stream_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload = {**body, "stream": True}
        opts = dict(payload.get("stream_options") or {})
        opts.setdefault("include_usage", True)  # needed for accurate token accounting
        payload["stream_options"] = opts
        return payload

    async def open_stream(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        req = self.client.build_request(
            "POST", self._url(), json=self._stream_payload(body), headers=self._headers(), timeout=self.timeout
        )
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise error_from_exception(exc) from None
        if resp.status_code >= 400:
            content = await resp.aread()
            await resp.aclose()
            raise error_from_response(resp.status_code, content, resp.headers)
        return self._relay(resp)

    async def _relay(self, resp: httpx.Response) -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await resp.aclose()


class LocalProvider(OpenAIProvider):
    """llama.cpp ``llama-server`` / vLLM / Ollama: OpenAI-compatible, usually no API key.

    Some local servers reject ``stream_options``; it is only sent when the request had it.
    """

    type = "local"

    def _stream_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {**body, "stream": True}
