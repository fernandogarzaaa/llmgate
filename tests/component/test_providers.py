import json

import httpx
import pytest

from llmgate.config import ProviderConfig
from llmgate.providers import make_provider
from llmgate.providers.anthropic import StreamTranslator, from_anthropic, to_anthropic
from llmgate.providers.base import ProviderError, SSEParser, error_from_response


def test_to_anthropic_system_merge_and_defaults():
    out = to_anthropic(
        {
            "model": "c",
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "a"},
                {"role": "user", "content": [{"type": "text", "text": "b"}]},
                {"role": "assistant", "content": "x"},
            ],
            "temperature": 0.2,
            "stop": "END",
        }
    )
    assert out["system"] == "be nice"
    assert out["messages"] == [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "x"}]
    assert out["max_tokens"] == 1024
    assert out["temperature"] == 0.2 and out["stop_sequences"] == ["END"]


def test_from_anthropic():
    r = from_anthropic(
        {
            "id": "m1",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
        "sonnet",
    )
    assert r["choices"][0]["message"]["content"] == "hi"
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_stream_translator_sequence():
    tr = StreamTranslator("m")
    out = []
    for ev in [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "llo"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]:
        out += tr.handle(ev["type"], json.dumps(ev))
    events = [b.decode()[6:].strip() for b in out]
    assert events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert "".join(p["choices"][0]["delta"].get("content", "") for p in parsed if p["choices"]) == "hello"
    assert parsed[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert parsed[-2]["choices"][0]["finish_reason"] == "stop"


def test_sse_parser_handles_split_chunks_and_crlf():
    p = SSEParser()
    assert p.feed(b'data: {"a"') == []
    assert p.feed(b": 1}\r\n\r\nevent: x\ndata: 2\n\n") == [("message", '{"a": 1}'), ("x", "2")]


def test_error_from_response_classification():
    e = error_from_response(429, b'{"error":{"message":"slow down"}}', httpx.Headers({"retry-after": "2"}))
    assert e.retryable and e.message == "slow down" and e.retry_after == "2"
    e = error_from_response(400, b"not json")
    assert not e.retryable and e.body["error"]["type"] == "upstream_error"


def _provider(type_, handler):
    cfg = ProviderConfig(name="p", type=type_, base_url="http://up/v1", api_key="secret")
    return make_provider(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_openai_provider_sends_auth_and_strips_stream():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": []})

    p = _provider("openai", handler)
    await p.complete({"model": "m", "messages": [], "stream_options": {"include_usage": True}})
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["stream"] is False and "stream_options" not in seen["body"]


async def test_openai_stream_requests_usage_but_local_does_not():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(
            200, stream=httpx.ByteStream(b"data: [DONE]\n\n"), headers={"content-type": "text/event-stream"}
        )

    for t in ("openai", "local"):
        it = await _provider(t, handler).open_stream({"model": "m", "messages": []})
        assert b"".join([c async for c in it]) == b"data: [DONE]\n\n"
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in bodies[1]


async def test_provider_maps_transport_errors():
    def boom(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(ProviderError) as ei:
        await _provider("openai", boom).complete({"model": "m", "messages": []})
    assert ei.value.kind == "connection" and ei.value.retryable

    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(ProviderError) as ei:
        await _provider("anthropic", slow).complete({"model": "m", "messages": []})
    assert ei.value.kind == "timeout" and ei.value.status == 504


async def test_stream_error_status_raised_before_first_byte():
    def handler(req):
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(ProviderError) as ei:
        await _provider("openai", handler).open_stream({"model": "m", "messages": []})
    assert ei.value.status == 503 and ei.value.retryable


async def test_anthropic_headers():
    seen = {}

    def handler(req):
        seen.update(req.headers)
        return httpx.Response(200, json={"content": [], "usage": {}})

    await _provider("anthropic", handler).complete({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert seen["x-api-key"] == "secret" and seen["anthropic-version"] == "2023-06-01"
