"""A mock LLM upstream with fixed latency, for demos, tests and load tests.

Serves both wire formats the gateway speaks:

* ``POST /v1/chat/completions`` (OpenAI-compatible, JSON or SSE)
* ``POST /v1/messages`` (Anthropic Messages, JSON or SSE)
* ``POST /v1/embeddings`` (deterministic hashing embeddings)

The answer is ``"[<name>:<model>] <last user message>"``. Latency is fixed
(``--latency-ms``); failures can be injected with ``--fail-rate`` or per request
with the header ``x-mock-fail: 500|429|timeout``. Not a real model.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .embeddings import HashingEmbedder


def _last_user(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c)
    return ""


def _tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def create_mock_app(name: str = "mock", latency_ms: float = 50.0, fail_rate: float = 0.0,
                    stream_chunks: int = 4, seed: int = 0) -> FastAPI:
    app = FastAPI(title=f"llmgate mock upstream ({name})")
    rng = random.Random(seed)
    emb = HashingEmbedder(256)
    stats = {"requests": 0}

    async def _gate(request: Request) -> Response | None:
        stats["requests"] += 1
        await asyncio.sleep(latency_ms / 1000.0)
        fail = request.headers.get("x-mock-fail", "")
        if not fail and fail_rate > 0 and rng.random() < fail_rate:
            fail = "500"
        if fail == "timeout":
            await asyncio.sleep(3600)
        if fail.isdigit():
            code = int(fail)
            return JSONResponse({"error": {"message": f"mock failure {code}", "type": "mock_error"}},
                                status_code=code, headers={"retry-after": "0"} if code == 429 else None)
        return None

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "name": name, "latency_ms": latency_ms, **stats}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.json()
        err = await _gate(request)
        if err is not None:
            return err
        model = str(body.get("model", name))
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        text = f"[{name}:{model}] {_last_user(body.get('messages', []))}"
        pt, ct = _tokens(prompt), _tokens(text)
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def gen() -> AsyncIterator[bytes]:
                n = max(1, stream_chunks)
                step = max(1, -(-len(text) // n))
                base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
                yield ("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                                   "finish_reason": None}]}) + "\n\n").encode()
                for i in range(0, len(text), step):
                    ch = {**base, "choices": [{"index": 0, "delta": {"content": text[i:i + step]},
                                               "finish_reason": None}]}
                    yield ("data: " + json.dumps(ch) + "\n\n").encode()
                yield ("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {},
                                                                   "finish_reason": "stop"}]}) + "\n\n").encode()
                if include_usage:
                    u = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                    yield ("data: " + json.dumps({**base, "choices": [], "usage": u}) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        })

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        body = await request.json()
        err = await _gate(request)
        if err is not None:
            return err
        model = str(body.get("model", name))
        text = f"[{name}:{model}] {_last_user(body.get('messages', []))}"
        prompt = (body.get("system") or "") + "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        pt, ct = _tokens(prompt), _tokens(text)
        mid = "msg_" + uuid.uuid4().hex[:12]
        if body.get("stream"):
            async def gen() -> AsyncIterator[bytes]:
                def ev(t: str, d: Dict[str, Any]) -> bytes:
                    return f"event: {t}\ndata: {json.dumps({'type': t, **d})}\n\n".encode()
                yield ev("message_start", {"message": {"id": mid, "model": model, "role": "assistant",
                                                       "usage": {"input_tokens": pt, "output_tokens": 0}}})
                yield ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
                half = len(text) // 2
                for piece in (text[:half], text[half:]):
                    yield ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": piece}})
                yield ev("content_block_stop", {"index": 0})
                yield ev("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": ct}})
                yield ev("message_stop", {})

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"id": mid, "type": "message", "role": "assistant", "model": model,
                             "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                             "usage": {"input_tokens": pt, "output_tokens": ct}})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        body = await request.json()
        inp = body.get("input", [])
        texts = [inp] if isinstance(inp, str) else list(inp)
        vecs = await emb(texts)
        return JSONResponse({"object": "list", "model": body.get("model", "mock-embed"),
                             "data": [{"object": "embedding", "index": i, "embedding": v.tolist()}
                                      for i, v in enumerate(vecs)]})

    return app
