import json
from typing import Any, Dict, Optional

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llmgate.app import create_app
from llmgate.config import load_config
from llmgate.gateway import Gateway
from llmgate.mock_upstream import create_mock_app
from llmgate.store import Store

KEY = "lg-test-key-0001"


class HostRouter(httpx.AsyncBaseTransport):
    """Routes requests to in-process ASGI apps by hostname."""

    def __init__(self, apps: Dict[str, Any]):
        self.transports = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "timeout":
            raise httpx.ReadTimeout("simulated timeout", request=request)
        t = self.transports.get(request.url.host)
        if t is None:
            raise httpx.ConnectError(f"no route to {request.url.host}", request=request)
        return await t.handle_async_request(request)


def flaky_app(fail_times: int, status: int = 503) -> FastAPI:
    """OpenAI-compatible upstream that fails ``fail_times`` times, then answers."""
    app = FastAPI()
    app.state.calls = 0
    inner = create_mock_app("flaky", latency_ms=0)
    inner_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=inner), base_url="http://inner")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        app.state.calls += 1
        if app.state.calls <= fail_times:
            return JSONResponse({"error": {"message": "boom", "type": "server_error"}}, status_code=status)
        r = await inner_client.post("/v1/chat/completions", content=await request.body())
        if (await request.json()).get("stream"):
            from fastapi.responses import Response

            return Response(r.content, media_type="text/event-stream")
        return JSONResponse(r.json())

    return app


def base_config(**over: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "server": {"admin_key": "admin-secret"},
        "database_url": "sqlite://",
        "providers": [
            {
                "name": "primary",
                "type": "openai",
                "base_url": "http://primary/v1",
                "api_key": "k",
                "max_retries": 1,
                "retry_base_s": 0,
                "breaker": {"failure_threshold": 3, "recovery_s": 60},
            },
            {"name": "backup", "type": "openai", "base_url": "http://backup/v1", "max_retries": 0},
            {"name": "claude", "type": "anthropic", "base_url": "http://claude/v1", "api_key": "k"},
            {"name": "local", "type": "local", "base_url": "http://local/v1"},
        ],
        "models": [
            {"name": "big", "provider": "primary", "input_per_1m": 10, "output_per_1m": 30, "fallbacks": ["spare"]},
            {"name": "small", "provider": "primary", "input_per_1m": 1, "output_per_1m": 2},
            {"name": "spare", "provider": "backup", "input_per_1m": 5, "output_per_1m": 5},
            {
                "name": "sonnet",
                "provider": "claude",
                "upstream_model": "claude-x",
                "input_per_1m": 3,
                "output_per_1m": 15,
            },
            {"name": "llama", "provider": "local"},
        ],
        "routing": {
            "costroute": {
                "enabled": True,
                "alias": "auto",
                "router": "heuristic",
                "threshold": 0.5,
                "strong": "big",
                "weak": "small",
            },
            "rules": [{"name": "pin", "when": {"header": {"x-pin": "llama"}}, "route_to": "llama"}],
        },
        "guard": {"enabled": True, "input": "block", "output": "redact"},
        "cache": {"exact": {"enabled": True}, "semantic": {"enabled": False}},
        "tenancy": {"bootstrap": [{"tenant": "acme", "api_key": KEY}], "budget_refresh_s": 3600},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


class Harness:
    def __init__(self, cfg_dict: Dict[str, Any], apps: Optional[Dict[str, Any]] = None):
        apps = apps or {}
        self.upstreams = {
            "primary": apps.get("primary") or create_mock_app("primary", latency_ms=0),
            "backup": apps.get("backup") or create_mock_app("backup", latency_ms=0),
            "claude": apps.get("claude") or create_mock_app("claude", latency_ms=0),
            "local": apps.get("local") or create_mock_app("local", latency_ms=0),
        }
        self.cfg = load_config(data=cfg_dict, env={})
        http = httpx.AsyncClient(transport=HostRouter(self.upstreams))
        self.gw = Gateway(self.cfg, store=Store("sqlite://"), http_client=http)

        # no retry sleeps in tests
        async def nosleep(_s):
            return None

        self.gw.dispatcher._sleep = nosleep
        self.app = create_app(self.cfg, self.gw)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gw")

    async def chat(self, content="hello", model="small", key=KEY, headers=None, **body):
        h = {"authorization": f"Bearer {key}"} if key else {}
        h.update(headers or {})
        payload = {"model": model, "messages": [{"role": "user", "content": content}], **body}
        return await self.client.post("/v1/chat/completions", json=payload, headers=h)

    async def flush(self):
        await self.gw.writer.flush()


@pytest.fixture
async def make_harness():
    made = []

    async def factory(cfg: Optional[Dict[str, Any]] = None, apps=None, **over):
        h = Harness(cfg or base_config(**over), apps)
        await h.gw.start()
        made.append(h)
        return h

    yield factory
    for h in made:
        await h.gw.stop()
        await h.client.aclose()


@pytest.fixture
async def gw(make_harness):
    return await make_harness()


def sse_events(text: str):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            data = block[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out
