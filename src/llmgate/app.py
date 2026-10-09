"""FastAPI application: the OpenAI-compatible surface, health, metrics and the admin API."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .cache import cache_key, is_cacheable, semantic_parts
from .config import GatewayConfig
from .cost import estimate_tokens, message_text, usage_from_response
from .errors import GatewayError
from .gateway import AuthContext, Gateway
from .providers.base import SSE_DONE, ProviderError, SSEParser, sse
from .routing import RouteContext, RoutingError, parse_threshold
from .tracing import extract, span

log = logging.getLogger("llmgate")


def _bearer(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    if h[:7].lower() == "bearer ":
        return h[7:].strip() or None
    return request.headers.get("x-api-key") or None


def completion_text(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return content if isinstance(content, str) else ""


def _set_completion_text(payload: Dict[str, Any], text: str) -> None:
    try:
        payload["choices"][0]["message"]["content"] = text
    except (KeyError, IndexError, TypeError):
        pass


def synth_stream(payload: Dict[str, Any], include_usage: bool) -> List[bytes]:
    """Render a complete chat.completion as an OpenAI SSE stream (used for cache hits and buffered guard)."""
    base = {
        "id": payload.get("id", "chatcmpl-" + uuid.uuid4().hex),
        "object": "chat.completion.chunk",
        "created": payload.get("created", int(time.time())),
        "model": payload.get("model", ""),
    }
    finish = "stop"
    try:
        finish = payload["choices"][0].get("finish_reason") or "stop"
    except (KeyError, IndexError, TypeError):
        pass
    out = [
        sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
        sse({**base, "choices": [{"index": 0, "delta": {"content": completion_text(payload)}, "finish_reason": None}]}),
        sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}),
    ]
    if include_usage and isinstance(payload.get("usage"), dict):
        out.append(sse({**base, "choices": [], "usage": payload["usage"]}))
    out.append(SSE_DONE)
    return out


def create_app(cfg: GatewayConfig, gateway: Optional[Gateway] = None) -> FastAPI:
    gw = gateway or Gateway(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await gw.start()
        try:
            yield
        finally:
            await gw.stop()

    app = FastAPI(title="llmgate", version=__version__, lifespan=lifespan)
    app.state.gateway = gw
    m = gw.metrics

    @app.exception_handler(GatewayError)
    async def _gateway_error(_req: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    # ------------------------------------------------------------------ health
    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "components": gw.components()}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        r = await gw.readiness()
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(m.registry), media_type=CONTENT_TYPE_LATEST)

    async def _auth(request: Request) -> AuthContext:
        auth = await gw.authenticate(_bearer(request))
        if auth is None:
            raise GatewayError(
                401,
                "invalid or missing API key",
                "authentication_error",
                "invalid_api_key",
                {"www-authenticate": "Bearer"},
            )
        return auth

    @app.get("/v1/models")
    async def models(request: Request) -> Dict[str, Any]:
        auth = await _auth(request)
        allowed = auth.tenant.allowed_models
        ids = [n for n in gw.router.public_models() if allowed is None or n in allowed]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "llmgate"} for i in ids]}

    # -------------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        m.inflight.inc()
        try:
            with span("llmgate.request", extract(request.headers), **{"llmgate.request_id": rid}) as root:
                return await _chat(request, rid, t0, root)
        except GatewayError as e:
            e.headers.setdefault("x-request-id", rid)
            raise
        finally:
            m.inflight.dec()

    async def _chat(request: Request, rid: str, t0: float, root: Any) -> Response:
        clen = request.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > cfg.server.max_body_bytes:
            raise GatewayError(413, "request body too large", code="body_too_large")
        auth = await _auth(request)
        tenant = auth.tenant
        root.set_attribute("llmgate.tenant", tenant.name)
        try:
            body = json.loads(await request.body())
        except ValueError:
            raise GatewayError(400, "request body is not valid JSON", code="invalid_json") from None
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            raise GatewayError(400, "'messages' must be a non-empty list", code="invalid_messages")
        if not all(isinstance(x, dict) for x in body["messages"]):
            raise GatewayError(400, "each message must be an object", code="invalid_messages")
        requested = body.get("model") or cfg.routing.default_model or ""
        if not isinstance(requested, str) or not requested:
            raise GatewayError(400, "'model' is required", code="model_required")
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        prompt = message_text(body["messages"])

        if tenant.allowed_models is not None and requested not in tenant.allowed_models:
            raise GatewayError(
                403, f"model {requested!r} is not allowed for this key", "permission_error", "model_not_allowed"
            )

        status = gw.budgets.check(tenant.id, auth.budget)
        if not status.allowed:
            m.budget.labels(tenant.name).inc()
            raise GatewayError(
                402,
                f"budget exhausted: spent ${status.spent_usd:.4f} of ${status.limit_usd:.2f} ({tenant.budget_period})",
                "insufficient_quota",
                "budget_exceeded",
            )

        limits = gw.limits_for(tenant)
        est = estimate_tokens(prompt)
        decision, which = await gw.limiter.check(tenant.id, limits, est)
        if not decision.allowed:
            m.ratelimited.labels(tenant.name, which).inc()
            retry = "60" if math.isinf(decision.retry_after_s) else str(max(1, math.ceil(decision.retry_after_s)))
            raise GatewayError(
                429, f"rate limit exceeded ({which})", "rate_limit_error", f"rate_limit_{which}", {"retry-after": retry}
            )

        headers: Dict[str, str] = {"x-request-id": rid}
        # ---- input guardrails
        if gw.guard is not None and cfg.guard.input != "off":
            with span("llmgate.guard.input"):
                g = (
                    await asyncio.to_thread(gw.guard.scan_input, prompt)
                    if cfg.guard.classifier_path
                    else gw.guard.scan_input(prompt)
                )
            if g.flagged:
                action = "block" if cfg.guard.input == "block" else "flag"
                for c in g.categories or ["unknown"]:
                    m.guard.labels("input", action, c).inc()
                if action == "block":
                    m.requests.labels(tenant.name, requested, "400", "none").inc()
                    raise GatewayError(
                        400,
                        "request blocked by input guardrails: " + ",".join(g.categories),
                        "guardrail_error",
                        "guardrail_input_blocked",
                        {"x-request-id": rid},
                    )
                headers["x-llmgate-guard"] = "input-flagged:" + ",".join(g.categories)

        # ---- routing
        ctx = RouteContext(requested, prompt, tenant.name, dict(request.headers))
        try:
            thr = parse_threshold(request.headers.get("x-llmgate-threshold"), cfg.routing.costroute.threshold)
            score = None
            if gw.router.needs_score(ctx):
                with span("llmgate.route.score"):
                    score = float((await asyncio.to_thread(gw.scorer.score, [prompt]))[0])
            route = gw.router.route(ctx, thr, score)
        except ValueError as e:
            raise GatewayError(400, str(e), code="invalid_threshold") from None
        except RoutingError as e:
            raise GatewayError(404, str(e), code="model_not_found") from None
        m.routes.labels(route.strategy, route.model).inc()
        headers.update({"x-llmgate-route": route.strategy})
        if route.score is not None:
            headers["x-llmgate-score"] = f"{route.score:.4f}"
        root.set_attribute("llmgate.model", route.model)

        # ---- cache lookup
        scope = tenant.id if cfg.cache.scope == "tenant" else "global"
        bypass = (
            request.headers.get("x-llmgate-cache", "").lower() in ("bypass", "no-cache")
            or "no-cache" in request.headers.get("cache-control", "").lower()
        )
        cacheable = is_cacheable(body, cfg.cache.only_deterministic) and not bypass
        ekey = cache_key(body, route.model, scope) if (cacheable and gw.exact is not None) else None
        sem = semantic_parts(body, route.model, scope) if (cacheable and gw.semantic is not None) else None
        sem_vec = None
        if ekey is not None:
            with span("llmgate.cache.exact"):
                hit = await gw.exact.get(ekey)
            m.cache.labels("exact", "hit" if hit is not None else "miss").inc()
            if hit is not None:
                return _from_cache(
                    hit, "exact", auth, rid, route.model, route.strategy, headers, stream, include_usage, t0, est
                )
        if sem is not None and gw.semantic is not None:
            with span("llmgate.cache.semantic") as s:
                res = await gw.semantic.lookup(sem[0], sem[1])
                s.set_attribute("llmgate.similarity", res.similarity)
            m.cache.labels("semantic", "hit" if res.hit else "miss").inc()
            sem_vec = res.vector
            if res.hit and isinstance(res.payload, dict):
                headers["x-llmgate-similarity"] = f"{res.similarity:.4f}"
                return _from_cache(
                    res.payload,
                    "semantic",
                    auth,
                    rid,
                    route.model,
                    route.strategy,
                    headers,
                    stream,
                    include_usage,
                    t0,
                    est,
                )

        # ---- upstream
        upstream_body = {k: v for k, v in body.items() if k not in ("model",)}
        try:
            with span("llmgate.upstream", None, **{"llmgate.candidates": ",".join(route.candidates())}):
                result = await (gw.dispatcher.stream if stream else gw.dispatcher.complete)(
                    route.candidates(), upstream_body
                )
        except ProviderError as e:
            m.requests.labels(tenant.name, route.model, str(e.status), "miss").inc()
            if isinstance(e.body, dict) and e.status in (400, 413, 422):
                return JSONResponse(e.body, status_code=e.status, headers=headers)
            code = {"timeout": "upstream_timeout", "circuit_open": "upstream_unavailable"}.get(e.kind, "upstream_error")
            raise GatewayError(
                e.status if e.status >= 500 else 502, e.message, "upstream_error", code, headers
            ) from None
        if result.fallback_used:
            m.fallbacks.labels(route.model, result.model).inc()
            headers["x-llmgate-fallback-from"] = route.model
        headers.update(
            {"x-llmgate-model": result.model, "x-llmgate-provider": result.provider, "x-llmgate-cache": "miss"}
        )
        upstream_total = sum(a.latency_s for a in result.attempts)

        if not stream:
            payload = result.payload or {}
            text = completion_text(payload)
            flagged_out = False
            if gw.guard is not None and cfg.guard.output != "off" and text:
                with span("llmgate.guard.output"):
                    g = gw.guard.scan_output(text)
                if g.flagged:
                    flagged_out = True
                    action = cfg.guard.output
                    for c in g.categories or ["unknown"]:
                        m.guard.labels("output", action, c).inc()
                    if action == "block":
                        raise GatewayError(
                            400,
                            "response blocked by output guardrails: " + ",".join(g.categories),
                            "guardrail_error",
                            "guardrail_output_blocked",
                            headers,
                        )
                    if action == "redact" and g.redacted_text is not None:
                        _set_completion_text(payload, g.redacted_text)
                    headers["x-llmgate-guard"] = "output-" + ("redacted" if action == "redact" else "flagged")
            u = usage_from_response(payload, prompt, text)
            elapsed = time.perf_counter() - t0
            cost, _ = gw.account(
                auth=auth,
                request_id=rid,
                model=result.model,
                provider=result.provider,
                route=route.strategy,
                cache="miss",
                status=200,
                prompt_tokens=u["prompt_tokens"],
                completion_tokens=u["completion_tokens"],
                latency_s=elapsed,
                est_tokens=est,
            )
            if not isinstance(payload.get("usage"), dict):
                payload["usage"] = {
                    "prompt_tokens": u["prompt_tokens"],
                    "completion_tokens": u["completion_tokens"],
                    "total_tokens": u["prompt_tokens"] + u["completion_tokens"],
                }
            if not flagged_out:
                await _store(ekey, sem, sem_vec, payload)
            m.latency.labels(result.model, "miss", "false").observe(elapsed)
            m.overhead.labels("miss").observe(max(0.0, elapsed - upstream_total))
            headers["x-llmgate-cost-usd"] = f"{cost:.8f}"
            return JSONResponse(payload, headers=headers)

        return StreamingResponse(
            _relay(
                result.stream,
                auth,
                rid,
                result.model,
                result.provider,
                route.strategy,
                prompt,
                include_usage,
                ekey,
                sem,
                sem_vec,
                t0,
                est,
                result.upstream_s,
            ),
            media_type="text/event-stream",
            headers={**headers, "cache-control": "no-cache"},
        )

    async def _store(ekey: Optional[str], sem: Any, sem_vec: Any, payload: Dict[str, Any]) -> None:
        try:
            if ekey is not None:
                await gw.exact.set(ekey, payload, cfg.cache.exact.ttl_s)
            if sem is not None and gw.semantic is not None:
                await gw.semantic.store(sem[0], sem[1], payload, sem_vec)
        except Exception:
            log.exception("cache store failed")

    def _from_cache(
        payload: Dict[str, Any],
        layer: str,
        auth: AuthContext,
        rid: str,
        model: str,
        strategy: str,
        headers: Dict[str, str],
        stream: bool,
        include_usage: bool,
        t0: float,
        est: int,
    ) -> Response:
        u = usage_from_response(payload, "", completion_text(payload))
        elapsed = time.perf_counter() - t0
        _, saved = gw.account(
            auth=auth,
            request_id=rid,
            model=model,
            provider="cache",
            route=strategy,
            cache=layer,
            status=200,
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
            latency_s=elapsed,
            est_tokens=est,
        )
        m.latency.labels(model, layer, str(stream).lower()).observe(elapsed)
        m.overhead.labels(layer).observe(elapsed)
        headers = {
            **headers,
            "x-llmgate-cache": layer,
            "x-llmgate-model": model,
            "x-llmgate-provider": "cache",
            "x-llmgate-cost-usd": "0",
            "x-llmgate-saved-usd": f"{saved:.8f}",
        }
        if stream:
            chunks = synth_stream(payload, include_usage)

            async def gen() -> AsyncIterator[bytes]:
                for c in chunks:
                    yield c

            return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
        return JSONResponse(payload, headers=headers)

    async def _relay(
        upstream: Any,
        auth: AuthContext,
        rid: str,
        model: str,
        provider: str,
        strategy: str,
        prompt: str,
        include_usage: bool,
        ekey: Optional[str],
        sem: Any,
        sem_vec: Any,
        t0: float,
        est: int,
        ttfb: float,
    ) -> AsyncIterator[bytes]:
        """Relay OpenAI SSE events while accumulating text and usage for accounting, guard and cache."""
        m.ttfb.labels(model).observe(ttfb)
        parser = SSEParser()
        parts: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        finish: Optional[str] = None
        meta: Dict[str, Any] = {}
        buffer_mode = gw.guard is not None and cfg.guard.output != "off" and cfg.guard.stream_output == "buffer"
        held: List[bytes] = []
        status = 200
        try:
            async for raw in upstream:
                for _event, data in parser.feed(raw):
                    if data.strip() == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in ev:
                        status = 502
                    if not meta and ev.get("id"):
                        meta = {"id": ev.get("id"), "created": ev.get("created"), "model": ev.get("model")}
                    if isinstance(ev.get("usage"), dict):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if isinstance(d.get("content"), str):
                            parts.append(d["content"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                    if not ev.get("choices") and "usage" in ev and not include_usage:
                        continue  # the gateway asked for usage; the client did not
                    out = b"data: " + data.encode() + b"\n\n"
                    if buffer_mode:
                        held.append(out)
                    else:
                        yield out
        except Exception as exc:  # upstream died mid-stream: tell the client in-band
            status = 502
            log.warning("stream from %s aborted: %s", provider, exc)
            yield sse({"error": {"message": "upstream stream aborted", "type": "upstream_error"}})
        text = "".join(parts)
        flagged = False
        if gw.guard is not None and cfg.guard.output != "off" and text:
            g = gw.guard.scan_output(text)
            if g.flagged:
                flagged = True
                action = cfg.guard.output
                for c in g.categories or ["unknown"]:
                    m.guard.labels("output", action, c).inc()
                if buffer_mode:
                    if action == "block":
                        held = [
                            sse(
                                {
                                    "error": {
                                        "message": "response blocked by output guardrails",
                                        "type": "guardrail_error",
                                        "code": "guardrail_output_blocked",
                                    }
                                }
                            )
                        ]
                    elif action == "redact":
                        p = {
                            **meta,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": g.redacted_text or ""},
                                    "finish_reason": finish or "stop",
                                }
                            ],
                            "usage": usage,
                        }
                        held = synth_stream(p, include_usage)[:-1]
                else:
                    yield sse(
                        {
                            "llmgate": {
                                "guard": "output-flagged",
                                "categories": g.categories,
                                "note": "streamed content was already sent; use guard.stream_output=buffer "
                                "to redact or block before sending",
                            }
                        }
                    )
        for h in held:
            yield h
        yield SSE_DONE
        u = usage_from_response({"usage": usage} if usage else {}, prompt, text)
        elapsed = time.perf_counter() - t0
        gw.account(
            auth=auth,
            request_id=rid,
            model=model,
            provider=provider,
            route=strategy,
            cache="miss",
            status=status,
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
            latency_s=elapsed,
            est_tokens=est,
        )
        m.latency.labels(model, "miss", "true").observe(elapsed)
        if status == 200 and not flagged and finish is not None:
            payload = {
                "id": meta.get("id") or "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": meta.get("created") or int(time.time()),
                "model": meta.get("model") or model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": u["prompt_tokens"],
                    "completion_tokens": u["completion_tokens"],
                    "total_tokens": u["prompt_tokens"] + u["completion_tokens"],
                },
            }
            await _store(ekey, sem, sem_vec, payload)

    # -------------------------------------------------------------------- admin
    def _admin(request: Request) -> None:
        key = cfg.server.admin_key
        if not key:
            raise GatewayError(404, "admin API is disabled (set server.admin_key)", code="admin_disabled")
        given = request.headers.get("x-admin-key") or _bearer(request)
        import hmac

        if not given or not hmac.compare_digest(given, key):
            raise GatewayError(401, "invalid admin key", "authentication_error", "invalid_admin_key")

    @app.get("/admin/tenants")
    async def admin_tenants(request: Request) -> Dict[str, Any]:
        _admin(request)
        return {"data": [t.as_dict() for t in await asyncio.to_thread(gw.store.list_tenants)]}

    @app.post("/admin/tenants")
    async def admin_create_tenant(request: Request) -> JSONResponse:
        _admin(request)
        b = await request.json()
        try:
            t = await asyncio.to_thread(
                gw.store.create_tenant,
                b["name"],
                int(b.get("rpm", 0)),
                int(b.get("tpm", 0)),
                float(b.get("budget_usd", 0)),
                b.get("budget_period", "monthly"),
                b.get("allowed_models"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise GatewayError(400, f"invalid tenant: {e}") from None
        except Exception as e:
            raise GatewayError(409, f"could not create tenant: {e.__class__.__name__}", code="conflict") from None
        return JSONResponse(t.as_dict(), status_code=201)

    @app.post("/admin/keys")
    async def admin_create_key(request: Request) -> JSONResponse:
        _admin(request)
        b = await request.json()
        try:
            plain, rec = await asyncio.to_thread(gw.store.create_key, b["tenant"], b.get("name", ""))
        except KeyError as e:
            raise GatewayError(404, str(e), code="tenant_not_found") from None
        return JSONResponse(
            {
                "id": rec.id,
                "tenant_id": rec.tenant_id,
                "key": plain,
                "note": "store this key now; it cannot be shown again",
            },
            status_code=201,
        )

    @app.delete("/admin/keys/{key_id}")
    async def admin_revoke_key(key_id: str, request: Request) -> Dict[str, Any]:
        _admin(request)
        ok = await asyncio.to_thread(gw.store.revoke_key, key_id)
        if not ok:
            raise GatewayError(404, "no such key", code="key_not_found")
        gw._keys.clear()
        return {"id": key_id, "revoked": True}

    @app.get("/admin/usage")
    async def admin_usage(
        request: Request, by: str = "tenant,model", days: float = 30.0, tenant: Optional[str] = None
    ) -> Dict[str, Any]:
        _admin(request)
        await gw.writer.flush()
        since = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            rows = await asyncio.to_thread(gw.store.usage_summary, since, [x for x in by.split(",") if x], tenant)
        except ValueError as e:
            raise GatewayError(400, str(e)) from None
        return {"since": since.isoformat(), "data": rows}

    return app
