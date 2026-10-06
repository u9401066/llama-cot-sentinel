"""OpenAI-compatible HTTP service: every chat completion is answered with step-checked reasoning (see core.py).

    POST /v1/chat/completions   guarded answer (stream or not); a leading trigger prefix such as "med:" is removed
    GET  /v1/models             the sentinel's model id
    GET  /health                {"status": "ok", ...}
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .core import SentinelConfig, SentinelRun, SystemOneCritic, sum_timings

log = logging.getLogger("llama_cot_sentinel")


def _text(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def strip_trigger(messages: List[Dict[str, Any]], prefixes: List[str]) -> None:
    """Remove a trigger prefix (e.g. 'med:') from the first user message."""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        for p in prefixes:
            if isinstance(c, str) and c.lstrip().lower().startswith(p.lower()):
                m["content"] = c.lstrip()[len(p):].lstrip()
        return


class Sentinel:
    def __init__(self, cfg: SentinelConfig, backend: str, timeout: float = 900.0, transport: Optional[httpx.AsyncBaseTransport] = None,
                 critic_transport: Optional[httpx.AsyncBaseTransport] = None):
        self.cfg = cfg
        self.backend_url = backend.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.backend_url, timeout=timeout, transport=transport)
        self.critic = SystemOneCritic(cfg, transport=critic_transport)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.critic.aclose()

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "model": self.cfg.model_id, "backend": self.backend_url, "critic": self.cfg.critic_url,
                             "checks": [c.id for c in self.cfg.checks]})

    async def models(self, request: Request) -> Response:
        mid = self.cfg.model_id
        return JSONResponse({"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "llama-cot-sentinel", "created": 0}],
                             "models": [{"name": mid, "model": mid, "type": "model", "description": "reasoning checked step by step by a decision model"}]})

    async def chat(self, request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        # Plain conversation only: tools, tool calls and earlier reasoning are not part of a guarded answer.
        messages = [dict(m) for m in body.get("messages") or [] if m.get("role") in ("system", "user", "assistant")]
        for m in messages:
            m.pop("reasoning_content", None)
            m.pop("tool_calls", None)
        if not any(m.get("role") == "user" and _text(m).strip() for m in messages):
            return JSONResponse({"error": {"message": "no user message"}}, status_code=400)
        strip_trigger(messages, self.cfg.triggers)
        options = {k: body[k] for k in ("temperature", "top_p", "chat_template_kwargs") if k in body}
        run = SentinelRun(self.cfg, self.client, self.critic, messages, options)
        model = body.get("model") or self.cfg.model_id
        created = int(time.time())
        cid = "chatcmpl-sentinel-%d" % created

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> str:
            data: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            data.update(extra or {})
            return "data: %s\n\n" % json.dumps(data, ensure_ascii=False)

        def report(ev: Dict[str, Any]) -> Dict[str, Any]:
            return {"sentinel": {k: ev[k] for k in ("interventions", "flagged", "steps_checked", "critic_ms", "attempts")},
                    "timings": sum_timings(ev["timings"]) or {}}

        if body.get("stream"):
            async def gen():
                yield chunk({"role": "assistant", "content": ""})
                try:
                    async for ev in run.events():
                        if ev["type"] in ("reasoning", "note"):
                            yield chunk({"reasoning_content": ev["text"]})
                        elif ev["type"] == "content":
                            yield chunk({"content": ev["text"]})
                        elif ev["type"] == "done":
                            yield chunk({}, "stop", report(ev))
                except httpx.HTTPError as e:
                    log.warning("sentinel answer failed: %s: %s", type(e).__name__, e)
                    yield chunk({"content": "\n\n**Sentinel error:** %s" % e}, "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-sentinel": "1"})
        content, reasoning, final = "", "", {}
        try:
            async for ev in run.events():
                if ev["type"] in ("reasoning", "note"):
                    reasoning += ev["text"]
                elif ev["type"] == "content":
                    content += ev["text"]
                elif ev["type"] == "done":
                    final = report(ev)
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": "backend error: %s" % e, "type": "upstream_error"}}, status_code=502)
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        t = final.get("timings") or {}
        usage = {"prompt_tokens": int(t.get("prompt_n", 0) + t.get("cache_n", 0)), "completion_tokens": int(t.get("predicted_n", 0))}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage, **final})


def create_app(cfg: SentinelConfig, backend: str, timeout: float = 900.0, transport: Optional[httpx.AsyncBaseTransport] = None,
               critic_transport: Optional[httpx.AsyncBaseTransport] = None) -> Starlette:
    sentinel = Sentinel(cfg, backend, timeout, transport, critic_transport)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        await sentinel.aclose()

    app = Starlette(routes=[Route("/v1/chat/completions", sentinel.chat, methods=["POST"]), Route("/v1/models", sentinel.models, methods=["GET"]),
                            Route("/models", sentinel.models, methods=["GET"]), Route("/health", sentinel.health, methods=["GET"])], lifespan=lifespan)
    app.state.sentinel = sentinel
    return app
