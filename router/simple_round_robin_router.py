#!/usr/bin/env python3
"""Small OpenAI-compatible round-robin router with pressure tracing."""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

BACKENDS = os.environ.get(
    "VLLM_BACKENDS",
    "http://127.0.0.1:8000/v1/chat/completions,http://127.0.0.1:8001/v1/chat/completions",
).split(",")
BACKENDS = [b.strip() for b in BACKENDS if b.strip()]
TIMEOUT_SECONDS = float(os.environ.get("ROUTER_TIMEOUT_SECONDS", "600"))
STATE_PATH = os.environ.get("ATC_SERVING_STATE_PATH", "")
TRACE_PATH = os.environ.get("ATC_ROUTER_TRACE_PATH", "")

counter = itertools.count(1)
rr_index = itertools.count(0)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def estimate_input_len(payload: dict[str, Any]) -> int:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return len(prompt)
    if isinstance(prompt, list):
        return sum(len(x) for x in prompt if isinstance(x, str))
    messages = payload.get("messages") or []
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(part.get("text", ""))
                         for part in content if isinstance(part, dict))
    return total


def _write_json_atomic(path: str, obj: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(target)


def _append_trace(obj: dict[str, Any]) -> None:
    if not TRACE_PATH:
        return
    target = Path(TRACE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _pressure_state(app: web.Application,
                    req_id: int,
                    headers: dict[str, str],
                    input_chars: int,
                    backend: str,
                    status: int | None = None,
                    elapsed: float | None = None) -> dict[str, Any]:
    max_concurrency = int(headers.get("X-ATC-Max-Concurrency", "1") or 1)
    inflight = int(app["inflight"])
    queue_length = max(0, inflight - max_concurrency)
    pressure = min(1.0, (inflight + queue_length) / max(1, max_concurrency))
    return {
        "request_id": req_id,
        "client_request_id": headers.get("X-ATC-Client-Request-ID", ""),
        "sample_id": headers.get("X-ATC-Sample-ID", ""),
        "method": headers.get("X-ATC-Method", ""),
        "dataset": headers.get("X-ATC-Dataset", ""),
        "workload": headers.get("X-ATC-Workload", "unknown"),
        "max_concurrency": max_concurrency,
        "inflight": inflight,
        "queue_length": queue_length,
        "pressure": pressure,
        "backend": backend,
        "input_chars": input_chars,
        "status": status,
        "elapsed_s": elapsed,
        "updated_at": time.time(),
    }


async def chat_completions(request: web.Request) -> web.StreamResponse:
    return await forward_openai(request, "/v1/chat/completions")


async def completions(request: web.Request) -> web.StreamResponse:
    return await forward_openai(request, "/v1/completions")


async def forward_openai(request: web.Request,
                         endpoint_path: str) -> web.StreamResponse:
    req_id = next(counter)
    started = time.perf_counter()
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid json"}, status=400)

    backend = BACKENDS[next(rr_index) % len(BACKENDS)]
    if endpoint_path == "/v1/completions":
        backend = backend.replace("/v1/chat/completions", "/v1/completions")
    input_chars = estimate_input_len(payload)
    status = 502
    headers_in = {k: v for k, v in request.headers.items()}
    request.app["inflight"] += 1
    state = _pressure_state(request.app, req_id, headers_in, input_chars, backend)
    _write_json_atomic(STATE_PATH, state)
    _append_trace({"event": "start", **state})
    try:
        async with request.app["session"].post(backend, json=payload) as resp:
            status = resp.status
            body = await resp.read()
            elapsed = time.perf_counter() - started
            done_state = _pressure_state(request.app, req_id, headers_in,
                                         input_chars, backend, status, elapsed)
            logging.info(
                "request_id=%s client_id=%s workload=%s backend=%s "
                "input_chars=%s inflight=%s queue=%s status=%s elapsed=%.3fs",
                req_id, headers_in.get("X-ATC-Client-Request-ID", ""),
                headers_in.get("X-ATC-Workload", "unknown"), backend,
                input_chars, done_state["inflight"], done_state["queue_length"],
                status, elapsed)
            _append_trace({"event": "finish", **done_state})
            headers = {"content-type": resp.headers.get("content-type", "application/json")}
            return web.Response(body=body, status=status, headers=headers)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        err_state = _pressure_state(request.app, req_id, headers_in,
                                    input_chars, backend, status, elapsed)
        err_state["error"] = repr(exc)
        _append_trace({"event": "error", **err_state})
        logging.exception(
            "request_id=%s backend=%s input_chars=%s status=%s elapsed=%.3fs error=%r",
            req_id, backend, input_chars, status, elapsed, exc)
        return web.json_response({"error": str(exc), "backend": backend}, status=502)
    finally:
        request.app["inflight"] = max(0, int(request.app["inflight"]) - 1)
        idle_state = _pressure_state(request.app, req_id, headers_in,
                                     input_chars, backend, status,
                                     time.perf_counter() - started)
        _write_json_atomic(STATE_PATH, idle_state)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "backends": BACKENDS})


async def on_startup(app: web.Application) -> None:
    app["session"] = ClientSession(timeout=ClientTimeout(total=TIMEOUT_SECONDS))
    app["inflight"] = 0
    logging.info("router started backends=%s state_path=%s trace_path=%s",
                 BACKENDS, STATE_PATH, TRACE_PATH)


async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def create_app() -> web.Application:
    if not BACKENDS:
        raise RuntimeError("No VLLM_BACKENDS configured")
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0",
                port=int(os.environ.get("ROUTER_PORT", "9000")))
