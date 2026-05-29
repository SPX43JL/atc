#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from urllib import request


def call(port: int) -> dict:
    payload = {
        "model": "Qwen2.5-7B-Instruct",
        "messages": [{
            "role": "user",
            "content": "用一句话说明KV cache是什么"
        }],
        "max_tokens": 32,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return {
        "port": port,
        "latency_s": time.perf_counter() - start,
        "content": obj["choices"][0]["message"]["content"],
    }


def main() -> int:
    ports = [int(p) for p in sys.argv[1:]] or [8100, 8101]
    results = [call(port) for port in ports]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
