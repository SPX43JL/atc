#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import aiohttp


def load_sharegpt(path: Path, limit: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for item in data:
        conv = item.get("conversations") or []
        if not conv:
            continue
        text = conv[0].get("value") or conv[0].get("content") or ""
        if text:
            prompts.append((text[:4096], 64))
        if len(prompts) >= limit:
            break
    return prompts


def load_burstgpt(path: Path, limit: int):
    prompts = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get("Model") != "GPT-4":
                    continue
                in_tok = int(float(row.get("Request tokens", "0")))
                out_tok = int(float(row.get("Response tokens", "0")))
            except ValueError:
                continue
            if out_tok <= 0:
                continue
            words = max(8, min(in_tok, 1024))
            prompt = "Please answer briefly. " + "benchmark " * words
            prompts.append((prompt, max(8, min(out_tok, 128))))
            if len(prompts) >= limit:
                break
    return prompts


def load_random(limit: int):
    return [("Please explain vLLM continuous batching in one short paragraph. " + str(i), 64) for i in range(limit)]


async def one(session, url, model, prompt, max_tokens, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
        elapsed = time.perf_counter() - start
        return {"idx": idx, "status": resp.status, "latency_s": elapsed, "bytes": len(text), "ok": resp.status == 200}


async def run(args):
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    if args.dataset_name == "sharegpt" and dataset_path and dataset_path.exists():
        prompts = load_sharegpt(dataset_path, args.num_prompts)
    elif args.dataset_name == "burstgpt" and dataset_path and dataset_path.exists():
        prompts = load_burstgpt(dataset_path, args.num_prompts)
    else:
        prompts = load_random(args.num_prompts)
    if not prompts:
        raise SystemExit("No prompts loaded")

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    results = []
    started = time.perf_counter()
    sem = asyncio.Semaphore(args.max_concurrency)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bound(i, prompt, max_tokens):
            if args.request_rate > 0:
                await asyncio.sleep(i / args.request_rate)
            async with sem:
                return await one(session, url, args.model, prompt, max_tokens, i)
        tasks = [asyncio.create_task(bound(i, p, mt)) for i, (p, mt) in enumerate(prompts)]
        for task in asyncio.as_completed(tasks):
            results.append(await task)

    total = time.perf_counter() - started
    lats = [r["latency_s"] for r in results if r["ok"]]
    summary = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "num_prompts": len(prompts),
        "ok": sum(r["ok"] for r in results),
        "failed": sum(not r["ok"] for r in results),
        "total_s": total,
        "request_throughput_rps": len(results) / total if total else 0,
        "latency_avg_s": statistics.mean(lats) if lats else None,
        "latency_p50_s": statistics.median(lats) if lats else None,
        "latency_p95_s": sorted(lats)[int(0.95 * (len(lats)-1))] if lats else None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "requests": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--dataset-name", choices=["sharegpt", "burstgpt", "random"], default="random")
    p.add_argument("--dataset-path")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--request-rate", type=float, default=2.0)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output", required=True)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
