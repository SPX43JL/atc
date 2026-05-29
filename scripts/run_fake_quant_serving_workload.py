#!/usr/bin/env python3
"""Small serving workload driver for fake-quant experiments."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import aiohttp


def load_qasper(path: Path, limit: int, max_context_chars: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for paper in data.values():
        context_parts = [paper.get("title", ""), paper.get("abstract", "")]
        for sec in paper.get("full_text", [])[:8]:
            context_parts.append(sec.get("section_name", ""))
            context_parts.extend(sec.get("paragraphs", [])[:2])
        context = "\n".join(x for x in context_parts if x)[:max_context_chars]
        for qa in paper.get("qas", []):
            prompt = (
                "You are answering questions about an NLP paper. "
                "Answer briefly and use only the provided context.\n\n"
                f"Paper context:\n{context}\n\n"
                f"Question: {qa.get('question', '')}\nAnswer:"
            )
            prompts.append((prompt, 64))
            if len(prompts) >= limit:
                return prompts
    return prompts


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
                in_tok = int(float(row.get("Request tokens", "0")))
                out_tok = int(float(row.get("Response tokens", "0")))
            except ValueError:
                continue
            prompt = "Please answer briefly. " + "benchmark " * max(
                8, min(in_tok, 1024))
            prompts.append((prompt, max(8, min(out_tok, 128))))
            if len(prompts) >= limit:
                break
    return prompts


def load_prompts(args):
    path = Path(args.dataset_path)
    if args.dataset == "qasper":
        return load_qasper(path, args.num_prompts, args.max_context_chars)
    if args.dataset == "sharegpt":
        return load_sharegpt(path, args.num_prompts)
    if args.dataset == "burstgpt":
        return load_burstgpt(path, args.num_prompts)
    return [("Explain vLLM continuous batching briefly.", 64)
            for _ in range(args.num_prompts)]


def arrival_delay(args, idx: int) -> float:
    if args.workload == "burst":
        return 0.0
    if args.request_rate <= 0:
        return 0.0
    return idx / args.request_rate


async def one(session, url, model, prompt, max_tokens, idx):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
        elapsed = time.perf_counter() - start
        return {
            "idx": idx,
            "status": resp.status,
            "ok": resp.status == 200,
            "latency_s": elapsed,
            "raw_bytes": len(text),
        }


async def run(args):
    prompts = load_prompts(args)
    if not prompts:
        raise SystemExit("No prompts loaded")
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    sem = asyncio.Semaphore(args.max_concurrency)
    created = []
    results = []
    started = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector,
                                     timeout=timeout) as session:

        async def bound(idx, prompt, max_tokens):
            delay = arrival_delay(args, idx)
            await asyncio.sleep(delay)
            queued_at = time.perf_counter() - started
            async with sem:
                started_at = time.perf_counter() - started
                result = await one(session, url, args.model, prompt,
                                   max_tokens, idx)
                result.update({
                    "arrival_delay_s": delay,
                    "queued_at_s": queued_at,
                    "started_at_s": started_at,
                    "queue_wait_s": max(0.0, started_at - queued_at),
                })
                return result

        for idx, (prompt, max_tokens) in enumerate(prompts):
            created.append(asyncio.create_task(bound(idx, prompt, max_tokens)))
        for task in asyncio.as_completed(created):
            results.append(await task)

    total = time.perf_counter() - started
    lats = [r["latency_s"] for r in results if r["ok"]]
    waits = [r["queue_wait_s"] for r in results]
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "workload": args.workload,
        "num_prompts": len(prompts),
        "max_concurrency": args.max_concurrency,
        "request_rate": args.request_rate,
        "ok": sum(r["ok"] for r in results),
        "failed": sum(not r["ok"] for r in results),
        "total_s": total,
        "throughput_rps": len(results) / total if total else 0.0,
        "latency_avg_s": statistics.mean(lats) if lats else None,
        "latency_p50_s": statistics.median(lats) if lats else None,
        "queue_wait_avg_s": statistics.mean(waits) if waits else None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "requests": sorted(results, key=lambda r: r["idx"]),
    },
                              indent=2,
                              ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--method", default="none")
    parser.add_argument("--dataset",
                        choices=["qasper", "sharegpt", "burstgpt", "random"],
                        default="qasper")
    parser.add_argument("--dataset-path",
                        default="/root/atc_vllm_sched/data/qasper/qasper-dev-v0.3.json")
    parser.add_argument("--workload",
                        choices=["burst", "time-series"],
                        default="burst")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
