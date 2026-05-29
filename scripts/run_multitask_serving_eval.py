#!/usr/bin/env python3
"""Serving evaluator for manifest-based LongBench/math fake-quant runs."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import re
import string
import time
from pathlib import Path
from typing import Any

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9100")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="none")
    parser.add_argument("--method-variant", default="")
    parser.add_argument("--workload", choices=["sequential", "burst", "time-series"], default="burst")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--indices", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path, indices: str) -> list[dict[str, Any]]:
    selected = parse_indices(indices)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if selected and int(row["index"]) not in selected:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return rows


def parse_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid range {part}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return out


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_tokens(prediction: list[str], ground_truth: list[str]) -> float:
    common = collections.Counter(prediction) & collections.Counter(ground_truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(1, len(prediction))
    recall = same / max(1, len(ground_truth))
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    return f1_tokens(normalize_answer(prediction).split(),
                     normalize_answer(ground_truth).split())


def retrieval_score(prediction: str, ground_truth: str) -> float:
    match = re.findall(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    target = match[0]
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return sum(1 for n in nums if n == target) / len(nums)


def count_score(prediction: str, ground_truth: str) -> float:
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    target = str(ground_truth)
    return sum(1 for n in nums if n == target) / len(nums)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred = prediction.split()
    ref = ground_truth.split()
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for token in pred:
        cur = [0]
        for j, ref_token in enumerate(ref, 1):
            if token == ref_token:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    brace = text.find("{", idx)
    if brace < 0:
        return ""
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:pos]
    return ""


def normalize_math(text: str) -> str:
    text = extract_boxed(text) or text
    text = text.strip()
    text = re.sub(r"(?i)final answer:?", "", text)
    text = re.sub(r"(?i)the answer is", "", text)
    text = text.replace("$", "")
    for pat in ["\\left", "\\right", "\\!", "\\,", "\\ "]:
        text = text.replace(pat, "")
    text = text.replace(" ", "")
    text = text.rstrip(".")
    return text.lower()


def extract_gsm8k_number(text: str) -> str:
    if "####" in text:
        text = text.split("####", 1)[1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else "[invalid]"


def metric_score(prediction: str, answers: list[str], metric: str) -> float:
    if metric == "qa_f1":
        return max((qa_f1_score(prediction, ans) for ans in answers), default=0.0)
    if metric == "retrieval":
        return max((retrieval_score(prediction, ans) for ans in answers), default=0.0)
    if metric == "count":
        return max((count_score(prediction, ans) for ans in answers), default=0.0)
    if metric == "rouge_l":
        return max((rouge_l_score(prediction, ans) for ans in answers), default=0.0)
    if metric == "gsm8k_exact":
        pred = extract_gsm8k_number(prediction)
        return float(any(pred == extract_gsm8k_number(ans) or pred == ans for ans in answers))
    if metric == "math_exact":
        pred = normalize_math(prediction)
        return float(any(pred == normalize_math(ans) for ans in answers))
    raise ValueError(f"Unsupported metric {metric}")


async def request_one(session: aiohttp.ClientSession, url: str, model: str,
                      row: dict[str, Any], args: argparse.Namespace,
                      request_index: int) -> dict[str, Any]:
    if args.endpoint == "completions":
        payload = {
            "model": model,
            "prompt": row["prompt"],
            "max_tokens": int(row["max_gen"]),
            "temperature": args.temperature,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "max_tokens": int(row["max_gen"]),
            "temperature": args.temperature,
        }
    headers = {
        "X-ATC-Method": args.method,
        "X-ATC-Dataset": args.dataset,
        "X-ATC-Workload": args.workload,
        "X-ATC-Max-Concurrency": str(args.max_concurrency),
        "X-ATC-Client-Request-ID": str(request_index),
        "X-ATC-Sample-ID": str(row["sample_id"]),
    }
    started = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            elapsed = time.perf_counter() - started
            pred = ""
            finish_reason = ""
            usage: dict[str, Any] = {}
            error = ""
            if resp.status == 200:
                obj = json.loads(text)
                choice = obj["choices"][0]
                if args.endpoint == "completions":
                    pred = choice.get("text") or ""
                else:
                    pred = choice.get("message", {}).get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "")
                usage = obj.get("usage") or {}
            else:
                error = text[:1000]
            completion_tokens = int(usage.get("completion_tokens") or 0)
            hit_max = finish_reason == "length" or (
                completion_tokens > 0 and completion_tokens >= int(row["max_gen"]))
            score = (metric_score(pred, list(row.get("answers") or []), row["metric"])
                     if resp.status == 200 else 0.0)
            return {
                "dataset": row["dataset"],
                "family": row.get("family"),
                "metric": row["metric"],
                "sample_id": row["sample_id"],
                "index": row["index"],
                "answers": row.get("answers") or [],
                "length": row.get("length"),
                "prompt_tokens": row.get("prompt_tokens"),
                "original_prompt_tokens": row.get("original_prompt_tokens"),
                "prompt_truncated": bool(row.get("prompt_truncated")),
                "max_gen": int(row["max_gen"]),
                "status": resp.status,
                "latency_s": elapsed,
                "prediction": pred,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "usage_prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "hit_max_tokens": bool(hit_max),
                "score": score,
                "raw_bytes": len(text),
                "error": error,
            }
    except Exception as exc:
        return {
            "dataset": row["dataset"],
            "family": row.get("family"),
            "metric": row["metric"],
            "sample_id": row["sample_id"],
            "index": row["index"],
            "answers": row.get("answers") or [],
            "length": row.get("length"),
            "prompt_tokens": row.get("prompt_tokens"),
            "original_prompt_tokens": row.get("original_prompt_tokens"),
            "prompt_truncated": bool(row.get("prompt_truncated")),
            "max_gen": int(row["max_gen"]),
            "status": -1,
            "latency_s": time.perf_counter() - started,
            "prediction": "",
            "finish_reason": "",
            "completion_tokens": 0,
            "usage_prompt_tokens": 0,
            "total_tokens": 0,
            "hit_max_tokens": False,
            "score": 0.0,
            "raw_bytes": 0,
            "error": repr(exc),
        }


async def run(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest, args.indices)
    endpoint_path = "/v1/completions" if args.endpoint == "completions" else "/v1/chat/completions"
    url = args.base_url.rstrip("/") + endpoint_path
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    sem = asyncio.Semaphore(args.max_concurrency)
    run_start = time.perf_counter()
    request_rate = max(float(args.request_rate), 1e-6)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def bound(idx: int, row: dict[str, Any]) -> dict[str, Any]:
            scheduled = time.perf_counter()
            if args.workload == "time-series":
                target = run_start + idx / request_rate
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
            arrived = time.perf_counter()
            async with sem:
                send_time = time.perf_counter()
                out = await request_one(session, url, args.model, row, args, idx)
                out["arrival_offset_s"] = arrived - run_start
                out["queue_wait_s"] = send_time - arrived
                out["scheduled_offset_s"] = scheduled - run_start
                return out

        if args.workload == "sequential":
            results = []
            for idx, row in enumerate(rows):
                results.append(await bound(idx, row))
        else:
            results = await asyncio.gather(*(bound(idx, row) for idx, row in enumerate(rows)))

    ok = [r for r in results if r["status"] == 200]
    elapsed = time.perf_counter() - run_start
    summary = {
        "dataset": args.dataset,
        "family": rows[0].get("family"),
        "metric": rows[0].get("metric"),
        "manifest": str(args.manifest),
        "method": args.method,
        "method_variant": args.method_variant or args.method,
        "workload": args.workload,
        "endpoint": args.endpoint,
        "base_url": args.base_url,
        "model": args.model,
        "num_examples": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "max_gen": rows[0].get("max_gen"),
        "max_concurrency": args.max_concurrency,
        "request_rate": args.request_rate,
        "temperature": args.temperature,
        "indices": args.indices,
        "avg_score": avg(ok, "score"),
        "score_pct": round(100 * (avg(ok, "score") or 0.0), 2),
        "avg_latency_s": avg(ok, "latency_s"),
        "p50_latency_s": percentile(ok, "latency_s", 0.50),
        "p95_latency_s": percentile(ok, "latency_s", 0.95),
        "avg_queue_wait_s": avg(ok, "queue_wait_s"),
        "truncate_rate": sum(1 for r in ok if r["hit_max_tokens"]) / len(ok) if ok else 0.0,
        "prompt_truncate_rate": sum(1 for r in results if r["prompt_truncated"]) / len(results),
        "finish_reason_counts": dict(collections.Counter(r["finish_reason"] for r in ok)),
        "avg_completion_tokens": avg(ok, "completion_tokens"),
        "avg_prompt_tokens": avg(ok, "usage_prompt_tokens"),
        "avg_official_prompt_tokens": avg(results, "prompt_tokens"),
        "avg_original_prompt_tokens": avg(results, "original_prompt_tokens"),
        "throughput_req_s": len(ok) / elapsed if elapsed > 0 else 0.0,
        "total_elapsed_s": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "examples": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {args.output}")


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(rows: list[dict[str, Any]], key: str, p: float) -> float | None:
    vals = sorted(float(r[key]) for r in rows if r.get(key) is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(math.ceil(p * len(vals))) - 1))
    return vals[idx]


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
