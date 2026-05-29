#!/usr/bin/env python3
"""Prepare deterministic first-N manifests for multitask serving evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_DIR = Path("/root/atc_vllm_sched")
PROMPT_CONFIG = PROJECT_DIR / "references/kv_methods/KIVI/config/dataset2prompt.json"
MAXLEN_CONFIG = PROJECT_DIR / "references/kv_methods/KIVI/config/dataset2maxlen.json"
KVTUNER_EVALS = PROJECT_DIR / "references/kv_methods/KVTuner/benckmarks/evals"

LONG_BENCH_DATASETS = [
    "qasper",
    "narrativeqa",
    "hotpotqa",
    "passage_retrieval_en",
    "passage_count",
    "qmsum",
]
DEFAULT_ALL_DATASETS = LONG_BENCH_DATASETS + ["math500", "gsm8k"]
LONG_BENCH_METRICS = {
    "qasper": "qa_f1",
    "narrativeqa": "qa_f1",
    "hotpotqa": "qa_f1",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "qmsum": "rouge_l",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_DIR / "data/eval_manifests/20260526_multitask_200")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--datasets", default=",".join(DEFAULT_ALL_DATASETS))
    parser.add_argument("--tokenizer-path", type=Path,
                        default=PROJECT_DIR / "models/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-input-tokens", type=int, default=7500)
    parser.add_argument("--longbench-cache-dir", type=Path,
                        default=PROJECT_DIR / "data/longbench_cache")
    parser.add_argument("--math-cache-dir", type=Path,
                        default=PROJECT_DIR / "data/math_cache")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256_file(path)


def middle_truncate_prompt(tokenizer: Any, prompt: str,
                           max_input_tokens: int) -> tuple[str, int, int, bool]:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original_len = int(tokenized.shape[0])
    if original_len <= max_input_tokens:
        return prompt, original_len, original_len, False
    half = max_input_tokens // 2
    truncated = tokenizer.decode(tokenized[:half], skip_special_tokens=True)
    truncated += tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    final_len = int(tokenizer(truncated, truncation=False,
                              return_tensors="pt").input_ids.shape[-1])
    return truncated, final_len, original_len, True


def official_longbench_config(dataset: str) -> tuple[str, int]:
    prompt_map = json.loads(PROMPT_CONFIG.read_text(encoding="utf-8"))
    maxlen_map = json.loads(MAXLEN_CONFIG.read_text(encoding="utf-8"))
    return prompt_map[dataset], int(maxlen_map[dataset])


def prepare_longbench(dataset: str, args: argparse.Namespace,
                      tokenizer: Any) -> tuple[Path, dict[str, Any]]:
    prompt_format, max_gen = official_longbench_config(dataset)
    ds = load_dataset("THUDM/LongBench", dataset, split="test",
                      cache_dir=str(args.longbench_cache_dir),
                      trust_remote_code=True)
    if len(ds) < args.limit:
        raise SystemExit(f"LongBench/{dataset} has {len(ds)} rows < {args.limit}")
    rows: list[dict[str, Any]] = []
    for idx, obj_raw in enumerate(ds.select(range(args.limit))):
        obj = dict(obj_raw)
        raw_prompt = prompt_format.format(**obj)
        prompt, prompt_tokens, original_prompt_tokens, prompt_truncated = (
            middle_truncate_prompt(tokenizer, raw_prompt, args.max_input_tokens))
        rows.append({
            "family": "longbench",
            "dataset": dataset,
            "hf_dataset": "THUDM/LongBench",
            "hf_config": dataset,
            "split": "test",
            "index": idx,
            "sample_id": str(obj.get("_id") or idx),
            "prompt": prompt,
            "answers": list(obj.get("answers") or []),
            "all_classes": obj.get("all_classes") or [],
            "length": obj.get("length"),
            "metric": LONG_BENCH_METRICS[dataset],
            "max_gen": max_gen,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_truncated": prompt_truncated,
            "prompt_source": str(PROMPT_CONFIG),
            "max_gen_source": str(MAXLEN_CONFIG),
            "truncation": "LongBench token middle truncation",
        })
    out = args.out_dir / f"longbench_{dataset}_first{args.limit}.jsonl"
    digest = write_jsonl(out, rows)
    return out, {
        "dataset": dataset,
        "family": "longbench",
        "path": str(out),
        "sha256": digest,
        "num_rows": len(rows),
        "hf_dataset": "THUDM/LongBench",
        "hf_config": dataset,
        "split": "test",
        "metric": LONG_BENCH_METRICS[dataset],
        "max_gen": max_gen,
        "prompt_source": str(PROMPT_CONFIG),
        "max_gen_source": str(MAXLEN_CONFIG),
    }


def normalize_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####", 1)[1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return nums[-1] if nums else "[invalid]"


def gsm8k_prompt_builder(seed: int):
    sys.path.insert(0, str(KVTUNER_EVALS))
    from gsm8k_utils import build_prompt  # type: ignore

    def build(question: str) -> str:
        random.seed(seed)
        return build_prompt(question, 8, True)

    return build


def prepare_gsm8k(args: argparse.Namespace, tokenizer: Any) -> tuple[Path, dict[str, Any]]:
    build_prompt = gsm8k_prompt_builder(args.seed)
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      cache_dir=str(args.math_cache_dir))
    if len(ds) < args.limit:
        raise SystemExit(f"GSM8K test has {len(ds)} rows < {args.limit}")
    rows: list[dict[str, Any]] = []
    for idx, obj_raw in enumerate(ds.select(range(args.limit))):
        obj = dict(obj_raw)
        prompt = build_prompt(str(obj["question"]))
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        rows.append({
            "family": "math",
            "dataset": "gsm8k",
            "hf_dataset": "openai/gsm8k",
            "hf_config": "main",
            "split": "test",
            "index": idx,
            "sample_id": f"gsm8k-{idx}",
            "prompt": prompt,
            "answers": [normalize_gsm8k_answer(str(obj["answer"]))],
            "raw_answer": str(obj["answer"]),
            "metric": "gsm8k_exact",
            "max_gen": 512,
            "prompt_tokens": int(tokenized.shape[0]),
            "original_prompt_tokens": int(tokenized.shape[0]),
            "prompt_truncated": False,
            "prompt_source": str(KVTUNER_EVALS / "gsm8k_utils.py"),
            "max_gen_source": "project multitask formal default",
            "seed": args.seed,
        })
    out = args.out_dir / f"gsm8k_first{args.limit}.jsonl"
    digest = write_jsonl(out, rows)
    return out, {
        "dataset": "gsm8k",
        "family": "math",
        "path": str(out),
        "sha256": digest,
        "num_rows": len(rows),
        "hf_dataset": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "metric": "gsm8k_exact",
        "max_gen": 512,
        "prompt_source": str(KVTUNER_EVALS / "gsm8k_utils.py"),
        "seed": args.seed,
    }


def prepare_math500(args: argparse.Namespace, tokenizer: Any) -> tuple[Path, dict[str, Any]]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                      cache_dir=str(args.math_cache_dir))
    if len(ds) < args.limit:
        raise SystemExit(f"MATH500 test has {len(ds)} rows < {args.limit}")
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    rows: list[dict[str, Any]] = []
    for idx, obj_raw in enumerate(ds.select(range(args.limit))):
        obj = dict(obj_raw)
        prompt = f"{obj['problem']}\n{instruction}"
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        rows.append({
            "family": "math",
            "dataset": "math500",
            "hf_dataset": "HuggingFaceH4/MATH-500",
            "hf_config": "default",
            "split": "test",
            "index": idx,
            "sample_id": str(obj.get("unique_id") or f"math500-{idx}"),
            "prompt": prompt,
            "answers": [str(obj["answer"])],
            "solution": str(obj.get("solution") or ""),
            "subject": str(obj.get("subject") or ""),
            "level": str(obj.get("level") or ""),
            "metric": "math_exact",
            "max_gen": 1024,
            "prompt_tokens": int(tokenized.shape[0]),
            "original_prompt_tokens": int(tokenized.shape[0]),
            "prompt_truncated": False,
            "prompt_source": "MATH500 CoT boxed formal prompt",
            "max_gen_source": "project multitask formal default",
        })
    out = args.out_dir / f"math500_first{args.limit}.jsonl"
    digest = write_jsonl(out, rows)
    return out, {
        "dataset": "math500",
        "family": "math",
        "path": str(out),
        "sha256": digest,
        "num_rows": len(rows),
        "hf_dataset": "HuggingFaceH4/MATH-500",
        "hf_config": "default",
        "split": "test",
        "metric": "math_exact",
        "max_gen": 1024,
        "prompt_source": "MATH500 CoT boxed formal prompt",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path),
                                              trust_remote_code=True,
                                              use_fast=False)
    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summaries = []
    for dataset in selected:
        if dataset in LONG_BENCH_DATASETS:
            _, summary = prepare_longbench(dataset, args, tokenizer)
        elif dataset == "math500":
            _, summary = prepare_math500(args, tokenizer)
        elif dataset == "gsm8k":
            _, summary = prepare_gsm8k(args, tokenizer)
        else:
            raise SystemExit(f"Unsupported dataset {dataset!r}")
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    manifest = {
        "limit": args.limit,
        "max_input_tokens": args.max_input_tokens,
        "tokenizer_path": str(args.tokenizer_path),
        "datasets": summaries,
    }
    manifest_path = args.out_dir / "manifest_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
