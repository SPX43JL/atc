#!/usr/bin/env python3
"""Summarize multitask fake-quant serving results and trace bit stats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASET_ORDER = {
    name: i for i, name in enumerate([
        "qasper",
        "narrativeqa",
        "hotpotqa",
        "passage_retrieval_en",
        "passage_count",
        "qmsum",
        "math500",
        "gsm8k",
    ])
}
METHOD_ORDER = {
    name: i for i, name in enumerate([
        "baseline",
        "kivi2",
        "kivi4",
        "kvtuner_pertoken_c4_00",
        "kvtuner_kivi_c3_92",
        "kvtuner",
        "kvquant",
        "pmkvq_cachewide_mem132",
    ])
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj["summary"]
    examples = obj.get("examples") or []
    ok = [x for x in examples if int(x.get("status") or 0) == 200]
    trunc = sum(1 for x in ok if x.get("hit_max_tokens"))
    prompt_trunc = sum(1 for x in examples if x.get("prompt_truncated"))
    return {
        "file": str(path),
        "dataset": summary.get("dataset"),
        "family": summary.get("family"),
        "metric": summary.get("metric"),
        "method": summary.get("method"),
        "method_variant": summary.get("method_variant") or summary.get("method"),
        "workload": summary.get("workload"),
        "endpoint": summary.get("endpoint"),
        "max_concurrency": summary.get("max_concurrency"),
        "num_examples": summary.get("num_examples"),
        "ok": summary.get("ok"),
        "failed": summary.get("failed"),
        "score_pct": summary.get("score_pct"),
        "truncate_count": trunc,
        "truncate_rate": summary.get("truncate_rate"),
        "prompt_truncate_count": prompt_trunc,
        "prompt_truncate_rate": summary.get("prompt_truncate_rate"),
        "avg_latency_s": summary.get("avg_latency_s"),
        "p50_latency_s": summary.get("p50_latency_s"),
        "p95_latency_s": summary.get("p95_latency_s"),
        "avg_queue_wait_s": summary.get("avg_queue_wait_s"),
        "throughput_req_s": summary.get("throughput_req_s"),
        "total_elapsed_s": summary.get("total_elapsed_s"),
        "avg_completion_tokens": summary.get("avg_completion_tokens"),
        "avg_prompt_tokens": summary.get("avg_prompt_tokens"),
        "avg_official_prompt_tokens": summary.get("avg_official_prompt_tokens"),
        "max_gen": summary.get("max_gen"),
        "temperature": summary.get("temperature"),
        "manifest": summary.get("manifest"),
    }




def _include_kvquant_sparse_outlier_effective_bits(
        row: dict[str, Any]) -> dict[str, Any]:
    if row.get("kvquant_sparse_outlier_bits_included"):
        return row
    try:
        ratio = max(0.0, min(1.0, float(row.get("outlier_ratio") or 0.0)))
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio <= 0.0:
        return row
    outlier_bits = 16
    row = dict(row)

    def adjusted_avg(counts_obj: object, fallback: object) -> float | None:
        if not isinstance(counts_obj, dict):
            return float(fallback) if isinstance(fallback, (int, float)) else None
        total = sum(int(v) for v in counts_obj.values())
        if total <= 0:
            return float(fallback) if isinstance(fallback, (int, float)) else None
        acc = 0.0
        for bit_obj, count_obj in counts_obj.items():
            bit = int(float(bit_obj))
            count = int(count_obj)
            if bit >= outlier_bits:
                acc += bit * count
            else:
                acc += ((1.0 - ratio) * bit + ratio * outlier_bits) * count
        return acc / total

    def adjusted_dist(dist_obj: object) -> dict[str, float] | None:
        if not isinstance(dist_obj, dict):
            return None
        out: dict[str, float] = {}
        for bit_obj, frac_obj in dist_obj.items():
            bit = int(float(bit_obj))
            frac = float(frac_obj)
            if bit >= outlier_bits:
                out[str(bit)] = out.get(str(bit), 0.0) + frac
            else:
                out[str(bit)] = out.get(str(bit), 0.0) + frac * (1.0 - ratio)
                out[str(outlier_bits)] = out.get(str(outlier_bits), 0.0) + frac * ratio
        total = sum(out.values()) or 1.0
        return {bit: val / total
                for bit, val in sorted(out.items(), key=lambda kv: int(kv[0]))}

    for src, dst in (("avg_k_bits", "kvquant_nominal_avg_k_bits_without_sparse_outlier"),
                     ("avg_v_bits", "kvquant_nominal_avg_v_bits_without_sparse_outlier"),
                     ("avg_kv_bits", "kvquant_nominal_avg_kv_bits_without_sparse_outlier")):
        row[dst] = row.get(src)
    row["kvquant_nominal_precision_distribution_without_sparse_outlier"] = (
        row.get("precision_distribution"))
    avg_k = adjusted_avg(row.get("k_bit_counts"), row.get("avg_k_bits"))
    avg_v = adjusted_avg(row.get("v_bit_counts"), row.get("avg_v_bits"))
    if avg_k is not None:
        row["avg_k_bits"] = avg_k
    if avg_v is not None:
        row["avg_v_bits"] = avg_v
    if avg_k is not None and avg_v is not None:
        row["avg_kv_bits"] = (avg_k + avg_v) / 2.0
    dist = adjusted_dist(row.get("precision_distribution"))
    if dist is not None:
        row["precision_distribution"] = dist
    row["kvquant_sparse_outlier_bits_included"] = True
    row["kvquant_sparse_outlier_ratio"] = ratio
    row["kvquant_sparse_outlier_bits"] = outlier_bits
    return row


def read_trace_bits(trace_dir: Path, variant: str, method: str,
                    dataset: str) -> dict[str, Any]:
    if method in {"none", "baseline"} or variant == "baseline":
        return {
            "avg_k_bits": "BF16",
            "avg_v_bits": "BF16",
            "avg_kv_bits": "BF16",
            "precision_distribution": "BF16/FP16",
            "trace_records": 0,
        }
    candidates = [
        trace_dir / f"{variant}_kv_fake_quant.jsonl",
        trace_dir / f"{method}_kv_fake_quant.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("dataset") and obj.get("dataset") != dataset:
                continue
            if obj.get("method") and method and obj.get("method") != method:
                continue
            if method == "kvquant":
                obj = _include_kvquant_sparse_outlier_effective_bits(obj)
            rows.append(obj)
    if not rows:
        return {"trace_records": 0}

    metric_rows = choose_precision_rows(rows, method)
    out: dict[str, Any] = {"trace_records": len(rows),
                           "precision_trace_records": len(metric_rows)}
    for key in ("avg_k_bits", "avg_v_bits", "avg_kv_bits"):
        weighted = [
            (float(row[key]), trace_weight(method, row))
            for row in metric_rows
            if isinstance(row.get(key), (int, float))
        ]
        if weighted:
            denom = sum(weight for _, weight in weighted)
            out[key] = sum(value * weight for value, weight in weighted) / max(1, denom)
    merged: dict[str, float] = {}
    denom = 0
    for row in metric_rows:
        dist = row.get("precision_distribution")
        if not isinstance(dist, dict):
            continue
        weight = trace_weight(method, row)
        denom += weight
        for bit, ratio in dist.items():
            merged[str(bit)] = merged.get(str(bit), 0.0) + float(ratio) * weight
    if denom:
        out["precision_distribution"] = {
            bit: val / denom
            for bit, val in sorted(merged.items(), key=lambda kv: float(kv[0]))
        }
    coverage = [
        (float(row["cache_wide_coverage"]), trace_weight(method, row))
        for row in metric_rows
        if isinstance(row.get("cache_wide_coverage"), (int, float))
    ]
    if coverage:
        denom = sum(weight for _, weight in coverage)
        out["pm_cache_wide_coverage"] = (
            sum(value * weight for value, weight in coverage) / max(1, denom))
    rewritten = [
        int(row.get("rewritten_slots") or 0)
        for row in metric_rows
        if "rewritten_slots" in row
    ]
    if rewritten:
        out["pm_rewritten_slots"] = sum(rewritten)
    return out


def choose_precision_rows(rows: list[dict[str, Any]],
                          method: str) -> list[dict[str, Any]]:
    if method in {"pmkvq_cachewide", "kivi", "kvtuner", "mixkvq",
                  "mixkvq_serving"}:
        cache_rows = [
            row for row in rows
            if int(row.get("cache_wide_expected_slots") or 0) > 0
            or int(row.get("cache_wide_total_slots") or 0) > 0
        ]
        return cache_rows or rows
    if method in {"pmkvq", "pmkvq_serving"}:
        prefill_rows = [
            row for row in rows
            if int(row.get("num_tokens") or 0) > int(row.get("window_tokens") or 128)
        ]
        return prefill_rows or rows
    return rows


def trace_weight(method: str, row: dict[str, Any]) -> int:
    if method in {"pmkvq_cachewide", "kivi", "kvtuner", "mixkvq",
                  "mixkvq_serving"}:
        return max(1, int(row.get("cache_wide_expected_slots")
                          or row.get("cache_wide_total_slots") or 1))
    return max(1, int(row.get("num_tokens") or 1))


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_rate(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{100 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_dist(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(
            f"{bit}:{float(ratio):.2f}" for bit, ratio in value.items())
    return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name.startswith("summary") or "smoke" in path.name:
            continue
        try:
            row = load_result(path)
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if row.get("workload") == "burst":
            rows.append(row)
    for row in rows:
        row.update(read_trace_bits(
            args.trace_dir,
            str(row.get("method_variant") or row.get("method")),
            str(row.get("method") or ""),
            str(row.get("dataset") or ""),
        ))
    rows.sort(key=lambda row: (
        DATASET_ORDER.get(str(row.get("dataset")), 999),
        METHOD_ORDER.get(str(row.get("method_variant")), 999),
        str(row.get("method_variant")),
    ))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "results_dir": str(args.results_dir),
        "trace_dir": str(args.trace_dir),
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_fields = [
        "dataset", "metric", "method_variant", "score_pct", "ok",
        "num_examples", "failed", "truncate_count", "truncate_rate",
        "prompt_truncate_count", "avg_latency_s", "p95_latency_s",
        "total_elapsed_s", "throughput_req_s", "avg_k_bits",
        "avg_v_bits", "avg_kv_bits", "precision_distribution",
        "pm_cache_wide_coverage", "file",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in csv_fields}
            out["precision_distribution"] = fmt_dist(out.get("precision_distribution"))
            writer.writerow(out)

    lines = [
        "# Multitask vLLM KV Fake Quant Formal Report",
        "",
        "Python-level fake quant only. These runs do not claim packed low-bit KV cache, real memory saving, or kernel acceleration.",
        "",
        "## Results",
        "",
        "| dataset | metric | method | score | ok/total | failed | trunc | prompt trunc | avg latency(s) | p95 latency(s) | total elapsed(s) | throughput(req/s) | avg_k_bits | avg_v_bits | avg_kv_bits | precision_distribution | PM cache-wide coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        total = row.get("num_examples") or 0
        ok = row.get("ok") or 0
        lines.append(
            f"| {row.get('dataset')} | {row.get('metric')} | "
            f"{row.get('method_variant')} | {fmt(row.get('score_pct'))} | "
            f"{ok}/{total} | {row.get('failed')} | "
            f"{row.get('truncate_count')} ({fmt_rate(row.get('truncate_rate'))}) | "
            f"{row.get('prompt_truncate_count')} ({fmt_rate(row.get('prompt_truncate_rate'))}) | "
            f"{fmt(row.get('avg_latency_s'), 3)} | "
            f"{fmt(row.get('p95_latency_s'), 3)} | "
            f"{fmt(row.get('total_elapsed_s'), 1)} | "
            f"{fmt(row.get('throughput_req_s'), 3)} | "
            f"{fmt(row.get('avg_k_bits'))} | "
            f"{fmt(row.get('avg_v_bits'))} | "
            f"{fmt(row.get('avg_kv_bits'))} | "
            f"{fmt_dist(row.get('precision_distribution'))} | "
            f"{fmt(row.get('pm_cache_wide_coverage'), 4)} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- LongBench prompts and max generation lengths come from the KIVI/LongBench config files under `references/kv_methods/KIVI/config`.",
        "- MATH500 and GSM8K are manifest-based project extensions for the same serving/fake-quant path; they are not LongBench tasks.",
        "- Bit statistics are read from fake-quant trace JSONL files. Blank bit fields mean no trace was emitted for that method/dataset.",
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
