"""JSONL tracing for fake-quant precision decisions."""

from __future__ import annotations

import json
import os
from collections import Counter

import torch

from vllm.attention.ops.atc_kv_fake_quant.runtime import (AttentionContext,
                                                          ServingState, now)


_CALL_COUNT = 0


def should_trace(method: str) -> bool:
    if method in {"pmkvq", "pmkvq_cachewide", "mixkvq"}:
        return True
    every = int(os.environ.get("ATC_KV_FAKE_QUANT_LOG_EVERY", "0") or 0)
    return every > 0


def emit_trace(method: str, attn: AttentionContext, serving: ServingState,
               key: torch.Tensor, bits: dict[str, object]) -> None:
    global _CALL_COUNT
    _CALL_COUNT += 1
    path = os.environ.get("ATC_KV_FAKE_QUANT_LOG_PATH", "")
    if not path:
        return
    every = int(os.environ.get("ATC_KV_FAKE_QUANT_LOG_EVERY", "1") or 1)
    if method not in {"pmkvq", "pmkvq_cachewide", "mixkvq"} and _CALL_COUNT % max(1, every):
        return
    record = {
        "time": now(),
        "call": _CALL_COUNT,
        "method": method,
        "request_id": serving.request_id,
        "workload": serving.workload,
        "dataset": serving.dataset,
        "running_batch_size": serving.inflight,
        "queue_length": serving.queue_length,
        "max_concurrency": serving.max_concurrency,
        "estimated_pressure": serving.pressure,
        "layer_name": attn.layer_name,
        "layer_idx": attn.layer_idx,
        "attn_type": attn.attn_type,
        "num_tokens": int(key.shape[0]) if key.ndim >= 1 else 1,
        "num_kv_heads": int(key.shape[1]) if key.ndim >= 2 else 1,
        "head_size": int(key.shape[-1]) if key.ndim >= 1 else 1,
        "dtype": str(key.dtype),
        "device": str(key.device),
        **bits,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def bit_summary(tensor: torch.Tensor, default_bits: int) -> dict[str, object]:
    if tensor.numel() == 0:
        return {"selected_bit_width": default_bits, "bit_ratio": {}}
    values = [int(v) for v in tensor.detach().flatten().cpu().tolist()]
    counts = Counter(values)
    total = sum(counts.values()) or 1
    avg_bits = sum(k * v for k, v in counts.items()) / total
    return {
        "selected_bit_width": int(round(avg_bits)),
        "avg_bits": avg_bits,
        "bit_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "bit_ratio": {str(k): v / total for k, v in sorted(counts.items())},
    }
