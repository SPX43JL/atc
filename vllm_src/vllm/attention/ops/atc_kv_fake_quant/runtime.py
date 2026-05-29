"""Runtime context for Python-only ATC KV fake quant experiments."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import torch


_TLS = threading.local()


@dataclass
class AttentionContext:
    layer_name: str = ""
    layer_idx: int = -1
    attn_type: str = ""
    query: torch.Tensor | None = None
    cache_segments: list[tuple[int, int]] | None = None
    cache_segment_positions: list[tuple[int, int, int, int]] | None = None
    cache_block_tables: torch.Tensor | None = None


@dataclass
class ServingState:
    request_id: int | None = None
    workload: str = "unknown"
    dataset: str = ""
    max_concurrency: int = 1
    inflight: int = 0
    queue_length: int = 0
    pressure: str = "low"
    updated_at: float = 0.0


def _parse_layer_idx(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", layer_name)
    return int(numbers[-1]) if numbers else -1


@contextlib.contextmanager
def attention_context(layer_name: str, query: torch.Tensor | None,
                      attn_type: str, attn_metadata: object | None = None
                      ) -> Iterator[None]:
    previous = getattr(_TLS, "attention_context", None)
    _TLS.attention_context = AttentionContext(
        layer_name=layer_name,
        layer_idx=_parse_layer_idx(layer_name),
        attn_type=attn_type,
        query=query.detach() if isinstance(query, torch.Tensor) else None,
        cache_segments=_metadata_query_segments(attn_metadata, query),
        cache_segment_positions=_metadata_query_segment_positions(attn_metadata,
                                                                  query),
        cache_block_tables=_metadata_block_tables(attn_metadata),
    )
    try:
        yield
    finally:
        _TLS.attention_context = previous


def current_attention_context() -> AttentionContext:
    ctx = getattr(_TLS, "attention_context", None)
    return ctx if isinstance(ctx, AttentionContext) else AttentionContext()


def _metadata_query_segments(attn_metadata: object | None,
                             query: torch.Tensor | None
                             ) -> list[tuple[int, int]] | None:
    positioned = _metadata_query_segment_positions(attn_metadata, query)
    if positioned:
        return [(start, end) for start, end, _, _ in positioned]
    if attn_metadata is None or not isinstance(query, torch.Tensor):
        return None
    total = int(query.shape[0]) if query.ndim >= 1 else 0
    if total <= 0:
        return None
    starts_obj = getattr(attn_metadata, "query_start_loc", None)
    if starts_obj is None:
        starts_obj = getattr(attn_metadata, "seq_start_loc", None)
    if not isinstance(starts_obj, torch.Tensor) or starts_obj.numel() < 2:
        return None
    try:
        starts = [int(v) for v in starts_obj.detach().flatten().to("cpu").tolist()]
    except Exception:
        return None
    spans: list[tuple[int, int]] = []
    for start, end in zip(starts[:-1], starts[1:]):
        start = max(0, min(total, start))
        end = max(0, min(total, end))
        if start < end:
            spans.append((start, end))
    if not spans:
        return None
    return spans


def _metadata_query_segment_positions(
        attn_metadata: object | None,
        query: torch.Tensor | None) -> list[tuple[int, int, int, int]] | None:
    """Return local spans plus global context/sequence lengths.

    Tuples are ``(local_start, local_end, context_len_before_chunk,
    sequence_len_after_chunk)``.  PM-KVQ uses this to avoid resetting its
    sink/window protection on every vLLM chunked-prefill write.
    """
    if attn_metadata is None or not isinstance(query, torch.Tensor):
        return None
    total = int(query.shape[0]) if query.ndim >= 1 else 0
    if total <= 0:
        return None

    starts_obj = getattr(attn_metadata, "query_start_loc", None)
    starts: list[int] | None = None
    if isinstance(starts_obj, torch.Tensor) and starts_obj.numel() >= 2:
        try:
            starts = [
                int(v)
                for v in starts_obj.detach().flatten().to("cpu").tolist()
            ]
        except Exception:
            starts = None
    seq_lens = _metadata_int_list(getattr(attn_metadata, "seq_lens_tensor", None))
    if not seq_lens:
        raw_seq_lens = getattr(attn_metadata, "seq_lens", None)
        if isinstance(raw_seq_lens, list):
            seq_lens = [int(v) for v in raw_seq_lens]
    context_lens = _metadata_int_list(
        getattr(attn_metadata, "context_lens_tensor", None))

    if starts is None:
        if seq_lens and len(seq_lens) == total:
            spans = []
            for i, seq_len in enumerate(seq_lens):
                spans.append((i, i + 1, max(0, int(seq_len) - 1),
                              max(1, int(seq_len))))
            return spans
        return None

    spans: list[tuple[int, int, int, int]] = []
    for idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        start = max(0, min(total, int(start)))
        end = max(0, min(total, int(end)))
        if start >= end:
            continue
        q_len = end - start
        seq_len = int(seq_lens[idx]) if idx < len(seq_lens) else q_len
        context_len = (int(context_lens[idx]) if idx < len(context_lens) else
                       max(0, seq_len - q_len))
        spans.append((start, end, max(0, context_len), max(q_len, seq_len)))
    return spans or None


def _metadata_int_list(obj: object | None) -> list[int]:
    if isinstance(obj, torch.Tensor) and obj.numel() > 0:
        try:
            return [int(v) for v in obj.detach().flatten().to("cpu").tolist()]
        except Exception:
            return []
    return []


def _metadata_block_tables(attn_metadata: object | None) -> torch.Tensor | None:
    if attn_metadata is None:
        return None
    block_tables = getattr(attn_metadata, "block_tables", None)
    if isinstance(block_tables, torch.Tensor) and block_tables.numel() > 0:
        return block_tables.detach()
    return None


def load_serving_state(num_tokens: int) -> ServingState:
    forced = os.environ.get("ATC_KV_FAKE_QUANT_PRESSURE", "auto").lower()
    path = os.environ.get("ATC_SERVING_STATE_PATH", "")
    data: dict = {}
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError:
            data = {}
        except json.JSONDecodeError:
            data = {}

    max_concurrency = int(data.get("max_concurrency") or 1)
    inflight = int(data.get("inflight") or 0)
    queue_length = int(data.get("queue_length") or 0)
    pressure = _estimate_pressure(forced, max_concurrency, inflight,
                                  queue_length, num_tokens)
    return ServingState(
        request_id=data.get("request_id"),
        workload=str(data.get("workload") or "unknown"),
        dataset=str(data.get("dataset") or ""),
        max_concurrency=max_concurrency,
        inflight=inflight,
        queue_length=queue_length,
        pressure=pressure,
        updated_at=float(data.get("updated_at") or 0.0),
    )


def _estimate_pressure(forced: str, max_concurrency: int, inflight: int,
                       queue_length: int, num_tokens: int) -> str:
    if forced in {"low", "medium", "high"}:
        return forced
    max_concurrency = max(1, max_concurrency)
    ratio = (inflight + queue_length) / max_concurrency
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.45:
        return "medium"
    # During prefill, many tokens arrive in one cache write. Treat that as at
    # least medium pressure so dynamic methods exercise their serving policy.
    if num_tokens >= int(os.environ.get("ATC_KV_FAKE_QUANT_HIGH_TOKENS", "512")):
        return "high"
    if num_tokens >= int(os.environ.get("ATC_KV_FAKE_QUANT_MEDIUM_TOKENS", "128")):
        return "medium"
    return "low"


def now() -> float:
    return time.time()
