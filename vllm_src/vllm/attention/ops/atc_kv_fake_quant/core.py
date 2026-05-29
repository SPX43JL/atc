"""Unified entry points for ATC KV cache fake quantization.

Method-specific logic lives in ``methods/`` and shared tensor/cache helpers live
in ``common.py``.  This module intentionally keeps the public hook surface that
vLLM calls: pre-write fake quant, post-write cache-wide fake quant, and the
Qwen2 KVQuant pre-RoPE key hook.
"""

from __future__ import annotations

import os

import torch

from vllm.attention.ops.atc_kv_fake_quant.common import (
    _actual_num_tokens, _kv_cache_gather_tokens, _kv_cache_scatter_tokens,
    _sequence_segments,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.kivi import _kivi
from vllm.attention.ops.atc_kv_fake_quant.methods.kvquant import (
    _kvquant, _load_kvquant_nuq_artifact_from_path,
    maybe_kvquant_prerope_key,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.kvtuner import (
    _kvtuner, _kvtuner_quant_mode, _load_kvtuner_config,
    _load_kvtuner_config_from_path,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.mixkvq import (
    _load_mixkvq_thresholds, _mixkvq, _mixkvq_assign_bits, _mixkvq_serving,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.pmkvq import (
    _PMKVQ_CACHEWIDE_LEDGER, _PMKVQ_CACHEWIDE_PENDING_POSITIONS,
    _PMKVQ_CACHEWIDE_SEQ_STATE, _pmkvq, _pmkvq_cachewide_after_write,
    _pmkvq_key_scale, _pmkvq_serving,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.residual_cachewide import (
    _RESIDUAL_CACHEWIDE_LEDGER, _RESIDUAL_CACHEWIDE_PENDING_POSITIONS,
    _RESIDUAL_CACHEWIDE_SEQ_STATE, _residual_cachewide_after_write,
)
from vllm.attention.ops.atc_kv_fake_quant.runtime import (
    attention_context, current_attention_context, load_serving_state,
)
from vllm.attention.ops.atc_kv_fake_quant.trace import emit_trace

SUPPORTED_METHODS = {
    "none", "kivi", "kvtuner", "kvquant", "pmkvq", "mixkvq",
    "pmkvq_serving", "pmkvq_cachewide", "mixkvq_serving"
}

_CACHEWIDE_RESIDUAL_METHODS = {
    "kivi", "kvtuner", "mixkvq", "mixkvq_serving"
}

_METHODS = {
    "kivi": _kivi,
    "kvtuner": _kvtuner,
    "kvquant": _kvquant,
    "pmkvq": _pmkvq,
    "pmkvq_serving": _pmkvq_serving,
    "mixkvq": _mixkvq,
    "mixkvq_serving": _mixkvq_serving,
}


def _normalize_method() -> str:
    method = os.environ.get("ATC_KV_FAKE_QUANT_METHOD", "none").strip().lower()
    if method in {"", "off", "false", "0", "baseline"}:
        method = "none"
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown ATC_KV_FAKE_QUANT_METHOD={method!r}")
    return method


def maybe_fake_quant_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    slot_mapping: torch.Tensor | None = None,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    method = _normalize_method()
    if method == "none" or key.numel() == 0 or value.numel() == 0:
        return key, value
    if kv_cache_dtype not in {"auto", "half", "float16", "bfloat16"}:
        return key, value
    if method == "pmkvq_cachewide" or method in _CACHEWIDE_RESIDUAL_METHODS:
        return key, value

    num_tokens = _actual_num_tokens(key, slot_mapping)
    attn = current_attention_context()
    serving = load_serving_state(num_tokens)
    segments = attn.cache_segments or _sequence_segments(slot_mapping, num_tokens,
                                                         block_size)
    with torch.no_grad():
        q_key, q_value, trace_bits = _METHODS[method](key, value, attn,
                                                      serving, segments)
    emit_trace(method, attn, serving, key, trace_bits)
    return q_key, q_value


def maybe_cachewide_fake_quant_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None = None,
    block_size: int | None = None,
) -> None:
    method = _normalize_method()
    if method != "pmkvq_cachewide" and method not in _CACHEWIDE_RESIDUAL_METHODS:
        return
    if key.numel() == 0 or value.numel() == 0:
        return
    num_tokens = _actual_num_tokens(key, slot_mapping)
    attn = current_attention_context()
    serving = load_serving_state(num_tokens)
    if method in _CACHEWIDE_RESIDUAL_METHODS:
        strict = os.environ.get("ATC_CACHEWIDE_RESIDUAL_STRICT", "1") != "0"
        try:
            with torch.no_grad():
                trace_bits = _residual_cachewide_after_write(
                    method, key_cache, value_cache, slot_mapping, block_size,
                    attn, serving)
        except Exception as exc:
            trace_bits = {
                "cache_wide_source": "error",
                "cache_wide_error": f"{type(exc).__name__}: {exc}",
                "cache_wide_coverage": 0.0,
                "cachewide_residual_coverage": 0.0,
                "rewritten_slots": 0,
                "skipped_slots": 0,
            }
            emit_trace(method, attn, serving, key, trace_bits)
            if strict:
                raise
            return
        emit_trace(method, attn, serving, key, trace_bits)
        return

    strict = os.environ.get("ATC_PMKVQ_CACHEWIDE_STRICT", "1") != "0"
    try:
        with torch.no_grad():
            trace_bits = _pmkvq_cachewide_after_write(
                key_cache, value_cache, slot_mapping, block_size, attn,
                serving)
    except Exception as exc:
        from vllm.attention.ops.atc_kv_fake_quant.adapters import reference_source
        trace_bits = {
            "cache_wide_source": "error",
            "cache_wide_error": f"{type(exc).__name__}: {exc}",
            "cache_wide_coverage": 0.0,
            "rewritten_slots": 0,
            "skipped_slots": 0,
            **reference_source("pmkvq"),
        }
        emit_trace(method, attn, serving, key, trace_bits)
        if strict:
            raise
        return
    emit_trace(method, attn, serving, key, trace_bits)


def _clear_pmkvq_cachewide_state() -> None:
    _PMKVQ_CACHEWIDE_LEDGER.clear()
    _PMKVQ_CACHEWIDE_SEQ_STATE.clear()
    _PMKVQ_CACHEWIDE_PENDING_POSITIONS.clear()
    _RESIDUAL_CACHEWIDE_LEDGER.clear()
    _RESIDUAL_CACHEWIDE_SEQ_STATE.clear()
    _RESIDUAL_CACHEWIDE_PENDING_POSITIONS.clear()


__all__ = [
    "maybe_fake_quant_kv",
    "maybe_cachewide_fake_quant_kv",
    "maybe_kvquant_prerope_key",
    "attention_context",
]
