"""PM-KVQ fake-quant and cache-wide serving approximation."""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import (
    pmkvq_official_fake_quant, reference_source,
)
from vllm.attention.ops.atc_kv_fake_quant.common import (
    _bit_summary_from_counts, _float_env, _int_env, _iter_segments,
    _kv_cache_elements_per_token, _kv_cache_gather_tokens, _kv_cache_layout,
    _kv_cache_block_size, _kv_cache_scatter_tokens, _merge_pending_positions,
    _positions_are_full_prefill, _precision_summary,
    _precision_summary_from_counts, _sequence_key_from_block_table,
    _sequence_key_from_slots, _serving_target_bits, _slot_in_cache,
    _slot_positions_from_slot_mapping, _slots_from_block_table_positions,
    _valid_slot_ids, _covered_tokens_from_block_table,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import quantize_last_dim_groups
from vllm.attention.ops.atc_kv_fake_quant.trace import bit_summary

_PMKVQ_CACHEWIDE_LEDGER: dict[str, dict[int, int]] = {}
_PMKVQ_CACHEWIDE_SEQ_STATE: dict[str, dict[str, int]] = {}
_PMKVQ_CACHEWIDE_PENDING_POSITIONS: dict[str, dict[str, list[int]]] = {}

def _pmkvq(key, value, attn, serving, segments):
    # PM-KVQ: progressive mixed precision driven by a memory/bit budget, not
    # serving queue pressure.  This fake path simulates the paper's sink/window
    # high precision regions and progressive 16->8->4->2 bulk shrinking before
    # the normal FP16/BF16 vLLM cache write.
    sink = _int_env("ATC_PMKVQ_SINK_TOKENS", 1)
    window = _int_env("ATC_PMKVQ_WINDOW_TOKENS", 128)
    group_size = _int_env("ATC_PMKVQ_GROUP_SIZE", 128)
    init_bits = _int_env("ATC_PMKVQ_INIT_BITS", 16)
    min_bits = _int_env("ATC_PMKVQ_MIN_BITS", 2)
    budget_eval_len = _int_env("ATC_PMKVQ_BUDGET_EVAL_LEN", 0)
    prefill_window_mode = os.environ.get(
        "ATC_PMKVQ_PREFILL_WINDOW_MODE", "chunk").strip().lower()
    mode = os.environ.get("ATC_PMKVQ_MODE", "paper").lower()
    target_avg_bits = _float_env("ATC_PMKVQ_TARGET_AVG_BITS", 4.5)
    if mode == "serving":
        target_avg_bits = _serving_target_bits(
            target_avg_bits, serving, "ATC_PMKVQ")
    budget_mb = _pmkvq_layer_budget_mb(attn.layer_idx)
    budget_source = "target_avg_bits_fallback"
    spans = _iter_segments(segments, key.shape[0])
    qk = key.clone()
    qv = value.clone()
    bit_map = torch.full((key.shape[0],), 16, device=key.device,
                         dtype=torch.int16)
    bulk_bits_used: list[int] = []
    elements_per_token = key.shape[1] * key.shape[2] + value.shape[1] * value.shape[2]
    positioned = attn.cache_segment_positions
    if not positioned or len(positioned) != len(spans):
        positioned = [(start, end, 0, end - start) for start, end in spans]
    for start, end, context_len, seq_len in positioned:
        seg_len = end - start
        eval_len = max(int(seq_len), int(budget_eval_len))
        if budget_mb is not None:
            bulk_bits = _pmkvq_budget_bulk_bits_from_mb(
                eval_len, sink, window, init_bits, budget_mb,
                elements_per_token)
            budget_source = "official_budget_artifact"
        else:
            bulk_bits = _pmkvq_budget_bulk_bits(eval_len, sink, window,
                                                init_bits, target_avg_bits)
        bulk_bits = max(min_bits, int(bulk_bits))
        bulk_bits_used.append(int(bulk_bits))
        bit_map[start:end] = int(bulk_bits)
        local_positions = torch.arange(seg_len, device=key.device) + int(
            context_len)
        if sink > 0:
            bit_map[start:end] = torch.where(local_positions < sink,
                                             torch.full_like(
                                                 bit_map[start:end], 16),
                                             bit_map[start:end])
        effective_window = window
        if (seg_len > 1
                and prefill_window_mode in {"defer", "none", "off", "0"}):
            effective_window = 0
        if effective_window > 0:
            window_start = max(0, int(seq_len) - effective_window)
            bit_map[start:end] = torch.where(local_positions >= window_start,
                                             torch.full_like(
                                                 bit_map[start:end], 16),
                                             bit_map[start:end])
        qk[start:end] = _pmkvq_quantize_key(
            key[start:end], bit_map[start:end], group_size, attn.layer_idx)
        qv[start:end] = _pmkvq_quantize_by_token_bits(
            value[start:end], bit_map[start:end], group_size)
    selected = int(round(sum(bulk_bits_used) / max(1, len(bulk_bits_used))))
    summary = bit_summary(bit_map, selected)
    summary.update(_precision_summary(bit_map, bit_map, selected))
    summary.update({
        "sink_tokens": sink,
        "window_tokens": window,
        "prefill_window_mode": prefill_window_mode,
        "init_bits": init_bits,
        "min_bits": min_bits,
        "budget_eval_len": budget_eval_len,
        "target_avg_bits": target_avg_bits,
        "budget_mb": budget_mb,
        "budget_source": budget_source,
        "bulk_bits_used": bulk_bits_used,
        "sequence_segments": len(spans),
        "rep_scales": bool(_load_pmkvq_rep_scales(
            os.environ.get("ATC_PMKVQ_REP_SCALES_PATH", "").strip())),
        "rep_scales_path": os.environ.get("ATC_PMKVQ_REP_SCALES_PATH", ""),
        "mode": mode,
        "policy": f"pmkvq_{mode}_progressive_budget_fake_quant",
        **reference_source("pmkvq"),
    })
    return qk, qv, summary

def _pmkvq_serving(key, value, attn, serving, segments):
    previous = os.environ.get("ATC_PMKVQ_MODE")
    os.environ["ATC_PMKVQ_MODE"] = "serving"
    try:
        return _pmkvq(key, value, attn, serving, segments)
    finally:
        if previous is None:
            os.environ.pop("ATC_PMKVQ_MODE", None)
        else:
            os.environ["ATC_PMKVQ_MODE"] = previous

def _pmkvq_cachewide_after_write(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    block_size: int | None,
    attn,
    serving,
) -> dict[str, object]:
    """Rewrite historical vLLM cache slots to mimic PM-KVQ progression."""
    block = _kv_cache_block_size(key_cache, value_cache, block_size)
    layout = _kv_cache_layout(key_cache, value_cache, block)
    if layout == "unsupported":
        raise RuntimeError(
            f"unsupported KV cache layout key={tuple(key_cache.shape)} "
            f"value={tuple(value_cache.shape)} block={block}")
    positions = attn.cache_segment_positions
    block_tables = attn.cache_block_tables
    if not positions:
        raise RuntimeError("PM-KVQ cache-wide rewrite needs sequence positions")
    current_slot_ids = _valid_slot_ids(slot_mapping)
    current_slot_set = set(current_slot_ids)
    timing = os.environ.get(
        "ATC_PMKVQ_CACHEWIDE_TIMING",
        "defer_current",
    ).strip().lower()
    if timing in {"", "default", "safe"}:
        timing = "defer_current"
    defer_current = timing in {"defer_current", "history_only", "official_safe"}
    use_slot_mapping_prefill = False
    block_tables_cpu: torch.Tensor | None = None
    if isinstance(block_tables, torch.Tensor) and block_tables.numel() > 0:
        block_tables_cpu = block_tables.detach().to("cpu")
        if block_tables_cpu.ndim == 1:
            block_tables_cpu = block_tables_cpu.unsqueeze(0)
        if block_tables_cpu.shape[0] < len(positions):
            raise RuntimeError(
                f"block_tables rows {block_tables_cpu.shape[0]} < "
                f"sequence segments {len(positions)}")
    elif current_slot_ids and _positions_are_full_prefill(positions):
        use_slot_mapping_prefill = True
    else:
        raise RuntimeError("PM-KVQ cache-wide rewrite needs block_tables")

    sink = _int_env("ATC_PMKVQ_SINK_TOKENS", 1)
    sink_bits = _int_env("ATC_PMKVQ_SINK_BITS", 16)
    window = _int_env("ATC_PMKVQ_WINDOW_TOKENS", 128)
    window_bits = _int_env("ATC_PMKVQ_WINDOW_BITS", 16)
    group_size = _int_env("ATC_PMKVQ_GROUP_SIZE", 128)
    init_bits = _int_env("ATC_PMKVQ_INIT_BITS", 16)
    min_bits = _int_env("ATC_PMKVQ_MIN_BITS", 2)
    mode = os.environ.get("ATC_PMKVQ_MODE", "paper").lower()
    target_avg_bits = _float_env("ATC_PMKVQ_TARGET_AVG_BITS", 4.5)
    if mode == "serving":
        target_avg_bits = _serving_target_bits(
            target_avg_bits, serving, "ATC_PMKVQ")
    budget_mb = _pmkvq_layer_budget_mb(attn.layer_idx)
    budget_source = ("official_budget_artifact"
                     if budget_mb is not None else "target_avg_bits_fallback")
    elements_per_token = _kv_cache_elements_per_token(key_cache, value_cache,
                                                      layout)

    ledger_key = _pmkvq_cachewide_ledger_key(
        key_cache, value_cache, attn.layer_idx)
    ledger = _PMKVQ_CACHEWIDE_LEDGER.setdefault(ledger_key, {})
    seq_state = _PMKVQ_CACHEWIDE_SEQ_STATE.setdefault(ledger_key, {})
    pending_positions = _PMKVQ_CACHEWIDE_PENDING_POSITIONS.setdefault(
        ledger_key, {})
    for slot in current_slot_ids:
        ledger.pop(slot, None)

    expected_slots = 0
    covered_slots = 0
    rewritten_slots = 0
    rewritten_historical_slots = 0
    rewritten_current_slots = 0
    skipped_slots = 0
    deferred_current_slots = 0
    current_slot_candidates = 0
    invalid_slots = 0
    target_bit_counts: dict[int, int] = {}
    per_seq_bulk_bits: list[int] = []

    for row, (_start, _end, _context_len, seq_len) in enumerate(positions):
        seq_len = max(0, int(seq_len))
        if seq_len <= 0:
            continue
        expected_slots += seq_len
        bulk_bits, live_window = _pmkvq_cachewide_bit_params(
            seq_len, sink, sink_bits, window, window_bits, init_bits,
            min_bits, budget_mb, elements_per_token, target_avg_bits)
        if use_slot_mapping_prefill:
            local_start = int(positions[row][0])
            local_end = int(positions[row][1])
            local_slots = current_slot_ids[local_start:local_end]
            seq_key = _sequence_key_from_slots(local_slots, block)
            candidate_positions = list(range(seq_len))
            candidate_positions = _merge_pending_positions(
                candidate_positions, pending_positions.get(seq_key), seq_len)
            slot_positions = _slot_positions_from_slot_mapping(
                local_slots, candidate_positions, block, key_cache.shape[0])
            covered_for_seq = len([
                slot for slot in local_slots
                if _slot_in_cache(slot, block, key_cache.shape[0])
            ])
        else:
            assert block_tables_cpu is not None
            seq_key = _sequence_key_from_block_table(block_tables_cpu[row])
            previous_seq_len = seq_state.get(seq_key)
            candidate_positions = _pmkvq_candidate_positions(
                previous_seq_len, seq_len, sink, window, init_bits, min_bits,
                budget_mb, elements_per_token, target_avg_bits,
                key_cache.device)
            candidate_positions = _merge_pending_positions(
                candidate_positions, pending_positions.get(seq_key), seq_len)
            slot_positions = _slots_from_block_table_positions(
                block_tables_cpu[row], candidate_positions, block,
                key_cache.shape[0])
            covered_for_seq = _covered_tokens_from_block_table(
                block_tables_cpu[row], seq_len, block, key_cache.shape[0])
        covered_slots += covered_for_seq
        invalid_slots += max(0, seq_len - covered_for_seq)
        per_seq_bulk_bits.append(int(bulk_bits))
        _pmkvq_accumulate_bit_counts(target_bit_counts, seq_len, sink,
                                     sink_bits, live_window, window_bits,
                                     bulk_bits)
        rewrite_slots: list[int] = []
        rewrite_bits: list[int] = []
        current_deferred_positions: list[int] = []
        for pos, slot in slot_positions:
            bit = _pmkvq_bit_for_position(pos, seq_len, sink, sink_bits,
                                          live_window, window_bits, bulk_bits)
            if slot in current_slot_set:
                current_slot_candidates += 1
                if defer_current:
                    deferred_current_slots += 1
                    current_deferred_positions.append(int(pos))
                    ledger.pop(slot, None)
                    continue
            if ledger.get(slot) == bit:
                skipped_slots += 1
                continue
            rewrite_slots.append(slot)
            rewrite_bits.append(bit)
        if not rewrite_slots:
            if seq_key:
                seq_state[seq_key] = seq_len
                pending_positions[seq_key] = current_deferred_positions
            continue
        k_tokens, v_tokens = _kv_cache_gather_tokens(
            key_cache, value_cache, rewrite_slots, block, layout)
        rewrite_bit_map = torch.tensor(rewrite_bits,
                                       device=key_cache.device,
                                       dtype=torch.int16)
        qk = _pmkvq_quantize_key(k_tokens, rewrite_bit_map, group_size,
                                 attn.layer_idx)
        qv = _pmkvq_quantize_by_token_bits(
            v_tokens, rewrite_bit_map, group_size)
        _kv_cache_scatter_tokens(key_cache, value_cache, rewrite_slots, qk, qv,
                                 block, layout)
        for slot, bit in zip(rewrite_slots, rewrite_bits):
            ledger[slot] = bit
        rewritten_slots += len(rewrite_slots)
        rewritten_current = sum(1 for slot in rewrite_slots
                                if slot in current_slot_set)
        rewritten_current_slots += rewritten_current
        rewritten_historical_slots += len(rewrite_slots) - rewritten_current
        if seq_key:
            seq_state[seq_key] = seq_len
            pending_positions[seq_key] = current_deferred_positions

    coverage = covered_slots / max(1, expected_slots)
    selected = (int(round(sum(per_seq_bulk_bits) / len(per_seq_bulk_bits)))
                if per_seq_bulk_bits else min_bits)
    summary = _bit_summary_from_counts(target_bit_counts, selected)
    summary.update(_precision_summary_from_counts(target_bit_counts,
                                                  target_bit_counts,
                                                  selected))
    summary.update({
        "cache_wide_source": ("post_write_vllm_cache_rewrite_slot_mapping_prefill"
                              if use_slot_mapping_prefill else
                              "post_write_vllm_cache_rewrite_block_tables"),
        "cache_wide_layout": layout,
        "cache_wide_coverage": coverage,
        "cache_wide_expected_slots": expected_slots,
        "cache_wide_covered_slots": covered_slots,
        "rewritten_slots": rewritten_slots,
        "rewritten_historical_slots": rewritten_historical_slots,
        "rewritten_current_slots": rewritten_current_slots,
        "skipped_slots": skipped_slots,
        "deferred_current_slots": deferred_current_slots,
        "current_slot_candidates": current_slot_candidates,
        "current_slot_skip_rate": (
            deferred_current_slots / max(1, current_slot_candidates)
            if current_slot_candidates else 0.0),
        "invalid_slots": invalid_slots,
        "slot_target_bit_distribution": summary.get("bit_ratio", {}),
        "sink_tokens": sink,
        "sink_bits": sink_bits,
        "window_tokens": window,
        "window_bits": window_bits,
        "window_policy": "official_modulo_progressive",
        "cache_wide_timing": timing,
        "attention_safe_defer": bool(defer_current),
        "init_bits": init_bits,
        "min_bits": min_bits,
        "budget_mb": budget_mb,
        "budget_source": budget_source,
        "bulk_bits_used": per_seq_bulk_bits,
        "target_avg_bits": target_avg_bits,
        "mode": mode,
        "rep_scales": bool(_load_pmkvq_rep_scales(
            os.environ.get("ATC_PMKVQ_REP_SCALES_PATH", "").strip())),
        "rep_scales_path": os.environ.get("ATC_PMKVQ_REP_SCALES_PATH", ""),
        "policy": "pmkvq_cachewide_progressive_budget_fake_quant",
        **reference_source("pmkvq"),
    })
    return summary

def _pmkvq_cachewide_bit_params(
    seq_len: int,
    sink: int,
    sink_bits: int,
    window: int,
    window_bits: int,
    init_bits: int,
    min_bits: int,
    budget_mb: float | None,
    elements_per_token: int,
    target_avg_bits: float,
) -> tuple[int, int]:
    seq_len = max(0, int(seq_len))
    if seq_len <= 0:
        return int(min_bits), 0
    sink_live = min(seq_len, max(0, int(sink)))
    body_after_sink = max(0, seq_len - sink_live)
    live_window = (body_after_sink % max(1, int(window))
                   if int(window) > 0 else 0)
    bulk_tokens = max(0, seq_len - sink_live - live_window)
    if budget_mb is not None:
        bulk_bits = _pmkvq_cachewide_bulk_bits_from_mb(
            bulk_tokens, sink_live, live_window, sink_bits, window_bits,
            init_bits, min_bits, budget_mb, elements_per_token)
    else:
        bulk_bits = _pmkvq_cachewide_bulk_bits_from_target(
            bulk_tokens, sink_live, live_window, sink_bits, window_bits,
            init_bits, min_bits, target_avg_bits, seq_len)
    return int(bulk_bits), int(live_window)

def _pmkvq_cachewide_bit_map(
    seq_len: int,
    sink: int,
    sink_bits: int,
    window: int,
    window_bits: int,
    init_bits: int,
    min_bits: int,
    budget_mb: float | None,
    elements_per_token: int,
    target_avg_bits: float,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    seq_len = max(0, int(seq_len))
    if seq_len <= 0:
        return torch.empty(0, device=device, dtype=torch.int16), int(min_bits), 0
    sink_live = min(seq_len, max(0, int(sink)))
    bulk_bits, live_window = _pmkvq_cachewide_bit_params(
        seq_len, sink, sink_bits, window, window_bits, init_bits, min_bits,
        budget_mb, elements_per_token, target_avg_bits)
    bit_map = torch.full((seq_len,), int(bulk_bits), device=device,
                         dtype=torch.int16)
    if sink_live > 0:
        bit_map[:sink_live] = int(sink_bits)
    if live_window > 0:
        bit_map[seq_len - live_window:] = int(window_bits)
    return bit_map, int(bulk_bits), int(live_window)

def _pmkvq_bit_for_position(pos: int,
                            seq_len: int,
                            sink: int,
                            sink_bits: int,
                            live_window: int,
                            window_bits: int,
                            bulk_bits: int) -> int:
    pos = int(pos)
    seq_len = max(0, int(seq_len))
    if int(live_window) > 0 and pos >= seq_len - int(live_window):
        return int(window_bits)
    if pos < min(seq_len, max(0, int(sink))):
        return int(sink_bits)
    return int(bulk_bits)

def _pmkvq_accumulate_bit_counts(counts: dict[int, int],
                                 seq_len: int,
                                 sink: int,
                                 sink_bits: int,
                                 live_window: int,
                                 window_bits: int,
                                 bulk_bits: int) -> None:
    seq_len = max(0, int(seq_len))
    if seq_len <= 0:
        return
    live_window = max(0, min(seq_len, int(live_window)))
    window_start = seq_len - live_window
    sink_count = min(min(seq_len, max(0, int(sink))), window_start)
    bulk_count = max(0, seq_len - live_window - sink_count)
    for bit, count in ((sink_bits, sink_count), (bulk_bits, bulk_count),
                       (window_bits, live_window)):
        if count > 0:
            bit = int(bit)
            counts[bit] = counts.get(bit, 0) + int(count)

def _pmkvq_cachewide_bulk_bits_from_mb(
    bulk_tokens: int,
    sink_tokens: int,
    window_tokens: int,
    sink_bits: int,
    window_bits: int,
    init_bits: int,
    min_bits: int,
    budget_mb: float,
    elements_per_token: int,
) -> int:
    bits = max(int(min_bits), min(16, int(init_bits)))
    if bulk_tokens <= 0:
        return bits
    while bits > int(min_bits):
        weighted_bits = (sink_tokens * sink_bits + window_tokens * window_bits
                         + bulk_tokens * bits)
        memory_mb = elements_per_token * weighted_bits / (8 * 1024 * 1024)
        if memory_mb <= float(budget_mb):
            break
        bits //= 2
    return max(int(min_bits), int(bits))

def _pmkvq_cachewide_bulk_bits_from_target(
    bulk_tokens: int,
    sink_tokens: int,
    window_tokens: int,
    sink_bits: int,
    window_bits: int,
    init_bits: int,
    min_bits: int,
    target_avg_bits: float,
    seq_len: int,
) -> int:
    bits = max(int(min_bits), min(16, int(init_bits)))
    if bulk_tokens <= 0:
        return bits
    target_total = max(1.0, float(seq_len) * max(1.0, target_avg_bits))
    protected_total = sink_tokens * sink_bits + window_tokens * window_bits
    while bits > int(min_bits) and protected_total + bulk_tokens * bits > target_total:
        bits //= 2
    return max(int(min_bits), int(bits))

def _pmkvq_quantize_key(x: torch.Tensor, bit_map: torch.Tensor,
                        group_size: int, layer_idx: int) -> torch.Tensor:
    scale = _pmkvq_key_scale(layer_idx, x.device, x.dtype, x.shape)
    if scale is None:
        return _pmkvq_quantize_by_token_bits(x, bit_map, group_size)
    scaled = x / scale
    quantized = _pmkvq_quantize_by_token_bits(scaled, bit_map, group_size)
    return (quantized * scale).to(dtype=x.dtype)

def _pmkvq_quantize_by_token_bits(x: torch.Tensor, bit_map: torch.Tensor,
                                  group_size: int) -> torch.Tensor:
    """PM-KVQ per-token mixed bits using the official per-group quantizer."""
    out = torch.empty_like(x)
    for bits in sorted(set(int(v) for v in bit_map.detach().cpu().tolist())):
        mask = bit_map == bits
        if bits >= 16:
            out[mask] = x[mask]
            continue
        official, source = pmkvq_official_fake_quant(x[mask], bits)
        if source.startswith("official"):
            out[mask] = official
        else:
            out[mask] = quantize_last_dim_groups(x[mask], bits, group_size)
    return out

@lru_cache(maxsize=4)
def _load_pmkvq_rep_scales(path: str) -> list[torch.Tensor]:
    if not path:
        return []
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [x.detach().cpu() for x in obj if isinstance(x, torch.Tensor)]
    return []

def _pmkvq_key_scale(layer_idx: int, device: torch.device, dtype: torch.dtype,
                     key_shape: torch.Size) -> torch.Tensor | None:
    rep_scales_path = os.environ.get("ATC_PMKVQ_REP_SCALES_PATH", "").strip()
    rep_scales = _load_pmkvq_rep_scales(rep_scales_path)
    if not rep_scales or len(key_shape) < 3:
        return None
    idx = layer_idx if layer_idx >= 0 else 0
    if idx >= len(rep_scales):
        _pmkvq_rep_scale_mismatch(
            f"rep scales has {len(rep_scales)} layers, needs layer {idx}")
        return None
    scale = rep_scales[idx].to(device=device, dtype=torch.float32).clamp(1e-4, 1e4)
    if scale.ndim == 4 and scale.shape[0] == 1 and scale.shape[-2] == 1:
        scale = scale.squeeze(0).squeeze(-2)
    elif scale.ndim == 3 and scale.shape[-2] == 1:
        scale = scale.squeeze(-2)
    elif scale.ndim == 3 and scale.shape[0] == 1:
        scale = scale.squeeze(0)
    if scale.ndim > 2:
        _pmkvq_rep_scale_mismatch(
            f"unsupported rep scale rank {scale.ndim} for layer {idx}")
        return None
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    heads = int(key_shape[-2])
    head_size = int(key_shape[-1])
    if scale.shape[-2] not in {1, heads}:
        _pmkvq_rep_scale_mismatch(
            f"rep scale heads {scale.shape[-2]} incompatible with {heads}")
        return None
    if scale.shape[-1] * 2 == head_size:
        scale = scale.repeat(1, 2)
    elif scale.shape[-1] != head_size:
        _pmkvq_rep_scale_mismatch(
            f"rep scale dim {scale.shape[-1]} incompatible with {head_size}")
        return None
    if scale.shape[-2] == 1 and heads != 1:
        scale = scale.expand(heads, -1)
    return scale.to(dtype=dtype).unsqueeze(0)

def _pmkvq_rep_scale_mismatch(message: str) -> None:
    if os.environ.get("ATC_PMKVQ_REP_SCALES_STRICT", "1") != "0":
        raise RuntimeError(f"PM-KVQ rep-scale mismatch: {message}")

def _pmkvq_budget_bulk_bits(num_tokens: int, sink: int, window: int,
                            init_bits: int, target_avg_bits: float) -> int:
    protected = min(max(0, num_tokens), max(0, sink) + max(0, window))
    body = max(0, num_tokens - protected)
    if body == 0:
        return max(2, min(16, init_bits))
    target_total = max(1.0, float(num_tokens) * max(1.0, target_avg_bits))
    protected_total = float(protected) * 16.0
    bits = max(2, min(16, init_bits))
    while bits > 2 and protected_total + body * bits > target_total:
        bits //= 2
    return int(bits)

@lru_cache(maxsize=1)
def _load_pmkvq_budgets() -> list[float]:
    path = os.environ.get("ATC_PMKVQ_BUDGET_PATH", "").strip()
    if not path:
        return []
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return []
    if isinstance(obj, torch.Tensor):
        return [float(v) for v in obj.flatten().tolist()]
    if isinstance(obj, (list, tuple)):
        return [float(v) for v in obj]
    return []

def _pmkvq_layer_budget_mb(layer_idx: int) -> float | None:
    budgets = _load_pmkvq_budgets()
    if not budgets:
        return None
    idx = layer_idx if layer_idx >= 0 else 0
    if idx >= len(budgets):
        return None
    return float(budgets[idx])

def _pmkvq_budget_bulk_bits_from_mb(num_tokens: int, sink: int, window: int,
                                    init_bits: int, budget_mb: float,
                                    elements_per_token: int) -> int:
    protected = min(max(0, num_tokens), max(0, sink) + max(0, window))
    body = max(0, num_tokens - protected)
    bits = max(2, min(16, init_bits))
    if body == 0:
        return bits
    while bits > 2:
        weighted_bits = protected * 16 + body * bits
        memory_mb = elements_per_token * weighted_bits / (8 * 1024 * 1024)
        if memory_mb <= budget_mb:
            break
        bits //= 2
    return int(bits)

def _pmkvq_candidate_positions(
    previous_seq_len: int | None,
    seq_len: int,
    sink: int,
    window: int,
    init_bits: int,
    min_bits: int,
    budget_mb: float | None,
    elements_per_token: int,
    target_avg_bits: float,
    device: torch.device,
) -> list[int]:
    seq_len = max(0, int(seq_len))
    if previous_seq_len is None or previous_seq_len <= 0 or previous_seq_len > seq_len:
        return list(range(seq_len))
    previous_seq_len = int(previous_seq_len)
    old_bulk, old_live = _pmkvq_cachewide_bit_params(
        previous_seq_len, sink, 16, window, 16, init_bits, min_bits, budget_mb,
        elements_per_token, target_avg_bits)
    new_bulk, new_live = _pmkvq_cachewide_bit_params(
        seq_len, sink, 16, window, 16, init_bits, min_bits, budget_mb,
        elements_per_token, target_avg_bits)
    if old_bulk != new_bulk:
        return list(range(seq_len))
    positions: set[int] = set(range(previous_seq_len, seq_len))
    if old_live > 0:
        positions.update(range(max(0, previous_seq_len - old_live),
                               previous_seq_len))
    if new_live > 0:
        positions.update(range(max(0, seq_len - new_live), seq_len))
    sink_live = min(seq_len, max(0, int(sink)))
    positions.update(range(sink_live))
    return [p for p in sorted(positions) if 0 <= p < seq_len]

def _pmkvq_cachewide_ledger_key(key_cache: torch.Tensor,
                                value_cache: torch.Tensor,
                                layer_idx: int) -> str:
    return (f"{key_cache.device}:{key_cache.data_ptr()}:"
            f"{value_cache.data_ptr()}:{int(layer_idx)}")
