"""Cache-wide residual rewrites for KIVI/KVTuner/MixKVQ."""

from __future__ import annotations

import os

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import reference_source
from vllm.attention.ops.atc_kv_fake_quant.common import (
    _add_count, _covered_tokens_from_block_table, _float_env, _int_env,
    _kv_cache_block_size, _kv_cache_gather_tokens, _kv_cache_layout,
    _kv_cache_scatter_tokens, _merge_pending_positions,
    _positions_are_full_prefill, _precision_summary_from_counts,
    _sequence_key_from_block_table, _sequence_key_from_slots,
    _serving_target_bits, _slot_in_cache, _slot_positions_from_slot_mapping,
    _slots_from_block_table_positions, _valid_slot_ids,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.kvtuner import (
    _kvtuner_config_path, _kvtuner_quant_mode, _load_kvtuner_config,
)
from vllm.attention.ops.atc_kv_fake_quant.methods.mixkvq import (
    _mixkvq_assign_bits, _mixkvq_salience, _quantize_key_by_channel_bits,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    quantize_last_dim_groups, quantize_token_groups_per_channel,
)

_RESIDUAL_CACHEWIDE_LEDGER: dict[str, dict[int, str]] = {}
_RESIDUAL_CACHEWIDE_SEQ_STATE: dict[str, dict[str, int]] = {}
_RESIDUAL_CACHEWIDE_PENDING_POSITIONS: dict[str, dict[str, list[int]]] = {}

def _residual_cachewide_after_write(
    method: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    block_size: int | None,
    attn,
    serving,
) -> dict[str, object]:
    """Rewrite old cache slots so residual/sink policy is global per request."""
    block = _kv_cache_block_size(key_cache, value_cache, block_size)
    layout = _kv_cache_layout(key_cache, value_cache, block)
    if layout == "unsupported":
        raise RuntimeError(
            f"unsupported KV cache layout key={tuple(key_cache.shape)} "
            f"value={tuple(value_cache.shape)} block={block}")
    positions = attn.cache_segment_positions
    if not positions:
        raise RuntimeError("cache-wide residual rewrite needs sequence positions")

    current_slot_ids = _valid_slot_ids(slot_mapping)
    current_slot_set = set(current_slot_ids)
    block_tables = attn.cache_block_tables
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
        raise RuntimeError("cache-wide residual rewrite needs block_tables")

    policy = _residual_cachewide_policy(method, attn, serving, key_cache)
    residual = int(policy["residual_tokens"])
    ledger_key = _residual_cachewide_ledger_key(
        method, key_cache, value_cache, attn.layer_idx)
    ledger = _RESIDUAL_CACHEWIDE_LEDGER.setdefault(ledger_key, {})
    seq_state = _RESIDUAL_CACHEWIDE_SEQ_STATE.setdefault(ledger_key, {})
    pending_positions = _RESIDUAL_CACHEWIDE_PENDING_POSITIONS.setdefault(
        ledger_key, {})
    for slot in current_slot_ids:
        ledger.pop(slot, None)

    expected_slots = 0
    covered_slots = 0
    rewritten_slots = 0
    rewritten_historical_slots = 0
    rewritten_current_slots = 0
    deferred_current_slots = 0
    current_slot_candidates = 0
    skipped_slots = 0
    invalid_slots = 0
    target_k_counts: dict[int, int] = {}
    target_v_counts: dict[int, int] = {}

    for row, (start, end, _context_len, seq_len) in enumerate(positions):
        seq_len = max(0, int(seq_len))
        if seq_len <= 0:
            continue
        expected_slots += seq_len
        _residual_accumulate_precision_counts(
            target_k_counts, target_v_counts, method, seq_len, residual,
            policy, key_cache.device)
        if use_slot_mapping_prefill:
            local_slots = current_slot_ids[int(start):int(end)]
            seq_key = _sequence_key_from_slots(local_slots, block)
            previous_seq_len = seq_state.get(seq_key)
            candidates = _residual_candidate_positions(
                previous_seq_len, seq_len, residual)
            candidates = _merge_pending_positions(
                candidates, pending_positions.get(seq_key), seq_len)
            slot_positions = _slot_positions_from_slot_mapping(
                local_slots, candidates, block, key_cache.shape[0])
            covered_for_seq = len([
                slot for slot in local_slots
                if _slot_in_cache(slot, block, key_cache.shape[0])
            ])
        else:
            assert block_tables_cpu is not None
            seq_key = _sequence_key_from_block_table(block_tables_cpu[row])
            previous_seq_len = seq_state.get(seq_key)
            candidates = _residual_candidate_positions(
                previous_seq_len, seq_len, residual)
            candidates = _merge_pending_positions(
                candidates, pending_positions.get(seq_key), seq_len)
            slot_positions = _slots_from_block_table_positions(
                block_tables_cpu[row], candidates, block, key_cache.shape[0])
            covered_for_seq = _covered_tokens_from_block_table(
                block_tables_cpu[row], seq_len, block, key_cache.shape[0])
        covered_slots += covered_for_seq
        invalid_slots += max(0, seq_len - covered_for_seq)

        rewrite_slots: list[int] = []
        rewrite_positions: list[int] = []
        current_deferred_positions: list[int] = []
        for pos, slot in slot_positions:
            marker = _residual_cachewide_marker(method, pos, seq_len,
                                                residual, policy)
            if slot in current_slot_set:
                current_slot_candidates += 1
                deferred_current_slots += 1
                current_deferred_positions.append(int(pos))
                ledger.pop(slot, None)
                continue
            if ledger.get(slot) == marker:
                skipped_slots += 1
                continue
            if marker == "16":
                ledger[slot] = marker
                skipped_slots += 1
                continue
            rewrite_slots.append(slot)
            rewrite_positions.append(int(pos))

        if rewrite_slots:
            k_tokens, v_tokens = _kv_cache_gather_tokens(
                key_cache, value_cache, rewrite_slots, block, layout)
            qk, qv = _residual_cachewide_quantize_tokens(
                method, k_tokens, v_tokens, rewrite_positions, seq_len,
                residual, policy, attn)
            _kv_cache_scatter_tokens(key_cache, value_cache, rewrite_slots,
                                     qk, qv, block, layout)
            for slot, pos in zip(rewrite_slots, rewrite_positions):
                ledger[slot] = _residual_cachewide_marker(
                    method, pos, seq_len, residual, policy)
            rewritten_slots += len(rewrite_slots)
            rewritten_current = sum(1 for slot in rewrite_slots
                                    if slot in current_slot_set)
            rewritten_current_slots += rewritten_current
            rewritten_historical_slots += len(rewrite_slots) - rewritten_current
        if seq_key:
            seq_state[seq_key] = seq_len
            pending_positions[seq_key] = current_deferred_positions

    selected = policy.get("selected_bit_width")
    summary = _precision_summary_from_counts(target_k_counts, target_v_counts,
                                             selected)
    coverage = covered_slots / max(1, expected_slots)
    summary.update({
        "cache_wide_source": ("post_write_vllm_cache_rewrite_slot_mapping_prefill"
                              if use_slot_mapping_prefill else
                              "post_write_vllm_cache_rewrite_block_tables"),
        "cache_wide_layout": layout,
        "cache_wide_coverage": coverage,
        "cachewide_residual_coverage": coverage,
        "cache_wide_expected_slots": expected_slots,
        "cache_wide_covered_slots": covered_slots,
        "rewritten_slots": rewritten_slots,
        "rewritten_historical_slots": rewritten_historical_slots,
        "rewritten_current_slots": rewritten_current_slots,
        "deferred_current_slots": deferred_current_slots,
        "current_slot_candidates": current_slot_candidates,
        "current_slot_skip_rate": (
            deferred_current_slots / max(1, current_slot_candidates)
            if current_slot_candidates else 0.0),
        "skipped_slots": skipped_slots,
        "invalid_slots": invalid_slots,
        "slot_bit_distribution": summary.get("precision_distribution", {}),
        "slot_target_bit_distribution": summary.get("precision_distribution", {}),
        "global_position_source": ("slot_mapping_full_prefill"
                                   if use_slot_mapping_prefill else
                                   "block_tables_seq_lens"),
        "residual_tokens": residual,
        "cache_wide_timing": "defer_current",
        "attention_safe_defer": True,
        **policy.get("trace", {}),
    })
    return summary

def _residual_cachewide_policy(method: str, attn, serving,
                               key_cache: torch.Tensor) -> dict[str, object]:
    if method == "kivi":
        k_bits = _int_env("ATC_KIVI_K_BITS", 2)
        v_bits = _int_env("ATC_KIVI_V_BITS", 2)
        return {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "group_size": _int_env("ATC_KIVI_GROUP_SIZE", 32),
            "residual_tokens": _int_env("ATC_KIVI_RESIDUAL_TOKENS", 128),
            "selected_bit_width": f"K{k_bits}V{v_bits}",
            "trace": {
                "nominal_k_bits": k_bits,
                "nominal_v_bits": v_bits,
                "nominal_kv_bits": (float(k_bits) + float(v_bits)) / 2.0,
                "effective_precision": "cachewide_residual_aware",
                **reference_source("kivi"),
            },
        }
    if method == "kvtuner":
        cfg = _load_kvtuner_config()
        layer = attn.layer_idx if attn.layer_idx >= 0 else 0
        pair = cfg.get(layer, cfg.get(str(layer),
                                      {"nbits_key": 4, "nbits_value": 4}))
        k_bits = int(pair.get("nbits_key", 4))
        v_bits = int(pair.get("nbits_value", 4))
        quant_mode = _kvtuner_quant_mode()
        residual_default = 32 if quant_mode == "kivi" else 0
        group_default = 32 if quant_mode == "kivi" else -1
        return {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "group_size": _int_env("ATC_KVTUNER_GROUP_SIZE", group_default),
            "residual_tokens": _int_env("ATC_KVTUNER_RESIDUAL_TOKENS",
                                        residual_default),
            "quant_mode": quant_mode,
            "selected_bit_width": f"K{k_bits}V{v_bits}",
            "trace": {
                "k_bits": k_bits,
                "v_bits": v_bits,
                "nominal_k_bits": k_bits,
                "nominal_v_bits": v_bits,
                "nominal_kv_bits": (float(k_bits) + float(v_bits)) / 2.0,
                "effective_precision": "cachewide_residual_aware",
                "kvtuner_config_path": str(_kvtuner_config_path()),
                "kvtuner_quant_mode": quant_mode,
                **reference_source("kvtuner"),
            },
        }
    mix_mode = "serving" if method == "mixkvq_serving" else os.environ.get(
        "ATC_MIXKVQ_MODE", "paper").lower()
    target_bits = _float_env("ATC_MIXKVQ_TARGET_BITS", 2.7)
    if mix_mode == "serving":
        target_bits = _serving_target_bits(target_bits, serving, "ATC_MIXKVQ")
    heads = int(key_cache.shape[-2]) if key_cache.ndim == 4 else int(
        key_cache.shape[1])
    head_size = int(key_cache.shape[-1]) if key_cache.ndim == 4 else int(
        key_cache.shape[2] * key_cache.shape[-1])
    channel_bits = _mixkvq_assign_bits(
        torch.ones((heads, head_size), device=key_cache.device),
        target_bits)
    return {
        "group_size": _int_env("ATC_MIXKVQ_GROUP_SIZE", 32),
        "residual_tokens": _int_env("ATC_MIXKVQ_RESIDUAL_TOKENS", 128),
        "sink_tokens": _int_env("ATC_MIXKVQ_SINK_TOKENS", 32),
        "target_bits": target_bits,
        "mode": mix_mode,
        "channel_bits": channel_bits,
        "selected_bit_width": "C2.7",
        "trace": {
            "target_bits": target_bits,
            "mode": mix_mode,
            "sink_tokens": _int_env("ATC_MIXKVQ_SINK_TOKENS", 32),
            "effective_precision": "cachewide_residual_aware",
            "residual_policy": "mixkvq_lazy_update_buffer",
            "policy": f"mixkvq_{mix_mode}_query_salience_budget_fake_quant",
        },
    }

def _residual_candidate_positions(previous_seq_len: int | None,
                                  seq_len: int,
                                  residual: int) -> list[int]:
    seq_len = max(0, int(seq_len))
    if seq_len <= 0:
        return []
    residual = max(0, int(residual))
    if previous_seq_len is None or previous_seq_len <= 0 or previous_seq_len > seq_len:
        return list(range(seq_len))
    start = max(0, int(previous_seq_len) - residual)
    return list(range(start, seq_len))

def _residual_cachewide_marker(method: str, pos: int, seq_len: int,
                               residual: int,
                               policy: dict[str, object]) -> str:
    k_bit, v_bit = _residual_cachewide_target_bits(method, pos, seq_len,
                                                   residual, policy)
    if k_bit == 16 and v_bit == 16:
        return "16"
    if method in {"mixkvq", "mixkvq_serving"}:
        return "mixkvq_body"
    return f"K{int(k_bit)}V{int(v_bit)}"

def _residual_cachewide_target_bits(method: str, pos: int, seq_len: int,
                                    residual: int,
                                    policy: dict[str, object]) -> tuple[int, int]:
    pos = int(pos)
    seq_len = max(0, int(seq_len))
    residual = max(0, int(residual))
    if method == "kivi" or method == "kvtuner":
        k_bits = int(policy["k_bits"])
        v_bits = int(policy["v_bits"])
        quant_mode = str(policy.get("quant_mode", "kivi")).lower()
        if method == "kivi" or quant_mode == "kivi":
            k_residual = (seq_len % residual) if residual > 0 else 0
            v_residual = min(seq_len, residual) if residual > 0 else 0
            k_start = seq_len - k_residual if k_residual > 0 else seq_len
            v_start = seq_len - v_residual if v_residual > 0 else seq_len
            return (16 if pos >= k_start else k_bits,
                    16 if pos >= v_start else v_bits)
        shared_residual = min(seq_len, residual) if residual > 0 else 0
        shared_start = seq_len - shared_residual if shared_residual > 0 else seq_len
        bit_k = 16 if pos >= shared_start else k_bits
        bit_v = 16 if pos >= shared_start else v_bits
        return bit_k, bit_v
    if method in {"mixkvq", "mixkvq_serving"}:
        sink, live = _mixkvq_protected_token_counts(seq_len, residual, policy)
        in_sink = pos < sink
        in_live_buffer = live > 0 and pos >= seq_len - live
        if in_sink or in_live_buffer:
            return 16, 16
        return 2, 2
    return 16, 16

def _residual_accumulate_precision_counts(
    k_counts: dict[int, int],
    v_counts: dict[int, int],
    method: str,
    seq_len: int,
    residual: int,
    policy: dict[str, object],
    device: torch.device,
) -> None:
    seq_len = max(0, int(seq_len))
    if method in {"kivi", "kvtuner"}:
        k_residual, v_residual = _kivi_style_residual_counts(
            method, seq_len, residual, policy)
        _add_count(k_counts, int(policy["k_bits"]), seq_len - k_residual)
        _add_count(v_counts, int(policy["v_bits"]), seq_len - v_residual)
        _add_count(k_counts, 16, k_residual)
        _add_count(v_counts, 16, v_residual)
        return
    protected_tokens = 0
    body_tokens = seq_len
    if method in {"mixkvq", "mixkvq_serving"}:
        sink, live = _mixkvq_protected_token_counts(seq_len, residual, policy)
        protected_tokens = min(seq_len, sink + live)
        body_tokens = max(0, seq_len - protected_tokens)
    channel_bits = policy.get("channel_bits")
    if not isinstance(channel_bits, torch.Tensor):
        channel_bits = torch.full((1,), 2, device=device, dtype=torch.int16)
    values = [int(v) for v in channel_bits.detach().flatten().cpu().tolist()]
    for bit in values:
        _add_count(k_counts, bit, body_tokens)
    elems = max(1, len(values))
    _add_count(k_counts, 16, protected_tokens * elems)
    _add_count(v_counts, 2, body_tokens * elems)
    _add_count(v_counts, 16, protected_tokens * elems)

def _kivi_style_residual_counts(method: str, seq_len: int, residual: int,
                                policy: dict[str, object]) -> tuple[int, int]:
    seq_len = max(0, int(seq_len))
    residual = max(0, int(residual))
    quant_mode = str(policy.get("quant_mode", "kivi")).lower()
    if method == "kivi" or quant_mode == "kivi":
        k_residual = (seq_len % residual) if residual > 0 else 0
        v_residual = min(seq_len, residual) if residual > 0 else 0
        return k_residual, v_residual
    shared_residual = min(seq_len, residual) if residual > 0 else 0
    return shared_residual, shared_residual

def _mixkvq_protected_token_counts(seq_len: int, residual: int,
                                   policy: dict[str, object]) -> tuple[int, int]:
    seq_len = max(0, int(seq_len))
    residual = max(0, int(residual))
    sink = min(seq_len, max(0, int(policy.get("sink_tokens", 0))))
    body_after_sink = max(0, seq_len - sink)
    live = (body_after_sink % residual) if residual > 0 else 0
    return sink, live

def _residual_cachewide_quantize_tokens(
    method: str,
    key_tokens: torch.Tensor,
    value_tokens: torch.Tensor,
    positions: list[int],
    seq_len: int,
    residual: int,
    policy: dict[str, object],
    attn,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method in {"kivi", "kvtuner"}:
        group_size = int(policy["group_size"])
        if method == "kvtuner" and str(policy.get("quant_mode",
                                                  "kivi")).lower() == "pertoken":
            qk = quantize_last_dim_groups(key_tokens, int(policy["k_bits"]),
                                          group_size)
        else:
            qk = quantize_token_groups_per_channel(
                key_tokens, int(policy["k_bits"]), group_size)
        qv = quantize_last_dim_groups(value_tokens, int(policy["v_bits"]),
                                      group_size)
        k_restore: list[int] = []
        v_restore: list[int] = []
        for idx, pos in enumerate(positions):
            k_bit, v_bit = _residual_cachewide_target_bits(
                method, pos, seq_len, residual, policy)
            if k_bit == 16:
                k_restore.append(idx)
            if v_bit == 16:
                v_restore.append(idx)
        if k_restore:
            qk[k_restore] = key_tokens[k_restore]
        if v_restore:
            qv[v_restore] = value_tokens[v_restore]
        return qk, qv
    group_size = int(policy["group_size"])
    target_bits = float(policy["target_bits"])
    if key_tokens.numel() == 0:
        return key_tokens, value_tokens
    salience = _mixkvq_salience(key_tokens, attn.query)
    bits = _mixkvq_assign_bits(salience, target_bits)
    qk = _quantize_key_by_channel_bits(key_tokens, bits, group_size)
    qv = quantize_last_dim_groups(value_tokens, 2, group_size)
    return qk, qv

def _residual_cachewide_ledger_key(method: str,
                                   key_cache: torch.Tensor,
                                   value_cache: torch.Tensor,
                                   layer_idx: int) -> str:
    return (f"{method}:{key_cache.device}:{key_cache.data_ptr()}:"
            f"{value_cache.data_ptr()}:{int(layer_idx)}")
