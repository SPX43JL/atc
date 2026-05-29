"""Shared helpers for ATC KV fake-quant method implementations."""

from __future__ import annotations

import os

import torch

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default

def _actual_num_tokens(key: torch.Tensor,
                       slot_mapping: torch.Tensor | None) -> int:
    if key.ndim == 0:
        return 1
    tokens = int(key.shape[0])
    if isinstance(slot_mapping, torch.Tensor) and slot_mapping.numel() > 0:
        return min(tokens, int(slot_mapping.numel()))
    return tokens

def _sequence_segments(slot_mapping: torch.Tensor | None,
                       num_tokens: int,
                       block_size: int | None) -> list[tuple[int, int]]:
    """Infer per-request spans from vLLM cache slots.

    vLLM flattens a cache write across the running batch. Sink/recent-window
    quantization policies must be applied per sequence, not once to the whole
    flattened tensor.
    """
    num_tokens = max(0, int(num_tokens))
    if num_tokens <= 0:
        return []
    if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
        return [(0, num_tokens)]
    try:
        slots = slot_mapping.detach().flatten()[:num_tokens].to("cpu").tolist()
    except Exception:
        return [(0, num_tokens)]
    if len(slots) != num_tokens:
        return [(0, num_tokens)]
    block = int(block_size or 0)
    starts = [0]
    for i in range(1, num_tokens):
        prev = int(slots[i - 1])
        cur = int(slots[i])
        same = cur == prev + 1
        if not same and block > 1:
            same = (prev % block == block - 1 and cur % block == 0)
        if not same:
            starts.append(i)
    starts.append(num_tokens)
    spans = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)
             if starts[i] < starts[i + 1]]
    max_segment = _int_env("ATC_KV_FAKE_QUANT_MAX_SEGMENT_TOKENS", 7500)
    if max_segment <= 0:
        return spans
    capped: list[tuple[int, int]] = []
    for start, end in spans:
        while end - start > max_segment:
            capped.append((start, start + max_segment))
            start += max_segment
        capped.append((start, end))
    return capped

def _iter_segments(segments: list[tuple[int, int]],
                   tokens: int) -> list[tuple[int, int]]:
    if not segments:
        return [(0, int(tokens))]
    return [(max(0, int(s)), min(int(tokens), int(e))) for s, e in segments
            if max(0, int(s)) < min(int(tokens), int(e))]

def _counts_and_avg(bits: torch.Tensor | int | float) -> tuple[dict[str, int], float]:
    if isinstance(bits, (int, float)):
        return {str(int(bits)): 1}, float(bits)
    if bits.numel() == 0:
        return {}, 0.0
    vals = [int(v) for v in bits.detach().flatten().cpu().tolist()]
    counts: dict[str, int] = {}
    for val in vals:
        counts[str(val)] = counts.get(str(val), 0) + 1
    total = max(1, len(vals))
    avg = sum(int(k) * v for k, v in counts.items()) / total
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0]))), float(avg)

def _precision_summary(k_bits: torch.Tensor | int | float,
                       v_bits: torch.Tensor | int | float,
                       selected: object | None = None) -> dict[str, object]:
    k_counts, avg_k = _counts_and_avg(k_bits)
    v_counts, avg_v = _counts_and_avg(v_bits)
    total_k = sum(k_counts.values()) or 1
    total_v = sum(v_counts.values()) or 1
    kv_counts: dict[str, int] = {}
    for source in (k_counts, v_counts):
        for bit, count in source.items():
            kv_counts[bit] = kv_counts.get(bit, 0) + count
    total_kv = total_k + total_v
    avg_kv = (avg_k * total_k + avg_v * total_v) / max(1, total_kv)
    summary: dict[str, object] = {
        "k_bit_counts": k_counts,
        "v_bit_counts": v_counts,
        "avg_k_bits": avg_k,
        "avg_v_bits": avg_v,
        "avg_kv_bits": avg_kv,
        "precision_distribution": {
            bit: count / max(1, total_kv)
            for bit, count in sorted(kv_counts.items(), key=lambda kv: int(kv[0]))
        },
    }
    if selected is not None:
        summary["selected_bit_width"] = selected
    return summary

def _include_sparse_outlier_effective_bits(
        summary: dict[str, object],
        outlier_ratio: float,
        outlier_bits: int = 16) -> dict[str, object]:
    """Account for KVQuant sparse outliers in effective bit statistics.

    KVQuant dense NUQ body is nominally low-bit, but include_sparse=True
    preserves a small outlier fraction outside the dense codebook. The fake
    quant tensor perturbation is unchanged here; this only makes trace bits
    comparable with methods whose FP16 residual/sink regions are included in
    avg_k/v/kv_bits.
    """
    ratio = max(0.0, min(1.0, float(outlier_ratio or 0.0)))
    if ratio <= 0.0 or summary.get("kvquant_sparse_outlier_bits_included"):
        return summary

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
                out[str(outlier_bits)] = (
                    out.get(str(outlier_bits), 0.0) + frac * ratio)
        total = sum(out.values()) or 1.0
        return {
            bit: val / total
            for bit, val in sorted(out.items(), key=lambda kv: int(kv[0]))
        }

    summary["kvquant_nominal_avg_k_bits_without_sparse_outlier"] = (
        summary.get("avg_k_bits"))
    summary["kvquant_nominal_avg_v_bits_without_sparse_outlier"] = (
        summary.get("avg_v_bits"))
    summary["kvquant_nominal_avg_kv_bits_without_sparse_outlier"] = (
        summary.get("avg_kv_bits"))
    summary["kvquant_nominal_precision_distribution_without_sparse_outlier"] = (
        summary.get("precision_distribution"))

    avg_k = adjusted_avg(summary.get("k_bit_counts"),
                         summary.get("avg_k_bits"))
    avg_v = adjusted_avg(summary.get("v_bit_counts"),
                         summary.get("avg_v_bits"))
    if avg_k is not None:
        summary["avg_k_bits"] = avg_k
    if avg_v is not None:
        summary["avg_v_bits"] = avg_v
    if avg_k is not None and avg_v is not None:
        summary["avg_kv_bits"] = (avg_k + avg_v) / 2.0
    dist = adjusted_dist(summary.get("precision_distribution"))
    if dist is not None:
        summary["precision_distribution"] = dist

    summary["kvquant_sparse_outlier_bits_included"] = True
    summary["kvquant_sparse_outlier_ratio"] = ratio
    summary["kvquant_sparse_outlier_bits"] = outlier_bits
    summary["kvquant_effective_bit_note"] = (
        "avg bits include first-token FP16 plus KVQuant sparse outlier "
        "fraction as FP16-equivalent trace accounting")
    return summary

def _residual_precision_summary(tokens: int,
                                spans: list[tuple[int, int]],
                                k_bits: int,
                                v_bits: int,
                                residual: int,
                                device: torch.device,
                                selected: object) -> dict[str, object]:
    k_map = torch.full((max(0, int(tokens)),), int(k_bits),
                       device=device, dtype=torch.int16)
    v_map = torch.full_like(k_map, int(v_bits))
    for start, end in _iter_segments(spans, tokens):
        tail = min(max(0, int(residual)), max(0, int(end) - int(start)))
        if tail > 0:
            k_map[end - tail:end] = 16
            v_map[end - tail:end] = 16
    summary = _precision_summary(k_map, v_map, selected)
    summary.update({
        "nominal_k_bits": int(k_bits),
        "nominal_v_bits": int(v_bits),
        "nominal_kv_bits": (float(k_bits) + float(v_bits)) / 2.0,
        "effective_precision": "residual_aware",
    })
    return summary

def _serving_target_bits(base: float, serving, prefix: str) -> float:
    if serving.pressure == "high":
        return _float_env(f"{prefix}_SERVING_HIGH_BITS", base)
    if serving.pressure == "medium":
        return _float_env(f"{prefix}_SERVING_MEDIUM_BITS", max(base, 3.0))
    return _float_env(f"{prefix}_SERVING_LOW_BITS", max(base, 4.0))

def _add_count(counts: dict[int, int], bit: int, count: int) -> None:
    if int(count) <= 0:
        return
    bit = int(bit)
    counts[bit] = counts.get(bit, 0) + int(count)

def _bit_summary_from_counts(counts: dict[int, int],
                             default_bits: int) -> dict[str, object]:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return {"selected_bit_width": default_bits, "bit_ratio": {}}
    avg_bits = sum(int(k) * int(v) for k, v in counts.items()) / total
    sorted_counts = dict(sorted(
        ((str(int(k)), int(v)) for k, v in counts.items()),
        key=lambda kv: int(kv[0])))
    return {
        "selected_bit_width": int(round(avg_bits)),
        "avg_bits": float(avg_bits),
        "bit_counts": sorted_counts,
        "bit_ratio": {
            bit: count / total
            for bit, count in sorted_counts.items()
        },
    }

def _precision_summary_from_counts(k_counts_in: dict[int, int],
                                   v_counts_in: dict[int, int],
                                   selected: object | None = None
                                   ) -> dict[str, object]:
    k_counts = {str(int(k)): int(v) for k, v in k_counts_in.items()}
    v_counts = {str(int(k)): int(v) for k, v in v_counts_in.items()}
    total_k = sum(k_counts.values()) or 1
    total_v = sum(v_counts.values()) or 1
    avg_k = sum(int(k) * v for k, v in k_counts.items()) / total_k
    avg_v = sum(int(k) * v for k, v in v_counts.items()) / total_v
    kv_counts: dict[str, int] = {}
    for source in (k_counts, v_counts):
        for bit, count in source.items():
            kv_counts[bit] = kv_counts.get(bit, 0) + count
    total_kv = total_k + total_v
    summary: dict[str, object] = {
        "k_bit_counts": dict(sorted(k_counts.items(),
                                    key=lambda kv: int(kv[0]))),
        "v_bit_counts": dict(sorted(v_counts.items(),
                                    key=lambda kv: int(kv[0]))),
        "avg_k_bits": float(avg_k),
        "avg_v_bits": float(avg_v),
        "avg_kv_bits": float((avg_k * total_k + avg_v * total_v) /
                             max(1, total_kv)),
        "precision_distribution": {
            bit: count / max(1, total_kv)
            for bit, count in sorted(kv_counts.items(),
                                     key=lambda kv: int(kv[0]))
        },
    }
    if selected is not None:
        summary["selected_bit_width"] = selected
    return summary

def _kv_cache_block_size(key_cache: torch.Tensor, value_cache: torch.Tensor,
                         block_size: int | None) -> int:
    block = int(block_size or 0)
    if block > 0:
        return block
    if key_cache.ndim == 4:
        return int(key_cache.shape[1])
    if key_cache.ndim == 5:
        return int(key_cache.shape[-2])
    if value_cache.ndim == 4:
        return int(value_cache.shape[-1])
    return -1

def _kv_cache_layout(key_cache: torch.Tensor, value_cache: torch.Tensor,
                     block_size: int) -> str:
    if block_size <= 0:
        return "unsupported"
    if (key_cache.ndim == 4 and value_cache.ndim == 4
            and key_cache.shape[1] == block_size
            and value_cache.shape[1] == block_size):
        return "flash4"
    if (key_cache.ndim == 5 and value_cache.ndim == 4
            and key_cache.shape[-2] == block_size
            and value_cache.shape[-1] == block_size):
        return "paged5"
    return "unsupported"

def _kv_cache_elements_per_token(key_cache: torch.Tensor,
                                 value_cache: torch.Tensor,
                                 layout: str) -> int:
    if layout == "flash4":
        return (int(key_cache.shape[-2]) * int(key_cache.shape[-1]) +
                int(value_cache.shape[-2]) * int(value_cache.shape[-1]))
    if layout == "paged5":
        key_elems = (int(key_cache.shape[1]) * int(key_cache.shape[2]) *
                     int(key_cache.shape[-1]))
        value_elems = int(value_cache.shape[1]) * int(value_cache.shape[2])
        return key_elems + value_elems
    return 0

def _slots_from_block_table(block_table: torch.Tensor,
                            seq_len: int,
                            block_size: int,
                            max_blocks: int) -> list[tuple[int, int]]:
    slots: list[tuple[int, int]] = []
    if block_size <= 0 or seq_len <= 0:
        return slots
    values = [int(v) for v in block_table.flatten().tolist()]
    for pos in range(int(seq_len)):
        logical_block = pos // block_size
        if logical_block >= len(values):
            break
        physical_block = values[logical_block]
        if physical_block < 0 or physical_block >= max_blocks:
            continue
        slots.append((pos, physical_block * block_size + (pos % block_size)))
    return slots

def _slots_from_block_table_positions(block_table: torch.Tensor,
                                      positions: list[int],
                                      block_size: int,
                                      max_blocks: int) -> list[tuple[int, int]]:
    slots: list[tuple[int, int]] = []
    if block_size <= 0:
        return slots
    values = [int(v) for v in block_table.flatten().tolist()]
    for pos in sorted(set(max(0, int(p)) for p in positions)):
        logical_block = pos // block_size
        if logical_block >= len(values):
            continue
        physical_block = values[logical_block]
        if physical_block < 0 or physical_block >= max_blocks:
            continue
        slots.append((pos, physical_block * block_size + (pos % block_size)))
    return slots

def _slot_positions_from_slot_mapping(slots: list[int],
                                      positions: list[int],
                                      block_size: int,
                                      max_blocks: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for pos in sorted(set(max(0, int(p)) for p in positions)):
        if pos >= len(slots):
            continue
        slot = int(slots[pos])
        if _slot_in_cache(slot, block_size, max_blocks):
            out.append((pos, slot))
    return out

def _covered_tokens_from_block_table(block_table: torch.Tensor,
                                     seq_len: int,
                                     block_size: int,
                                     max_blocks: int) -> int:
    if seq_len <= 0 or block_size <= 0:
        return 0
    values = [int(v) for v in block_table.flatten().tolist()]
    covered = 0
    for logical_block in range((int(seq_len) + block_size - 1) // block_size):
        if logical_block >= len(values):
            break
        physical_block = values[logical_block]
        if physical_block < 0 or physical_block >= max_blocks:
            continue
        covered += min(block_size, int(seq_len) - logical_block * block_size)
    return covered

def _valid_slot_ids(slot_mapping: torch.Tensor | None) -> list[int]:
    if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
        return []
    try:
        values = slot_mapping.detach().flatten().to("cpu").tolist()
    except Exception:
        return []
    return [int(v) for v in values if int(v) >= 0]

def _positions_are_full_prefill(
    positions: list[tuple[int, int, int, int]] | None
) -> bool:
    if not positions:
        return False
    for start, end, context_len, seq_len in positions:
        if int(context_len) != 0:
            return False
        if int(seq_len) != int(end) - int(start):
            return False
    return True

def _slot_in_cache(slot: int, block_size: int, max_blocks: int) -> bool:
    if int(slot) < 0 or block_size <= 0:
        return False
    block, _offset = _slot_to_block_offset(slot, block_size)
    return 0 <= block < int(max_blocks)

def _sequence_key_from_slots(slots: list[int], block_size: int) -> str:
    if not slots or block_size <= 0:
        return ""
    return f"slot_first_block:{max(0, int(slots[0])) // int(block_size)}"

def _sequence_key_from_block_table(block_table: torch.Tensor) -> str:
    values = [int(v) for v in block_table.flatten().tolist() if int(v) >= 0]
    if not values:
        return ""
    return f"block_table_first:{values[0]}"

def _merge_pending_positions(positions: list[int],
                             pending: list[int] | None,
                             seq_len: int) -> list[int]:
    if not pending:
        return positions
    merged = set(int(p) for p in positions)
    merged.update(int(p) for p in pending if 0 <= int(p) < int(seq_len))
    return sorted(merged)

def _slot_to_block_offset(slot: int, block_size: int) -> tuple[int, int]:
    return int(slot) // int(block_size), int(slot) % int(block_size)

def _slot_block_offset_tensors(slots: list[int],
                               block_size: int,
                               device: torch.device) -> tuple[torch.Tensor,
                                                               torch.Tensor]:
    slot_tensor = torch.tensor(slots, device=device, dtype=torch.long)
    blocks = torch.div(slot_tensor, int(block_size), rounding_mode="floor")
    offsets = torch.remainder(slot_tensor, int(block_size))
    return blocks, offsets

def _kv_cache_gather_tokens(key_cache: torch.Tensor,
                            value_cache: torch.Tensor,
                            slots: list[int],
                            block_size: int,
                            layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    blocks, offsets = _slot_block_offset_tensors(
        slots, block_size, key_cache.device)
    if layout == "flash4":
        return (key_cache[blocks, offsets].contiguous(),
                value_cache[blocks, offsets].contiguous())
    if layout == "paged5":
        keys = key_cache[blocks, :, :, offsets, :].reshape(
            len(slots), key_cache.shape[1],
            key_cache.shape[2] * key_cache.shape[-1])
        values = value_cache[blocks, :, :, offsets]
        return keys.contiguous(), values.contiguous()
    raise RuntimeError(f"unsupported KV cache layout {layout}")

def _kv_cache_scatter_tokens(key_cache: torch.Tensor,
                             value_cache: torch.Tensor,
                             slots: list[int],
                             keys: torch.Tensor,
                             values: torch.Tensor,
                             block_size: int,
                             layout: str) -> None:
    blocks, offsets = _slot_block_offset_tensors(
        slots, block_size, key_cache.device)
    if layout == "flash4":
        key_cache[blocks, offsets] = keys
        value_cache[blocks, offsets] = values
        return
    if layout == "paged5":
        key_cache[blocks, :, :, offsets, :] = keys.reshape(
            len(slots), key_cache.shape[1], key_cache.shape[2],
            key_cache.shape[-1])
        value_cache[blocks, :, :, offsets] = values
        return
    raise RuntimeError(f"unsupported KV cache layout {layout}")
