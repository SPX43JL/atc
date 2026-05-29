"""MixKVQ diagnostic fake-quant implementation."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from vllm.attention.ops.atc_kv_fake_quant.common import (
    _float_env, _int_env, _iter_segments, _precision_summary,
    _serving_target_bits,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    keep_recent, quantize_last_dim_groups, quantize_token_groups_per_channel,
    split_recent,
)
from vllm.attention.ops.atc_kv_fake_quant.trace import bit_summary

def _mixkvq(key, value, attn, serving, segments):
    # MixKVQ: query-aware key-channel salience A_d = I_d * S_d; BF16/INT4/INT2
    # mixed precision for keys, per-token value quantization.
    group_size = _int_env("ATC_MIXKVQ_GROUP_SIZE", 32)
    residual = _int_env("ATC_MIXKVQ_RESIDUAL_TOKENS", 128)
    mode = os.environ.get("ATC_MIXKVQ_MODE", "paper").lower()
    target_bits = _float_env("ATC_MIXKVQ_TARGET_BITS", 2.7)
    if mode == "serving":
        target_bits = _serving_target_bits(target_bits, serving, "ATC_MIXKVQ")
    spans = _iter_segments(segments, key.shape[0])
    qk = key.clone()
    qv = value.clone()
    summary_bits: list[torch.Tensor] = []
    for start, end in spans:
        k_body, k_tail = split_recent(key[start:end], residual)
        if k_body.numel() == 0:
            bits = torch.full((key.shape[1], key.shape[2]), 16,
                              device=key.device, dtype=torch.int16)
            seg_qk = key[start:end]
        else:
            salience = _mixkvq_salience(k_body, attn.query)
            bits = _mixkvq_assign_bits(salience, target_bits)
            qk_body = _quantize_key_by_channel_bits(k_body, bits, group_size)
            seg_qk = torch.cat([qk_body, k_tail], dim=0) if k_tail.numel() else qk_body
        qk[start:end] = seg_qk
        seg_qv = quantize_last_dim_groups(value[start:end], 2, group_size)
        qv[start:end] = keep_recent(value[start:end], seg_qv, residual)
        summary_bits.append(bits.reshape(-1))
    if summary_bits:
        bits = torch.cat(summary_bits)
    else:
        bits = torch.full((key.shape[1] * key.shape[2],), 16,
                          device=key.device, dtype=torch.int16)
    summary = bit_summary(bits, 2)
    summary.update(_precision_summary(bits, 2, summary.get("selected_bit_width")))
    thresholds = _load_mixkvq_thresholds()
    summary.update({
        "residual_tokens": residual,
        "target_bits": target_bits,
        "mode": mode,
        "thresholds_source": "artifact" if thresholds else "ratio_approx",
        "thresholds_path": os.environ.get("ATC_MIXKVQ_THRESHOLDS_PATH", ""),
        "thresholds_metadata": thresholds.get("metadata", {}) if thresholds else {},
        "sequence_segments": len(spans),
        "policy": f"mixkvq_{mode}_query_salience_budget_fake_quant",
    })
    return qk, qv, summary

def _mixkvq_serving(key, value, attn, serving, segments):
    previous = os.environ.get("ATC_MIXKVQ_MODE")
    os.environ["ATC_MIXKVQ_MODE"] = "serving"
    try:
        return _mixkvq(key, value, attn, serving, segments)
    finally:
        if previous is None:
            os.environ.pop("ATC_MIXKVQ_MODE", None)
        else:
            os.environ["ATC_MIXKVQ_MODE"] = previous

def _mixkvq_salience(key: torch.Tensor, query: torch.Tensor | None) -> torch.Tensor:
    scale = (key.float().amax(dim=0) - key.float().amin(dim=0)).clamp_min(1e-6) / 3.0
    if query is None or not isinstance(query, torch.Tensor):
        importance = torch.ones_like(scale)
        return importance * scale
    q = query.float()
    try:
        q = q.reshape(q.shape[0], -1, key.shape[-1])
        if q.shape[1] % key.shape[1] == 0:
            group = q.shape[1] // key.shape[1]
            q = q.reshape(q.shape[0], key.shape[1], group, key.shape[-1])
            importance = q.abs().mean(dim=(0, 2))
        else:
            importance = q.abs().mean(dim=0)[:key.shape[1]]
            importance = importance.reshape(key.shape[1], key.shape[-1])
    except Exception:
        importance = torch.ones_like(scale)
    return importance.to(device=key.device) * scale

def _mixkvq_assign_bits(salience: torch.Tensor,
                        target_bits: float) -> torch.Tensor:
    calibration = _load_mixkvq_thresholds()
    if calibration:
        selected = calibration.get("selected", calibration)
        tau_bf16 = float(selected.get("tau_bf16", float("inf")))
        tau_int4 = float(selected.get("tau_int4", float("inf")))
        norm = salience.float() / salience.float().mean().clamp_min(1e-6)
        bits = torch.full_like(norm, 2, dtype=torch.int16)
        bits[norm >= tau_int4] = 4
        bits[norm >= tau_bf16] = 16
        return bits
    if os.environ.get("ATC_MIXKVQ_STRICT_THRESHOLDS", "0") != "0":
        raise RuntimeError(
            "MixKVQ strict formal requires ATC_MIXKVQ_THRESHOLDS_PATH "
            "threshold-search artifact")

    # Paper-aligned approximation when no threshold-search artifact is
    # available: allocate precision by descending query salience under the
    # target average bit-width reported by the paper (for example C2.7).
    flat = salience.flatten()
    n = flat.numel()
    bits = torch.full((n,), 2, device=salience.device, dtype=torch.int16)
    extra_budget = max(0, int(round((target_bits - 2.0) * n)))
    if extra_budget <= 0 or n == 0:
        return bits.reshape_as(salience)
    order = torch.argsort(flat, descending=True)
    # Reserve expensive BF16 only for extreme salience outliers; spend the
    # remaining budget on UINT4. This follows MixKVQ's BF16/UINT4/UINT2
    # hierarchy without hard-coding a fixed channel ratio.
    norm = flat / flat.mean().clamp_min(1e-6)
    high_candidates = order[norm[order] >= _float_env("ATC_MIXKVQ_BF16_TAU", 2.0)]
    high_budget = min(len(high_candidates), extra_budget // 14)
    if high_budget > 0:
        bits[high_candidates[:high_budget]] = 16
        extra_budget -= 14 * high_budget
    if extra_budget > 0:
        remaining = order[bits[order] == 2]
        med = min(len(remaining), extra_budget // 2)
        if med > 0:
            bits[remaining[:med]] = 4
    return bits.reshape_as(salience)

@lru_cache(maxsize=1)
def _load_mixkvq_thresholds() -> dict:
    path = os.environ.get("ATC_MIXKVQ_THRESHOLDS_PATH", "").strip()
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("thresholds_path", path)
            return data
    except Exception as exc:
        if os.environ.get("ATC_MIXKVQ_STRICT_THRESHOLDS", "0") != "0":
            raise RuntimeError(
                f"failed to load MixKVQ thresholds artifact {path}: "
                f"{type(exc).__name__}: {exc}") from exc
    return {}

def _quantize_key_by_channel_bits(key: torch.Tensor, bits: torch.Tensor,
                                  group_size: int) -> torch.Tensor:
    low = quantize_token_groups_per_channel(key, 2, group_size)
    mid = quantize_token_groups_per_channel(key, 4, group_size)
    flat_out = low.reshape(low.shape[0], -1).clone()
    flat_mid = mid.reshape(mid.shape[0], -1)
    flat_key = key.reshape(key.shape[0], -1)
    flat_bits = bits.reshape(-1)
    mid_mask = flat_bits == 4
    high_mask = flat_bits >= 16
    if mid_mask.any():
        flat_out[:, mid_mask] = flat_mid[:, mid_mask]
    if high_mask.any():
        flat_out[:, high_mask] = flat_key[:, high_mask]
    return flat_out.reshape_as(key)

def _quantize_by_token_bits(x: torch.Tensor, bit_map: torch.Tensor,
                            group_size: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for bits in sorted(set(int(v) for v in bit_map.detach().cpu().tolist())):
        mask = bit_map == bits
        if bits >= 16:
            out[mask] = x[mask]
        else:
            out[mask] = quantize_last_dim_groups(x[mask], bits, group_size)
    return out
