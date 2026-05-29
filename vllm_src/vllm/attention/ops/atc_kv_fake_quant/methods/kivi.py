"""KIVI fake-quant implementation."""

from __future__ import annotations

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import reference_source
from vllm.attention.ops.atc_kv_fake_quant.common import (
    _int_env, _iter_segments, _residual_precision_summary,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    quantize_last_dim_groups, quantize_token_groups_per_channel, split_recent,
)

def _kivi(key, value, attn, serving, segments):
    # KIVI: K per-channel, V per-token, asymmetric 2/4-bit, residual full precision.
    k_bits = _int_env("ATC_KIVI_K_BITS", 2)
    v_bits = _int_env("ATC_KIVI_V_BITS", 2)
    group_size = _int_env("ATC_KIVI_GROUP_SIZE", 32)
    residual = _int_env("ATC_KIVI_RESIDUAL_TOKENS", 128)
    spans = _iter_segments(segments, key.shape[0])
    out_k = key.clone()
    out_v = value.clone()
    for start, end in spans:
        k_body, k_tail = split_recent(key[start:end], residual)
        v_body, v_tail = split_recent(value[start:end], residual)
        qk = quantize_token_groups_per_channel(k_body, k_bits, group_size)
        qv = quantize_last_dim_groups(v_body, v_bits, group_size)
        out_k[start:end] = torch.cat([qk, k_tail], dim=0) if k_tail.numel() else qk
        out_v[start:end] = torch.cat([qv, v_tail], dim=0) if v_tail.numel() else qv
    summary = _residual_precision_summary(
        key.shape[0], spans, k_bits, v_bits, residual, key.device,
        f"K{k_bits}V{v_bits}")
    summary.update({
        "residual_tokens": residual,
        "sequence_segments": len(spans),
        **reference_source("kivi"),
    })
    return out_k, out_v, summary
