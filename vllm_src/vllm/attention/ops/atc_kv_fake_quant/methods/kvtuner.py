"""KVTuner fake-quant implementation and preset loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import reference_source
from vllm.attention.ops.atc_kv_fake_quant.common import (
    _int_env, _iter_segments, _residual_precision_summary,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    quantize_last_dim_groups, quantize_token_groups_per_channel, split_recent,
)

def _kvtuner(key, value, attn, serving, segments):
    # KVTuner: offline searched layer-wise K/V precision pairs.
    cfg = _load_kvtuner_config()
    layer = attn.layer_idx if attn.layer_idx >= 0 else 0
    pair = cfg.get(layer, cfg.get(str(layer), {"nbits_key": 4, "nbits_value": 4}))
    k_bits = int(pair.get("nbits_key", 4))
    v_bits = int(pair.get("nbits_value", 4))
    quant_mode = _kvtuner_quant_mode()
    group_default = 32 if quant_mode == "kivi" else -1
    residual_default = 32 if quant_mode == "kivi" else 0
    group_size = _int_env("ATC_KVTUNER_GROUP_SIZE", group_default)
    residual = _int_env("ATC_KVTUNER_RESIDUAL_TOKENS", residual_default)
    spans = _iter_segments(segments, key.shape[0])
    out_k = key.clone()
    out_v = value.clone()
    for start, end in spans:
        k_body, k_tail = split_recent(key[start:end], residual)
        v_body, v_tail = split_recent(value[start:end], residual)
        if quant_mode == "pertoken":
            qk = quantize_last_dim_groups(k_body, k_bits, group_size)
        else:
            qk = quantize_token_groups_per_channel(k_body, k_bits, group_size)
        qv = quantize_last_dim_groups(v_body, v_bits, group_size)
        out_k[start:end] = torch.cat([qk, k_tail], dim=0) if k_tail.numel() else qk
        out_v[start:end] = torch.cat([qv, v_tail], dim=0) if v_tail.numel() else qv
    summary = _residual_precision_summary(
        key.shape[0], spans, k_bits, v_bits, residual, key.device,
        f"K{k_bits}V{v_bits}")
    summary.update({
        "k_bits": k_bits, "v_bits": v_bits,
        "residual_tokens": residual,
        "sequence_segments": len(spans),
        "kvtuner_config_path": str(_kvtuner_config_path()),
        "kvtuner_variant": os.environ.get("ATC_KVTUNER_VARIANT_LABEL", ""),
        "kvtuner_quant_mode": quant_mode,
        "kvtuner_group_size": group_size,
        **_kvtuner_config_summary(cfg),
        **reference_source("kvtuner"),
    })
    return out_k, out_v, summary

def _kvtuner_config_path() -> Path:
    default = ("/root/atc_vllm_sched/references/kv_methods/KVTuner/"
               "calibration_presets/"
               "Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml")
    return Path(os.environ.get("ATC_KVTUNER_CONFIG_PATH")
                or os.environ.get("ATC_KVTUNER_PRESET_PATH")
                or default)

def _kvtuner_quant_mode() -> str:
    configured = os.environ.get("ATC_KVTUNER_QUANT_MODE", "").strip().lower()
    if configured in {"kivi", "pertoken", "per-token", "per_token"}:
        return "pertoken" if configured in {"pertoken", "per-token",
                                            "per_token"} else "kivi"
    path = str(_kvtuner_config_path()).lower()
    return "pertoken" if "pertoken" in path or "per-token" in path else "kivi"

@lru_cache(maxsize=4)
def _load_kvtuner_config_from_path(path_str: str) -> dict:
    path = Path(path_str)
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}
    except Exception:
        return {}

def _load_kvtuner_config() -> dict:
    return _load_kvtuner_config_from_path(str(_kvtuner_config_path()))

def _kvtuner_config_summary(cfg: dict) -> dict:
    if not cfg:
        return {}
    k_bits: list[int] = []
    v_bits: list[int] = []
    for item in cfg.values():
        if not isinstance(item, dict):
            continue
        try:
            k_bits.append(int(item.get("nbits_key", 4)))
            v_bits.append(int(item.get("nbits_value", 4)))
        except Exception:
            continue
    if not k_bits or not v_bits:
        return {}
    counts: dict[str, int] = {}
    for bit in k_bits + v_bits:
        key = str(int(bit))
        counts[key] = counts.get(key, 0) + 1
    return {
        "kvtuner_num_layers": len(k_bits),
        "kvtuner_nominal_avg_k_bits": sum(k_bits) / len(k_bits),
        "kvtuner_nominal_avg_v_bits": sum(v_bits) / len(v_bits),
        "kvtuner_nominal_avg_kv_bits": (
            sum(k_bits) + sum(v_bits)) / (len(k_bits) + len(v_bits)),
        "kvtuner_nominal_precision_counts": counts,
    }
