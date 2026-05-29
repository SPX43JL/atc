"""Paper-aligned Python-level KV cache fake quantization.

The hook perturbs key/value tensors with quantize-dequantize before the normal
vLLM cache writer runs. It intentionally leaves cache allocation and all
CUDA/Triton/PagedAttention kernels untouched.
"""

from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import (
    kvquant_official_nuq, kvquant_official_zp, pmkvq_official_fake_quant,
    reference_source)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    dense_sparse_normal_float_quant, dense_sparse_quant, keep_recent,
    normal_float_quant, quantize_last_dim_groups,
    quantize_token_groups_per_channel, split_recent)
from vllm.attention.ops.atc_kv_fake_quant.runtime import (
    attention_context, current_attention_context, load_serving_state)
from vllm.attention.ops.atc_kv_fake_quant.trace import bit_summary, emit_trace

SUPPORTED_METHODS = {
    "none", "kivi", "kvtuner", "kvquant", "pmkvq", "mixkvq",
    "pmkvq_serving", "pmkvq_cachewide", "mixkvq_serving"
}

_CACHEWIDE_RESIDUAL_METHODS = {
    "kivi", "kvtuner", "mixkvq", "mixkvq_serving"
}

_PMKVQ_CACHEWIDE_LEDGER: dict[str, dict[int, int]] = {}
_PMKVQ_CACHEWIDE_SEQ_STATE: dict[str, dict[str, int]] = {}
_PMKVQ_CACHEWIDE_PENDING_POSITIONS: dict[str, dict[str, list[int]]] = {}
_RESIDUAL_CACHEWIDE_LEDGER: dict[str, dict[int, str]] = {}
_RESIDUAL_CACHEWIDE_SEQ_STATE: dict[str, dict[str, int]] = {}
_RESIDUAL_CACHEWIDE_PENDING_POSITIONS: dict[str, dict[str, list[int]]] = {}
_KVQUANT_PREROPE_TLS = threading.local()


def maybe_fake_quant_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    slot_mapping: torch.Tensor | None = None,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    method = os.environ.get("ATC_KV_FAKE_QUANT_METHOD", "none").strip().lower()
    if method in {"", "off", "false", "0", "baseline"}:
        method = "none"
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown ATC_KV_FAKE_QUANT_METHOD={method!r}")
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
    method = os.environ.get("ATC_KV_FAKE_QUANT_METHOD", "none").strip().lower()
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


@lru_cache(maxsize=4)
def _load_kvquant_artifact_metadata() -> dict:
    path = os.environ.get("ATC_KVQUANT_ARTIFACT_PATH", "").strip()
    if not path:
        return {}
    base = Path(path)
    candidates = [base]
    if base.is_dir():
        candidates = [base / "metadata.json", base / "kvquant_metadata.json"]
    for candidate in candidates:
        try:
            if candidate.suffix == ".json":
                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"path": path, "metadata_load": "unavailable"}


@lru_cache(maxsize=4)
def _load_kvquant_nuq_artifact_from_path(path_str: str) -> dict:
    if not path_str:
        return {}
    base = Path(path_str)
    candidates = [base]
    if base.is_dir():
        candidates = [base / "nuq_artifact.pt", base / "kvquant_nuq.pt"]
    for candidate in candidates:
        if not candidate.exists() or candidate.suffix not in {".pt", ".pth"}:
            continue
        try:
            obj = torch.load(candidate, map_location="cpu")
        except Exception as exc:
            return {
                "artifact_complete": False,
                "artifact_load_error": f"{type(exc).__name__}: {exc}",
                "artifact_path": str(candidate),
            }
        if isinstance(obj, dict):
            obj.setdefault("artifact_path", str(candidate))
            return obj
    return {
        "artifact_complete": False,
        "artifact_load_error": "nuq_artifact.pt not found",
        "artifact_path": str(base),
    }


def _kvquant_nuq_artifact_path() -> str:
    return (os.environ.get("ATC_KVQUANT_NUQ_ARTIFACT_PATH", "").strip()
            or os.environ.get("ATC_KVQUANT_ARTIFACT_PATH", "").strip())


def _load_kvquant_nuq_artifact() -> dict:
    return _load_kvquant_nuq_artifact_from_path(_kvquant_nuq_artifact_path())


def _kvquant_artifact_complete(artifact: dict) -> bool:
    if not artifact:
        return False
    if bool(artifact.get("nuq_artifact_complete")):
        return True
    if bool(artifact.get("artifact_complete")) and artifact.get("layers"):
        return True
    return artifact.get("format") == "atc_kvquant_nuq_v1" and bool(
        artifact.get("layers"))


def _kvquant_layer_entry(artifact: dict, layer_idx: int,
                         kind: str) -> dict | None:
    layers = artifact.get("layers") if isinstance(artifact, dict) else None
    if not isinstance(layers, dict):
        return None
    layer = layers.get(str(layer_idx))
    if layer is None:
        layer = layers.get(layer_idx)
    if not isinstance(layer, dict):
        return None
    entry = layer.get(kind)
    return entry if isinstance(entry, dict) else None


def _kvquant_entry_lut(entry: dict | None) -> object | None:
    if not entry:
        return None
    for key in ("lut", "signposts", "centroids", "codebook"):
        if key in entry:
            return entry[key]
    return None


def _kvquant_strict_artifact() -> bool:
    return os.environ.get("ATC_KVQUANT_STRICT_ARTIFACT", "0") != "0"


def _kvquant_prerope_enabled() -> bool:
    return os.environ.get("ATC_KVQUANT_PREROPE", "0").lower() in {
        "1", "true", "yes", "on"
    }


def _kvquant_layer_idx(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)", layer_name or "")
    return int(match.group(1)) if match else -1


def _kvquant_prerope_mark(layer_idx: int, info: dict[str, object]) -> None:
    markers = getattr(_KVQUANT_PREROPE_TLS, "markers", None)
    if not isinstance(markers, dict):
        markers = {}
        _KVQUANT_PREROPE_TLS.markers = markers
    markers[int(layer_idx)] = info


def _kvquant_prerope_pop(layer_idx: int) -> dict[str, object] | None:
    markers = getattr(_KVQUANT_PREROPE_TLS, "markers", None)
    if not isinstance(markers, dict):
        return None
    marker = markers.pop(int(layer_idx), None)
    return marker if isinstance(marker, dict) else None


def _kvquant_nuq_quantize(
    flat: torch.Tensor,
    bits: int,
    qchannel: int,
    artifact_entry: dict | None,
    kind: str,
) -> tuple[torch.Tensor, str]:
    artifact = _load_kvquant_nuq_artifact()
    lut = _kvquant_entry_lut(artifact_entry)
    if lut is None or not _kvquant_artifact_complete(artifact):
        if _kvquant_strict_artifact():
            raise RuntimeError(
                f"KVQuant {kind} requires complete NUQ artifact at "
                f"{_kvquant_nuq_artifact_path()}")
        return kvquant_official_zp(
            flat,
            bits,
            qchannel,
            _float_env("ATC_KVQUANT_OUTLIER_RATIO",
                       _float_env("ATC_KVQUANT_OUTLIER_RATE", 0.01)),
            first_tokens_fp16=-1,
        )
    sparsity_threshold = float(
        artifact_entry.get(
            "sparsity_threshold",
            artifact.get("sparsity_threshold",
                         _float_env("ATC_KVQUANT_SPARSITY_THRESHOLD", 0.99))))
    return kvquant_official_nuq(
        flat,
        bits,
        int(artifact_entry.get("qchannel", qchannel)),
        sparsity_threshold,
        lut,
        first_tokens_fp16=-1,
    )


def maybe_kvquant_prerope_key(
    key: torch.Tensor,
    positions: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    method = os.environ.get("ATC_KV_FAKE_QUANT_METHOD", "none").strip().lower()
    if method != "kvquant" or not _kvquant_prerope_enabled():
        return key
    if key.numel() == 0:
        return key
    bits = _int_env("ATC_KVQUANT_BITS", 3)
    first_fp16 = _int_env("ATC_KVQUANT_FIRST_TOKENS_FP16", 1)
    layer_idx = _kvquant_layer_idx(layer_name)
    artifact = _load_kvquant_nuq_artifact()
    entry = _kvquant_layer_entry(artifact, layer_idx, "key")
    flat = key.reshape(key.shape[0], -1)
    qflat, source = _kvquant_nuq_quantize(flat, bits, 0, entry, "key")
    out = qflat.reshape_as(key)
    protected = 0
    if isinstance(positions, torch.Tensor) and first_fp16 > 0:
        try:
            first_mask = positions.reshape(-1).to(device=key.device) < first_fp16
            if first_mask.numel() == out.shape[0] and first_mask.any():
                out = out.clone()
                out[first_mask] = key[first_mask]
                protected = int(first_mask.sum().item())
        except Exception:
            protected = 0
    _kvquant_prerope_mark(layer_idx, {
        "pre_rope_applied": True,
        "pre_rope_layer_name": layer_name,
        "pre_rope_source": source,
        "pre_rope_tokens": int(key.shape[0]),
        "pre_rope_first_tokens_fp16_protected": protected,
        "nuq_artifact_complete": _kvquant_artifact_complete(artifact),
        "nuq_artifact_path": _kvquant_nuq_artifact_path(),
    })
    return out


def _kvquant(key, value, attn, serving, segments):
    # KVQuant: first-token sink preservation, dense-sparse outliers, optional
    # official NUQ artifact, and optional Qwen2 pre-RoPE K fake quant.
    bits = _int_env("ATC_KVQUANT_BITS", 3)
    group_size = _int_env("ATC_KVQUANT_GROUP_SIZE", 32)
    outlier_ratio = _float_env(
        "ATC_KVQUANT_OUTLIER_RATIO",
        _float_env("ATC_KVQUANT_OUTLIER_RATE", 0.01))
    first_fp16 = _int_env("ATC_KVQUANT_FIRST_TOKENS_FP16", 1)
    use_nuq = (
        os.environ.get("ATC_KVQUANT_NUQ")
        or os.environ.get("ATC_KVQUANT_USE_NUQ")
        or "1") != "0"
    artifact = _load_kvquant_nuq_artifact()
    artifact_complete = _kvquant_artifact_complete(artifact)
    pre_rope_requested = _kvquant_prerope_enabled()
    pre_rope_marker = (_kvquant_prerope_pop(attn.layer_idx)
                       if pre_rope_requested else None)
    pre_rope_applied = bool(pre_rope_marker)
    if pre_rope_requested and not pre_rope_applied and _kvquant_strict_artifact():
        raise RuntimeError(
            "KVQuant pre-RoPE mode requested but Qwen2 pre-RoPE hook did not "
            f"mark layer {attn.layer_idx}")
    spans = _iter_segments(segments, key.shape[0])
    positioned = attn.cache_segment_positions
    if not positioned or len(positioned) != len(spans):
        positioned = [(start, end, 0, end - start) for start, end in spans]
    qk = key.clone()
    qv = value.clone()
    k_bit_map = torch.full((key.shape[0],), int(bits), device=key.device,
                           dtype=torch.int16)
    v_bit_map = torch.full_like(k_bit_map, int(bits))
    k_sources: set[str] = set()
    v_sources: set[str] = set()
    protected_tokens = 0
    for start, end, context_len, _seq_len in positioned:
        seg_key = key[start:end]
        seg_value = value[start:end]
        seg_len = int(end) - int(start)
        if seg_len <= 0:
            continue
        local_positions = torch.arange(seg_len, device=key.device) + int(
            context_len)
        first_mask = local_positions < max(0, int(first_fp16))
        if pre_rope_applied:
            seg_qk = seg_key
            k_source = str(pre_rope_marker.get("pre_rope_source", "prerope"))
            k_sources.add(f"prerope:{k_source}")
        else:
            flat_key = seg_key.reshape(seg_key.shape[0], -1)
            key_entry = _kvquant_layer_entry(artifact, attn.layer_idx, "key")
            if use_nuq and artifact_complete and key_entry is not None:
                qk_flat, k_source = _kvquant_nuq_quantize(
                    flat_key, bits, 0, key_entry, "key")
                seg_qk = qk_flat.reshape_as(seg_key)
            else:
                qk_flat, k_source = kvquant_official_zp(
                    flat_key, bits, 0, outlier_ratio, first_tokens_fp16=-1)
                if k_source.startswith("official"):
                    seg_qk = qk_flat.reshape_as(seg_key)
                else:
                    seg_qk = dense_sparse_quant(
                        seg_key, bits, 0, outlier_ratio,
                        first_tokens_fp16=-1)
            k_sources.add(k_source)
        if first_mask.any():
            seg_qk = seg_qk.clone()
            seg_qk[first_mask] = seg_key[first_mask]
        qk[start:end] = seg_qk
        value_entry = _kvquant_layer_entry(artifact, attn.layer_idx, "value")
        if use_nuq and artifact_complete and value_entry is not None:
            flat_value = seg_value.reshape(-1, seg_value.shape[-1])
            qv_flat, v_source = _kvquant_nuq_quantize(
                flat_value, bits, -1, value_entry, "value")
            seg_qv = qv_flat.reshape_as(seg_value)
            v_sources.add(v_source)
        elif use_nuq:
            seg_qv = dense_sparse_normal_float_quant(
                seg_value, bits, -1, outlier_ratio, first_tokens_fp16=-1)
            v_sources.add("local_nuq_dense_sparse")
        else:
            flat_value = seg_value.reshape(-1, seg_value.shape[-1])
            qv_flat, v_source = kvquant_official_zp(
                flat_value, bits, -1, outlier_ratio, first_tokens_fp16=-1)
            v_sources.add(v_source)
            if v_source.startswith("official"):
                seg_qv = qv_flat.reshape_as(seg_value)
            else:
                seg_qv = dense_sparse_quant(
                    seg_value, bits, -1, outlier_ratio,
                    first_tokens_fp16=-1)
        if first_mask.any():
            seg_qv = seg_qv.clone()
            seg_qv[first_mask] = seg_value[first_mask]
            seg_k_bits = k_bit_map[start:end].clone()
            seg_v_bits = v_bit_map[start:end].clone()
            seg_k_bits[first_mask] = 16
            seg_v_bits[first_mask] = 16
            k_bit_map[start:end] = seg_k_bits
            v_bit_map[start:end] = seg_v_bits
            protected_tokens += int(first_mask.sum().item())
        qv[start:end] = seg_qv
    summary = _precision_summary(k_bit_map, v_bit_map, bits)
    summary = _include_sparse_outlier_effective_bits(summary, outlier_ratio)
    summary.update({
        "outlier_ratio": outlier_ratio,
        "first_tokens_fp16": first_fp16,
        "first_tokens_fp16_scope": "global_sequence_positions",
        "first_tokens_fp16_protected_in_current_write": protected_tokens,
        "nuq": use_nuq,
        "kvquant_mode": (
            "prerope_nuq" if pre_rope_applied and artifact_complete else
            "prerope_approx" if pre_rope_applied else
            "postrope_nuq" if artifact_complete else "postrope_approx"),
        "pre_rope_requested": pre_rope_requested,
        "pre_rope_applied": pre_rope_applied,
        "pre_rope_metadata": pre_rope_marker or {},
        "nuq_artifact_complete": artifact_complete,
        "nuq_artifact_path": _kvquant_nuq_artifact_path(),
        "artifact_path": os.environ.get("ATC_KVQUANT_ARTIFACT_PATH", ""),
        "artifact_metadata": _load_kvquant_artifact_metadata(),
        "k_source": "+".join(sorted(k_sources)) or "unknown",
        "v_source": "+".join(sorted(v_sources)) or "unknown",
        "group_size": group_size,
        "sequence_segments": len(spans),
        **reference_source("kvquant"),
    })
    return qk, qv, summary


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


def _add_count(counts: dict[int, int], bit: int, count: int) -> None:
    if int(count) <= 0:
        return
    bit = int(bit)
    counts[bit] = counts.get(bit, 0) + int(count)


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


def _pmkvq_cachewide_ledger_key(key_cache: torch.Tensor,
                                value_cache: torch.Tensor,
                                layer_idx: int) -> str:
    return (f"{key_cache.device}:{key_cache.data_ptr()}:"
            f"{value_cache.data_ptr()}:{int(layer_idx)}")


def _residual_cachewide_ledger_key(method: str,
                                   key_cache: torch.Tensor,
                                   value_cache: torch.Tensor,
                                   layer_idx: int) -> str:
    return (f"{method}:{key_cache.device}:{key_cache.data_ptr()}:"
            f"{value_cache.data_ptr()}:{int(layer_idx)}")


def _clear_pmkvq_cachewide_state() -> None:
    _PMKVQ_CACHEWIDE_LEDGER.clear()
    _PMKVQ_CACHEWIDE_SEQ_STATE.clear()
    _PMKVQ_CACHEWIDE_PENDING_POSITIONS.clear()
    _RESIDUAL_CACHEWIDE_LEDGER.clear()
    _RESIDUAL_CACHEWIDE_SEQ_STATE.clear()
    _RESIDUAL_CACHEWIDE_PENDING_POSITIONS.clear()


_METHODS = {
    "kivi": _kivi,
    "kvtuner": _kvtuner,
    "kvquant": _kvquant,
    "pmkvq": _pmkvq,
    "pmkvq_serving": _pmkvq_serving,
    "mixkvq": _mixkvq,
    "mixkvq_serving": _mixkvq_serving,
}

__all__ = [
    "maybe_fake_quant_kv",
    "maybe_cachewide_fake_quant_kv",
    "maybe_kvquant_prerope_key",
    "attention_context",
]
