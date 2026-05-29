"""KVQuant fake-quant implementation and Qwen2 pre-RoPE hook."""

from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import torch

from vllm.attention.ops.atc_kv_fake_quant.adapters import (
    kvquant_official_nuq, kvquant_official_zp, reference_source,
)
from vllm.attention.ops.atc_kv_fake_quant.common import (
    _float_env, _include_sparse_outlier_effective_bits, _int_env,
    _iter_segments, _precision_summary,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import (
    dense_sparse_normal_float_quant, dense_sparse_quant,
)

_KVQUANT_PREROPE_TLS = threading.local()

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
