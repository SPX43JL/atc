#!/usr/bin/env python3
"""Smoke-test all Python-only fake quant methods without starting vLLM."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from vllm.attention.ops.atc_kv_fake_quant import (
    attention_context,
    maybe_cachewide_fake_quant_kv,
    maybe_fake_quant_kv,
)
from vllm.attention.ops.atc_kv_fake_quant.core import (
    _PMKVQ_CACHEWIDE_LEDGER,
    _PMKVQ_CACHEWIDE_PENDING_POSITIONS,
    _RESIDUAL_CACHEWIDE_LEDGER,
    _RESIDUAL_CACHEWIDE_PENDING_POSITIONS,
    _clear_pmkvq_cachewide_state,
    _kv_cache_gather_tokens,
    _kv_cache_scatter_tokens,
    _load_kvtuner_config,
    _load_kvtuner_config_from_path,
    _kvtuner_quant_mode,
    _load_kvquant_nuq_artifact_from_path,
    _load_mixkvq_thresholds,
    _mixkvq_assign_bits,
    _kvquant,
    _pmkvq_key_scale,
    maybe_kvquant_prerope_key,
)
from vllm.attention.ops.atc_kv_fake_quant.adapters import (
    kvquant_official_nuq,
    pmkvq_official_fake_quant,
)
from vllm.attention.ops.atc_kv_fake_quant.quant_utils import affine_quant_dequant
from vllm.attention.ops.atc_kv_fake_quant.trace import bit_summary


def test_asymmetric_min_offset():
    x = torch.tensor([
        [1.0, 1.2, 1.4, 1.6],
        [-2.0, -1.8, -1.6, -1.4],
        [-1.0, -0.3, 0.5, 1.0],
    ])
    y = affine_quant_dequant(x, 2, -1, symmetric=False)
    expected = torch.tensor([
        [1.0, 1.2, 1.4, 1.6],
        [-2.0, -1.8, -1.6, -1.4],
        [-1.0, -0.33333334, 0.33333334, 1.0],
    ])
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_kvtuner_preset_loaded():
    cfg = _load_kvtuner_config()
    assert len(cfg) == 28, f"expected 28 Qwen2.5-7B layers, got {len(cfg)}"
    avg = sum(v["nbits_key"] + v["nbits_value"] for v in cfg.values()) / (2 * len(cfg))
    assert 3.8 < avg < 4.1, f"unexpected KVTuner average bit width: {avg}"


def test_kvtuner_official_variants_loaded():
    old_env = dict(os.environ)
    try:
        base = Path("/root/atc_vllm_sched/references/kv_methods/KVTuner/calibration_presets")
        cases = [
            ("Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml",
             "pertoken", -1, 0, 4.0),
            ("Qwen2.5-7B-Instruct_kivi_KVTuner4_0.yaml",
             "kivi", 32, 32, 3.9285714285714284),
        ]
        for filename, mode, group_size, residual, expected_avg in cases:
            path = base / filename
            os.environ["ATC_KVTUNER_CONFIG_PATH"] = str(path)
            os.environ["ATC_KVTUNER_QUANT_MODE"] = mode
            os.environ["ATC_KVTUNER_GROUP_SIZE"] = str(group_size)
            os.environ["ATC_KVTUNER_RESIDUAL_TOKENS"] = str(residual)
            _load_kvtuner_config_from_path.cache_clear()
            cfg = _load_kvtuner_config()
            assert len(cfg) == 28
            avg = sum(v["nbits_key"] + v["nbits_value"]
                      for v in cfg.values()) / (2 * len(cfg))
            assert abs(avg - expected_avg) < 1e-6
            assert _kvtuner_quant_mode() == mode
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _load_kvtuner_config_from_path.cache_clear()


def test_bit_summary_has_average():
    bits = torch.tensor([2, 2, 4, 16], dtype=torch.int16)
    summary = bit_summary(bits, 2)
    assert summary["avg_bits"] == 6.0
    assert summary["bit_counts"] == {"2": 2, "4": 1, "16": 1}


def test_kvquant_nuq_artifact_loader_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nuq_artifact.pt"
        torch.save({
            "format": "atc_kvquant_nuq_v1",
            "artifact_complete": True,
            "layers": {
                "0": {
                    "key": {
                        "qchannel": 0,
                        "lut": [torch.tensor([[-1.0], [0.0], [1.0]])],
                    },
                    "value": {
                        "qchannel": -1,
                        "lut": [torch.tensor([[-1.0], [0.0], [1.0]])],
                    },
                }
            },
        }, path)
        _load_kvquant_nuq_artifact_from_path.cache_clear()
        obj = _load_kvquant_nuq_artifact_from_path(str(path))
        assert obj["format"] == "atc_kvquant_nuq_v1"
        assert "key" in obj["layers"]["0"]


def test_kvquant_official_nuq_toy_matches_direct_when_available():
    x = torch.tensor([[-1.0, -0.2, 0.3, 1.0],
                      [0.5, -0.5, 0.1, -0.1]], dtype=torch.float32)
    lut = [torch.tensor([[-1.0], [-0.25], [0.25], [1.0]])]
    y, source = kvquant_official_nuq(
        x, bits=2, qchannel=0, sparsity_threshold=0.99, lut=lut)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    if source.startswith("official"):
        from kvquant.simquant_module_quantizer import (  # type: ignore
            get_outliers_dynamic, quant_fn_nuq_recon)
        mask = get_outliers_dynamic(x, channel=0, thresh=0.99,
                                    first_few_fp16=-1)
        expected = quant_fn_nuq_recon(
            x,
            bits=2,
            qchannel=0,
            dynamicquantization=True,
            include_sparse=True,
            outlier_mask=mask,
            lut=lut,
            first_few_fp16=-1,
        ).to(dtype=x.dtype)
        torch.testing.assert_close(y, expected)


def test_mixkvq_threshold_artifact_and_strict_mode():
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps({
                "format": "atc_mixkvq_thresholds_v1",
                "selected": {
                    "tau_bf16": 2.0,
                    "tau_int4": 1.0,
                },
                "metadata": {
                    "dataset": "gsm8k",
                    "seed": 0,
                },
            }), encoding="utf-8")
            os.environ["ATC_MIXKVQ_THRESHOLDS_PATH"] = str(path)
            os.environ["ATC_MIXKVQ_STRICT_THRESHOLDS"] = "1"
            _load_mixkvq_thresholds.cache_clear()
            salience = torch.tensor([[0.5, 1.5, 3.0]], dtype=torch.float32)
            bits = _mixkvq_assign_bits(salience, 2.7)
            assert bits.tolist() == [[2, 4, 16]]

        os.environ["ATC_MIXKVQ_THRESHOLDS_PATH"] = str(Path(tmp) / "missing.json")
        os.environ["ATC_MIXKVQ_STRICT_THRESHOLDS"] = "1"
        _load_mixkvq_thresholds.cache_clear()
        try:
            _mixkvq_assign_bits(torch.ones(1, 3), 2.7)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict MixKVQ must fail without thresholds")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _load_mixkvq_thresholds.cache_clear()


class _FakeAttentionMetadata:

    def __init__(self, query_start, seq_lens, context_lens, block_tables):
        self.query_start_loc = torch.tensor(query_start, dtype=torch.int32)
        self.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
        self.context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32)
        self.block_tables = torch.tensor(block_tables, dtype=torch.int32)


class _FakeAttnContext:

    layer_name = "model.layers.0.self_attn"
    layer_idx = 0
    attn_type = "decoder"
    query = None
    cache_segments = [(0, 2)]
    cache_segment_positions = [(0, 2, 0, 2)]
    cache_block_tables = None


def test_kvquant_prerope_marker_skips_cache_key_quant():
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nuq_artifact.pt"
            lut = [torch.tensor([[-1.0], [-0.25], [0.25], [1.0]])]
            torch.save({
                "format": "atc_kvquant_nuq_v1",
                "artifact_complete": True,
                "nuq_artifact_complete": True,
                "bits": 2,
                "layers": {
                    "0": {
                        "key": {"qchannel": 0, "lut": lut},
                        "value": {"qchannel": -1, "lut": lut},
                    }
                },
            }, path)
            os.environ["ATC_KV_FAKE_QUANT_METHOD"] = "kvquant"
            os.environ["ATC_KVQUANT_PREROPE"] = "1"
            os.environ["ATC_KVQUANT_STRICT_ARTIFACT"] = "1"
            os.environ["ATC_KVQUANT_NUQ_ARTIFACT_PATH"] = str(path)
            os.environ["ATC_KVQUANT_BITS"] = "2"
            os.environ["ATC_KVQUANT_FIRST_TOKENS_FP16"] = "1"
            _load_kvquant_nuq_artifact_from_path.cache_clear()
            key = torch.tensor([[-1.0, -0.2, 0.3, 1.0],
                                [0.5, -0.5, 0.1, -0.1]])
            pre = maybe_kvquant_prerope_key(
                key, torch.tensor([0, 1]), "model.layers.0.self_attn")
            value = key.reshape(2, 1, 4).clone()
            qk, _qv, summary = _kvquant(
                pre.reshape(2, 1, 4), value, _FakeAttnContext(),
                serving=None, segments=[(0, 2)])
            torch.testing.assert_close(qk, pre.reshape(2, 1, 4))
            assert summary["pre_rope_applied"] is True
            assert summary["kvquant_mode"] == "prerope_nuq"
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _load_kvquant_nuq_artifact_from_path.cache_clear()


def _write_flash_cache(key_cache, value_cache, key, value, start_slot=0):
    for i in range(key.shape[0]):
        slot = start_slot + i
        block = slot // key_cache.shape[1]
        offset = slot % key_cache.shape[1]
        key_cache[block, offset].copy_(key[i])
        value_cache[block, offset].copy_(value[i])


def test_pmkvq_cachewide_rewrites_old_slots():
    _clear_pmkvq_cachewide_state()
    old_env = dict(os.environ)
    try:
        os.environ["ATC_KV_FAKE_QUANT_METHOD"] = "pmkvq_cachewide"
        os.environ["ATC_PMKVQ_CACHEWIDE_STRICT"] = "1"
        os.environ["ATC_PMKVQ_SINK_TOKENS"] = "1"
        os.environ["ATC_PMKVQ_WINDOW_TOKENS"] = "2"
        os.environ["ATC_PMKVQ_TARGET_AVG_BITS"] = "4"
        os.environ["ATC_PMKVQ_MIN_BITS"] = "2"
        os.environ["ATC_PMKVQ_CACHEWIDE_TIMING"] = "defer_current"
        os.environ.pop("ATC_PMKVQ_BUDGET_PATH", None)
        os.environ.pop("ATC_PMKVQ_REP_SCALES_PATH", None)

        key_cache = torch.zeros(2, 4, 1, 4)
        value_cache = torch.zeros_like(key_cache)
        key = torch.linspace(-1.0, 1.0, 6 * 4).reshape(6, 1, 4)
        value = torch.linspace(1.0, -1.0, 6 * 4).reshape(6, 1, 4)
        _write_flash_cache(key_cache, value_cache, key, value)
        meta = _FakeAttentionMetadata([0, 6], [6], [0], [[0, 1]])
        with attention_context("model.layers.0.self_attn", key, "decoder", meta):
            maybe_cachewide_fake_quant_kv(
                key, value, key_cache, value_cache, torch.arange(6), 4)
        ledger = next(iter(_PMKVQ_CACHEWIDE_LEDGER.values()))
        assert len(ledger) == 0
        pending = next(iter(_PMKVQ_CACHEWIDE_PENDING_POSITIONS.values()))
        assert list(pending.values()) == [list(range(6))]

        new_key = torch.tensor([[[0.5, 0.25, -0.25, -0.5]]])
        new_value = -new_key
        _write_flash_cache(key_cache, value_cache, new_key, new_value,
                           start_slot=6)
        decode_meta = _FakeAttentionMetadata([0, 1], [7], [6], [[0, 1]])
        with attention_context("model.layers.0.self_attn", new_key, "decoder",
                               decode_meta):
            maybe_cachewide_fake_quant_kv(
                new_key, new_value, key_cache, value_cache,
                torch.tensor([6]), 4)
        ledger = next(iter(_PMKVQ_CACHEWIDE_LEDGER.values()))
        assert len(ledger) == 6
        assert ledger[0] == 16
        assert ledger[5] == 2
        assert 6 not in ledger

        newer_key = torch.tensor([[[0.75, 0.5, -0.5, -0.75]]])
        newer_value = -newer_key
        _write_flash_cache(key_cache, value_cache, newer_key, newer_value,
                           start_slot=7)
        decode_meta = _FakeAttentionMetadata([0, 1], [8], [7], [[0, 1]])
        with attention_context("model.layers.0.self_attn", newer_key,
                               "decoder", decode_meta):
            maybe_cachewide_fake_quant_kv(
                newer_key, newer_value, key_cache, value_cache,
                torch.tensor([7]), 4)
        ledger = next(iter(_PMKVQ_CACHEWIDE_LEDGER.values()))
        assert len(ledger) == 7
        assert ledger[6] == 2
        assert 7 not in ledger
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _clear_pmkvq_cachewide_state()


def test_kivi_cachewide_residual_rewrites_global_slots():
    _clear_pmkvq_cachewide_state()
    old_env = dict(os.environ)
    try:
        os.environ["ATC_KV_FAKE_QUANT_METHOD"] = "kivi"
        os.environ["ATC_CACHEWIDE_RESIDUAL_STRICT"] = "1"
        os.environ["ATC_KIVI_K_BITS"] = "2"
        os.environ["ATC_KIVI_V_BITS"] = "2"
        os.environ["ATC_KIVI_GROUP_SIZE"] = "4"
        os.environ["ATC_KIVI_RESIDUAL_TOKENS"] = "2"

        key_cache = torch.zeros(2, 4, 1, 4)
        value_cache = torch.zeros_like(key_cache)
        key = torch.linspace(-1.0, 1.0, 6 * 4).reshape(6, 1, 4)
        value = torch.linspace(1.0, -1.0, 6 * 4).reshape(6, 1, 4)
        _write_flash_cache(key_cache, value_cache, key, value)
        meta = _FakeAttentionMetadata([0, 6], [6], [0], [[0, 1]])
        with attention_context("model.layers.0.self_attn", key, "decoder", meta):
            maybe_cachewide_fake_quant_kv(
                key, value, key_cache, value_cache, torch.arange(6), 4)
        assert all(not value for value in _RESIDUAL_CACHEWIDE_LEDGER.values())
        pending = next(iter(_RESIDUAL_CACHEWIDE_PENDING_POSITIONS.values()))
        assert list(pending.values()) == [list(range(6))]

        new_key = torch.tensor([[[0.5, 0.25, -0.25, -0.5]]])
        new_value = -new_key
        _write_flash_cache(key_cache, value_cache, new_key, new_value,
                           start_slot=6)
        decode_meta = _FakeAttentionMetadata([0, 1], [7], [6], [[0, 1]])
        with attention_context("model.layers.0.self_attn", new_key, "decoder",
                               decode_meta):
            maybe_cachewide_fake_quant_kv(
                new_key, new_value, key_cache, value_cache,
                torch.tensor([6]), 4)
        ledger = next(iter(_RESIDUAL_CACHEWIDE_LEDGER.values()))
        assert ledger[0] == "K2V2"
        assert ledger[4] == "K2V2"
        assert ledger[5] == "K2V16"
        assert 6 not in ledger

        newer_key = torch.tensor([[[0.75, 0.5, -0.5, -0.75]]])
        newer_value = -newer_key
        _write_flash_cache(key_cache, value_cache, newer_key, newer_value,
                           start_slot=7)
        decode_meta = _FakeAttentionMetadata([0, 1], [8], [7], [[0, 1]])
        with attention_context("model.layers.0.self_attn", newer_key,
                               "decoder", decode_meta):
            maybe_cachewide_fake_quant_kv(
                newer_key, newer_value, key_cache, value_cache,
                torch.tensor([7]), 4)
        ledger = next(iter(_RESIDUAL_CACHEWIDE_LEDGER.values()))
        assert ledger[5] == "K2V2"
        assert ledger[6] == "K2V16"
        assert 7 not in ledger
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _clear_pmkvq_cachewide_state()


def test_mixkvq_cachewide_uses_sink_and_lazy_residual_buffer():
    _clear_pmkvq_cachewide_state()
    old_env = dict(os.environ)
    try:
        os.environ["ATC_KV_FAKE_QUANT_METHOD"] = "mixkvq"
        os.environ["ATC_CACHEWIDE_RESIDUAL_STRICT"] = "1"
        os.environ["ATC_MIXKVQ_GROUP_SIZE"] = "4"
        os.environ["ATC_MIXKVQ_RESIDUAL_TOKENS"] = "2"
        os.environ["ATC_MIXKVQ_SINK_TOKENS"] = "1"
        os.environ["ATC_MIXKVQ_TARGET_BITS"] = "2.7"

        key_cache = torch.zeros(2, 4, 1, 4)
        value_cache = torch.zeros_like(key_cache)
        key = torch.linspace(-1.0, 1.0, 6 * 4).reshape(6, 1, 4)
        value = torch.linspace(1.0, -1.0, 6 * 4).reshape(6, 1, 4)
        _write_flash_cache(key_cache, value_cache, key, value)
        meta = _FakeAttentionMetadata([0, 6], [6], [0], [[0, 1]])
        with attention_context("model.layers.0.self_attn", key, "decoder",
                               meta):
            maybe_cachewide_fake_quant_kv(
                key, value, key_cache, value_cache, torch.arange(6), 4)
        assert all(not value for value in _RESIDUAL_CACHEWIDE_LEDGER.values())

        new_key = torch.tensor([[[0.5, 0.25, -0.25, -0.5]]])
        new_value = -new_key
        _write_flash_cache(key_cache, value_cache, new_key, new_value,
                           start_slot=6)
        decode_meta = _FakeAttentionMetadata([0, 1], [7], [6], [[0, 1]])
        with attention_context("model.layers.0.self_attn", new_key, "decoder",
                               decode_meta):
            maybe_cachewide_fake_quant_kv(
                new_key, new_value, key_cache, value_cache,
                torch.tensor([6]), 4)
        ledger = next(iter(_RESIDUAL_CACHEWIDE_LEDGER.values()))
        assert ledger[0] == "16"
        assert ledger[1] == "mixkvq_body"
        assert ledger[5] == "mixkvq_body"
        assert 6 not in ledger

        newer_key = torch.tensor([[[0.75, 0.5, -0.5, -0.75]]])
        newer_value = -newer_key
        _write_flash_cache(key_cache, value_cache, newer_key, newer_value,
                           start_slot=7)
        decode_meta = _FakeAttentionMetadata([0, 1], [8], [7], [[0, 1]])
        with attention_context("model.layers.0.self_attn", newer_key,
                               "decoder", decode_meta):
            maybe_cachewide_fake_quant_kv(
                newer_key, newer_value, key_cache, value_cache,
                torch.tensor([7]), 4)
        ledger = next(iter(_RESIDUAL_CACHEWIDE_LEDGER.values()))
        assert ledger[0] == "16"
        assert ledger[6] == "mixkvq_body"
        assert 7 not in ledger
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        _clear_pmkvq_cachewide_state()


def test_kvquant_first_token_is_global_sink_only():
    old_env = dict(os.environ)
    try:
        os.environ["ATC_KV_FAKE_QUANT_METHOD"] = "kvquant"
        os.environ["ATC_KVQUANT_BITS"] = "3"
        os.environ["ATC_KVQUANT_FIRST_TOKENS_FP16"] = "1"
        os.environ["ATC_KVQUANT_OUTLIER_RATIO"] = "0"
        os.environ["ATC_KVQUANT_USE_NUQ"] = "1"
        key = torch.tensor([[[0.1, 0.3, 0.8, 1.7, -0.4, -1.2, 0.5, 2.0]]])
        value = -key

        first_meta = _FakeAttentionMetadata([0, 1], [1], [0], [[0]])
        with attention_context("model.layers.0.self_attn", key, "decoder",
                               first_meta):
            q_key, q_value = maybe_fake_quant_kv(
                key, value, "auto", torch.tensor([0]), 4)
        torch.testing.assert_close(q_key, key)
        torch.testing.assert_close(q_value, value)

        decode_meta = _FakeAttentionMetadata([0, 1], [101], [100], [[0]])
        with attention_context("model.layers.0.self_attn", key, "decoder",
                               decode_meta):
            q_key, q_value = maybe_fake_quant_kv(
                key, value, "auto", torch.tensor([0]), 4)
        assert not torch.equal(q_value, value)
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_pmkvq_official_quantizer_imports():
    x = torch.stack([
        torch.linspace(0.1, 2.0, 128),
        torch.linspace(-2.0, -0.1, 128),
        torch.linspace(-1.0, 1.0, 128),
    ]).reshape(3, 1, 128)
    for bits in (2, 4, 8):
        y, source = pmkvq_official_fake_quant(x, bits)
        assert source.startswith("official"), source
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert not torch.isnan(y).any()


def test_pmkvq_rep_scale_strict_shape():
    path = "/root/atc_vllm_sched/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192/rep_scales_k4v4.pt"
    if not os.path.exists(path):
        return
    old_env = dict(os.environ)
    try:
        os.environ["ATC_PMKVQ_REP_SCALES_PATH"] = path
        os.environ["ATC_PMKVQ_REP_SCALES_STRICT"] = "1"
        scale = _pmkvq_key_scale(
            0, torch.device("cpu"), torch.float32, torch.Size([1, 4, 128]))
        assert scale is not None
        assert tuple(scale.shape) == (1, 4, 128)
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_kv_cache_gather_scatter_layouts():
    key4 = torch.arange(2 * 4 * 1 * 4, dtype=torch.float32).reshape(2, 4, 1, 4)
    val4 = -key4.clone()
    gathered_k, gathered_v = _kv_cache_gather_tokens(
        key4, val4, [0, 5], 4, "flash4")
    assert gathered_k.shape == (2, 1, 4)
    _kv_cache_scatter_tokens(key4, val4, [0, 5],
                             torch.zeros_like(gathered_k),
                             torch.zeros_like(gathered_v), 4, "flash4")
    assert torch.count_nonzero(key4[0, 0]) == 0
    assert torch.count_nonzero(val4[1, 1]) == 0

    key5 = torch.arange(2 * 1 * 2 * 4 * 2,
                        dtype=torch.float32).reshape(2, 1, 2, 4, 2)
    val5 = torch.arange(2 * 1 * 4 * 4,
                        dtype=torch.float32).reshape(2, 1, 4, 4)
    gathered_k, gathered_v = _kv_cache_gather_tokens(
        key5, val5, [5], 4, "paged5")
    assert gathered_k.shape == (1, 1, 4)
    assert gathered_v.shape == (1, 1, 4)
    _kv_cache_scatter_tokens(key5, val5, [5],
                             torch.zeros_like(gathered_k),
                             torch.zeros_like(gathered_v), 4, "paged5")
    assert torch.count_nonzero(key5[1, :, :, 1, :]) == 0
    assert torch.count_nonzero(val5[1, :, :, 1]) == 0


def main():
    test_asymmetric_min_offset()
    test_kvtuner_preset_loaded()
    test_kvtuner_official_variants_loaded()
    test_bit_summary_has_average()
    test_pmkvq_cachewide_rewrites_old_slots()
    test_kivi_cachewide_residual_rewrites_global_slots()
    test_mixkvq_cachewide_uses_sink_and_lazy_residual_buffer()
    test_kvquant_first_token_is_global_sink_only()
    test_pmkvq_official_quantizer_imports()
    test_pmkvq_rep_scale_strict_shape()
    test_kv_cache_gather_scatter_layouts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(0)
    key = torch.randn(257, 4, 32, device=device, dtype=dtype)
    value = torch.randn(257, 4, 32, device=device, dtype=dtype)
    results = {}
    for method in [
        "none", "kivi", "kvtuner", "kvquant", "pmkvq", "mixkvq",
        "pmkvq_serving", "pmkvq_cachewide", "mixkvq_serving",
    ]:
        saved_rep_scales = os.environ.get("ATC_PMKVQ_REP_SCALES_PATH")
        if method.startswith("pmkvq") and key.shape[-1] != 128:
            os.environ.pop("ATC_PMKVQ_REP_SCALES_PATH", None)
        os.environ["ATC_KV_FAKE_QUANT_METHOD"] = method
        os.environ["ATC_KV_FAKE_QUANT_PRESSURE"] = "medium"
        try:
            q_key, q_value = maybe_fake_quant_kv(key, value, "auto")
        finally:
            if saved_rep_scales is not None:
                os.environ["ATC_PMKVQ_REP_SCALES_PATH"] = saved_rep_scales
        assert q_key.shape == key.shape and q_value.shape == value.shape
        assert q_key.dtype == key.dtype and q_value.dtype == value.dtype
        assert not torch.isnan(q_key.float()).any()
        assert not torch.isnan(q_value.float()).any()
        results[method] = {
            "key_shape": list(q_key.shape),
            "value_shape": list(q_value.shape),
            "key_dtype": str(q_key.dtype),
            "value_dtype": str(q_value.dtype),
            "key_mean_abs_diff": float((q_key - key).abs().mean().item()),
            "value_mean_abs_diff": float((q_value - value).abs().mean().item()),
        }
    print(json.dumps({
        "device": device,
        "dtype": str(dtype),
        "methods": results,
    },
                     indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
