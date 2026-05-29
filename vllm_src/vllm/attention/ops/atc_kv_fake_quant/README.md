# ATC KV Fake Quant Implementation Layout

This directory contains the Python fake-quant approximation used by the ATC
vLLM serving experiments. It intentionally keeps quantize-dequantize tensors in
the normal vLLM cache path; it does not provide packed low-bit KV storage,
low-bit cache kernels, real memory savings, or serving acceleration.

## Entry Points

- `core.py` is the unified vLLM hook surface.
  - `maybe_fake_quant_kv(...)` runs pre-cache-write fake quant for methods that
    operate on the current key/value tensors.
  - `maybe_cachewide_fake_quant_kv(...)` runs post-cache-write cache-wide
    rewrites for methods whose residual/sink policy needs global cache state.
  - `maybe_kvquant_prerope_key(...)` is re-exported for the Qwen2 KVQuant
    pre-RoPE key hook.
- `common.py` contains shared serving, sequence, slot-mapping, cache-layout, and
  trace-accounting helpers used by multiple methods.
- `quant_utils.py`, `runtime.py`, `trace.py`, and `adapters.py` provide shared
  quantization primitives, per-request context, trace output, and paper/repo
  reference metadata.

## Method Modules

- `methods/kivi.py`: KIVI-style residual-window fake quant with K per-channel
  and V per-token grouping.
- `methods/kvtuner.py`: KVTuner presets, including per-token C4.00 and
  KVTuner-KIVI C3.92 configurations.
- `methods/kvquant.py`: KVQuant NUQ3 approximation, effective first-token
  FP16/outlier accounting, and the Qwen2 pre-RoPE key hook.
- `methods/pmkvq.py`: PM-KVQ per-token mixed precision and cache-wide
  progressive rewrite approximation.
- `methods/residual_cachewide.py`: shared post-write residual/cache-wide rewrite
  path used by KIVI, KVTuner, and MixKVQ-style policies.
- `methods/mixkvq.py`: diagnostic MixKVQ approximation kept for inspection but
  not part of the current formal main table.

The module split is organizational only: it preserves the public hook behavior
while making each paper method easier to audit against its own config, artifact,
and official-repo reference notes.
