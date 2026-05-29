"""Python-only KV cache fake quantization hooks for ATC serving studies."""

from vllm.attention.ops.atc_kv_fake_quant.core import (
    attention_context,
    maybe_cachewide_fake_quant_kv,
    maybe_fake_quant_kv,
    maybe_kvquant_prerope_key,
)

__all__ = [
    "maybe_fake_quant_kv",
    "maybe_cachewide_fake_quant_kv",
    "maybe_kvquant_prerope_key",
    "attention_context",
]
