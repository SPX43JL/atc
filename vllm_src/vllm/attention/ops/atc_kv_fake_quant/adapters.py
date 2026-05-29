"""Thin adapters around official related-work repositories.

The serving experiment stays inside vLLM's Python cache-write path, so most
official CUDA/cache classes cannot be used directly.  These helpers import
official pure-Python quantizer functions when their dependencies are available
and otherwise report the fallback explicitly through trace metadata.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


PROJECT_DIR = Path("/root/atc_vllm_sched")
KIVI_REPO = PROJECT_DIR / "references/kv_methods/KIVI"
KVTUNER_REPO = PROJECT_DIR / "references/kv_methods/KVTuner"
KVQUANT_REPO = PROJECT_DIR / "references/kv_methods/KVQuant"
PMKVQ_REPO = PROJECT_DIR / "references/kv_methods/PM-KVQ"

REFERENCE_COMMITS = {
    "kivi": "876b4d2",
    "kvtuner": "96dd05e",
    "kvquant": "57a2383",
    "pmkvq": "c8a2bff",
}


@lru_cache(maxsize=1)
def _load_kvquant_official(
) -> tuple[Any | None, Any | None, Any | None, str]:
    errors: list[str] = []
    for subdir in ("benchmarking", "deployment", "quant"):
        quant_root = KVQUANT_REPO / subdir
        if str(quant_root) not in sys.path:
            sys.path.insert(0, str(quant_root))
        try:
            from kvquant.simquant_module_quantizer import (  # type: ignore
                get_outliers_dynamic,
                quant_fn_nuq_recon,
                quant_fn_zp,
            )
            return (quant_fn_zp, quant_fn_nuq_recon, get_outliers_dynamic,
                    f"official:{subdir}")
        except Exception as exc:
            errors.append(f"{subdir}:{type(exc).__name__}:{exc}")
            sys.modules.pop("kvquant.simquant_module_quantizer", None)
            sys.modules.pop("kvquant", None)
    return None, None, None, "fallback:" + "|".join(errors)


def kvquant_official_zp(
    x: torch.Tensor,
    bits: int,
    qchannel: int,
    outlier_ratio: float,
    first_tokens_fp16: int = 0,
) -> tuple[torch.Tensor, str]:
    """Run KVQuant's official simulated integer quantizer when importable."""
    quant_fn_zp, _quant_fn_nuq_recon, get_outliers_dynamic, source = (
        _load_kvquant_official())
    if quant_fn_zp is None or get_outliers_dynamic is None:
        return x, source

    original_dtype = x.dtype
    inp = x.float()
    threshold = 1.0 - max(0.0, min(float(outlier_ratio), 0.5))
    mask = get_outliers_dynamic(
        inp,
        channel=qchannel,
        thresh=threshold,
        first_few_fp16=-1,
    )
    out = quant_fn_zp(
        inp,
        bits=int(bits),
        qchannel=qchannel,
        dynamicquantization=True,
        include_sparse=bool(outlier_ratio > 0),
        outlier_mask=mask,
        clamp=False,
    )
    if first_tokens_fp16 > 0 and out.ndim >= 1:
        out = out.clone()
        out[:first_tokens_fp16] = inp[:first_tokens_fp16]
    return out.to(dtype=original_dtype), source


def kvquant_official_nuq(
    x: torch.Tensor,
    bits: int,
    qchannel: int,
    sparsity_threshold: float,
    lut: object,
    first_tokens_fp16: int = 0,
) -> tuple[torch.Tensor, str]:
    """Run KVQuant's official NUQ simulated quantizer when importable."""
    _quant_fn_zp, quant_fn_nuq_recon, get_outliers_dynamic, source = (
        _load_kvquant_official())
    if quant_fn_nuq_recon is None or get_outliers_dynamic is None:
        return x, source

    original_dtype = x.dtype
    inp = x.float()
    sparsity_threshold = max(0.0, min(float(sparsity_threshold), 1.0))
    mask = get_outliers_dynamic(
        inp,
        channel=qchannel,
        thresh=sparsity_threshold,
        first_few_fp16=-1,
    )
    lut_arg = lut
    if isinstance(lut, torch.Tensor):
        lut_arg = [lut.detach().cpu()]
    elif isinstance(lut, (list, tuple)):
        if len(lut) > 0 and isinstance(lut[0], torch.Tensor):
            lut_arg = [lut[0].detach().cpu()]
        else:
            lut_arg = lut
    out = quant_fn_nuq_recon(
        inp,
        bits=int(bits),
        qchannel=qchannel,
        dynamicquantization=True,
        include_sparse=True,
        outlier_mask=mask,
        lut=lut_arg,
        first_few_fp16=-1,
    )
    if first_tokens_fp16 > 0 and out.ndim >= 1:
        out = out.clone()
        out[:first_tokens_fp16] = inp[:first_tokens_fp16]
    return out.to(dtype=original_dtype), source


@lru_cache(maxsize=8)
def _load_pmkvq_quantizer(bits: int) -> tuple[Any | None, str]:
    if str(PMKVQ_REPO) not in sys.path:
        sys.path.insert(0, str(PMKVQ_REPO))
    try:
        from pm_kvq.quantization.quantizer.quantizer import (  # type: ignore
            UntrainableQuantizer,
        )

        quantizer = UntrainableQuantizer(
            n_bits=int(bits),
            granularity="per_group",
            symmetric=False,
            group_size=128,
            round_zeros=False,
        )
        return quantizer, "official:pm_kvq.UntrainableQuantizer"
    except Exception as exc:
        return None, f"fallback:{type(exc).__name__}:{exc}"


def pmkvq_official_fake_quant(
    x: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, str]:
    """Run PM-KVQ's official fake quantizer for one bit-width if importable."""
    if int(bits) >= 16 or x.numel() == 0:
        return x, "identity:bf16_fp16"
    if x.shape[-1] % 128 != 0:
        return x, "fallback:head_dim_not_multiple_128"
    quantizer, source = _load_pmkvq_quantizer(int(bits))
    if quantizer is None:
        return x, source
    quantizer = quantizer.to(device=x.device)
    return quantizer(x).to(dtype=x.dtype), source


def reference_source(method: str) -> dict[str, str]:
    return {
        "repo_commit": REFERENCE_COMMITS.get(method, ""),
        "adapter": "official_repo_adapter",
    }
