"""Shared fake-quant utilities.

All functions return dequantized tensors in the original dtype.  They do not
pack data and do not alter vLLM's KV cache allocation.
"""

from __future__ import annotations

import math

import torch


def clamp_bits(bits: int) -> int:
    return max(1, min(int(bits), 16))


def affine_quant_dequant(x: torch.Tensor,
                         bits: int,
                         reduce_dim: int,
                         symmetric: bool = False,
                         eps: float = 1e-6) -> torch.Tensor:
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    xf = x.float()
    if symmetric:
        qmax = float((1 << (bits - 1)) - 1)
        qmin = -float(1 << (bits - 1))
        scale = xf.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(eps) / qmax
        q = torch.round(xf / scale).clamp(qmin, qmax)
        out = q * scale
    else:
        # KIVI/KVQuant use min/max asymmetric quantization without clamping the
        # real-valued offset.  Clamping the zero point collapses all-positive or
        # all-negative groups and was the main source of over-strong fake quant.
        qmin = 0.0
        qmax = float((1 << bits) - 1)
        xmin = xf.amin(dim=reduce_dim, keepdim=True)
        xmax = xf.amax(dim=reduce_dim, keepdim=True)
        scale = (xmax - xmin).clamp_min(eps) / qmax
        q = torch.round((xf - xmin) / scale).clamp(qmin, qmax)
        out = q * scale + xmin
    return out.to(dtype=x.dtype)


def quantize_last_dim_groups(x: torch.Tensor,
                             bits: int,
                             group_size: int,
                             symmetric: bool = False) -> torch.Tensor:
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    group_size = _resolve_group_size(x.shape[-1], group_size)
    last = x.shape[-1]
    pad = (group_size - last % group_size) % group_size
    xf = x
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    grouped = xf.reshape(*xf.shape[:-1], -1, group_size)
    out = affine_quant_dequant(grouped, bits, -1, symmetric=symmetric)
    out = out.reshape(*xf.shape[:-1], -1)
    if pad:
        out = out[..., :last]
    return out.to(dtype=x.dtype)


def quantize_token_groups_per_channel(x: torch.Tensor,
                                      bits: int,
                                      group_size: int,
                                      symmetric: bool = False) -> torch.Tensor:
    """KIVI-style key quantization: groups along token dim, per channel."""
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    if x.ndim != 3:
        return affine_quant_dequant(x, bits, 0, symmetric=symmetric)
    tokens = x.shape[0]
    group_size = _resolve_group_size(tokens, group_size)
    full = tokens // group_size * group_size
    if full == 0:
        return x
    grouped = x[:full].reshape(full // group_size, group_size, x.shape[1],
                               x.shape[2])
    q = affine_quant_dequant(grouped, bits, 1, symmetric=symmetric)
    out = x.clone()
    out[:full] = q.reshape_as(x[:full])
    return out


def keep_recent(original: torch.Tensor, quantized: torch.Tensor,
                residual_tokens: int) -> torch.Tensor:
    residual_tokens = max(0, int(residual_tokens))
    if residual_tokens == 0 or original.shape[0] <= residual_tokens:
        return quantized
    out = quantized.clone()
    out[-residual_tokens:] = original[-residual_tokens:]
    return out


def split_recent(x: torch.Tensor, residual_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    residual_tokens = max(0, int(residual_tokens))
    if residual_tokens == 0:
        return x, x[:0]
    if x.shape[0] <= residual_tokens:
        return x[:0], x
    return x[:-residual_tokens], x[-residual_tokens:]


def dense_sparse_quant(x: torch.Tensor,
                       bits: int,
                       reduce_dim: int,
                       outlier_ratio: float,
                       symmetric: bool = False,
                       first_tokens_fp16: int = 0) -> torch.Tensor:
    """KVQuant-style dense-and-sparse simulated quantization."""
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    outlier_ratio = max(0.0, min(float(outlier_ratio), 0.5))
    if outlier_ratio <= 0:
        q = affine_quant_dequant(x, bits, reduce_dim, symmetric=symmetric)
    else:
        xf = x.float()
        # Dynamic percentile outliers.  Compute a global mask to keep the
        # implementation affordable in Python while preserving the paper's
        # dense/sparse effect.
        flat = xf.abs().flatten()
        k = max(1, int(math.ceil(flat.numel() * outlier_ratio)))
        threshold = flat.topk(k).values.min()
        mask = xf.abs() >= threshold
        median = xf.median()
        trimmed = torch.where(mask, median, xf)
        q = affine_quant_dequant(trimmed.to(dtype=x.dtype), bits, reduce_dim,
                                 symmetric=symmetric).float()
        q = torch.where(mask, xf, q)
        q = q.to(dtype=x.dtype)
    if first_tokens_fp16 > 0 and q.ndim >= 1:
        q = q.clone()
        q[:first_tokens_fp16] = x[:first_tokens_fp16]
    return q


def dense_sparse_normal_float_quant(x: torch.Tensor,
                                    bits: int,
                                    reduce_dim: int,
                                    outlier_ratio: float,
                                    first_tokens_fp16: int = 0) -> torch.Tensor:
    """KVQuant-style dense/sparse wrapper around the local NF approximation."""
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    outlier_ratio = max(0.0, min(float(outlier_ratio), 0.5))
    xf = x.float()
    if outlier_ratio <= 0:
        q = normal_float_quant(x, bits, reduce_dim)
    else:
        flat = xf.abs().flatten()
        k = max(1, int(math.ceil(flat.numel() * outlier_ratio)))
        threshold = flat.topk(k).values.min()
        mask = xf.abs() >= threshold
        median = xf.median()
        trimmed = torch.where(mask, median, xf)
        q = normal_float_quant(trimmed.to(dtype=x.dtype), bits, reduce_dim).float()
        q = torch.where(mask, xf, q).to(dtype=x.dtype)
    if first_tokens_fp16 > 0 and q.ndim >= 1:
        q = q.clone()
        q[:first_tokens_fp16] = x[:first_tokens_fp16]
    return q


def normal_float_quant(x: torch.Tensor, bits: int,
                       reduce_dim: int) -> torch.Tensor:
    """Cheap NF/NUQ approximation using normal quantile signposts."""
    bits = clamp_bits(bits)
    if bits >= 16 or x.numel() == 0:
        return x
    levels = 1 << bits
    device = x.device
    dtype = torch.float32
    probs = torch.linspace(0.5 / levels, 1.0 - 0.5 / levels, levels,
                           device=device, dtype=dtype)
    lut = torch.distributions.Normal(0, 1).icdf(probs)
    lut = lut / lut.abs().max().clamp_min(1e-6)
    xf = x.float()
    center = (xf.amax(dim=reduce_dim, keepdim=True) +
              xf.amin(dim=reduce_dim, keepdim=True)) / 2
    radius = (xf.amax(dim=reduce_dim, keepdim=True) -
              xf.amin(dim=reduce_dim, keepdim=True)).clamp_min(1e-6) / 2
    normalized = ((xf - center) / radius).clamp(-1, 1)
    idx = (normalized.flatten().unsqueeze(-1) - lut).abs().argmin(dim=-1)
    q = lut[idx].reshape_as(xf)
    return (q * radius + center).to(dtype=x.dtype)


def _resolve_group_size(length: int, group_size: int) -> int:
    if group_size is None or int(group_size) <= 0:
        return max(1, int(length))
    return max(1, int(group_size))
