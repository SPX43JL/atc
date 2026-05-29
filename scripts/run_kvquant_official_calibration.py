#!/usr/bin/env python3
"""KVQuant-style Fisher calibration for Qwen2.5 KV fake quant.

The official KVQuant Fisher script is LLaMA/Mistral-fork specific: it relies on
projection modules exposing ``.act`` and ``set_devices``.  This driver keeps the
official calibration objective and Wikitext-2 16x2K setup, but adds a Qwen2.5
runtime wrapper for K/V projections without modifying the official repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LinearAct(nn.Linear):
    """Linear layer that records activation and its gradient like KVQuant."""

    def __init__(self, source: nn.Linear):
        super().__init__(source.in_features, source.out_features,
                         bias=source.bias is not None,
                         device=source.weight.device,
                         dtype=source.weight.dtype)
        self.weight.data.copy_(source.weight.data)
        if source.bias is not None and self.bias is not None:
            self.bias.data.copy_(source.bias.data)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.act: torch.Tensor | None = None
        self.retain_act_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        y = F.linear(x, self.weight, self.bias)
        if self.retain_act_grad:
            y.retain_grad()
        self.act = y
        return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--kvquant-repo", required=True)
    p.add_argument("--calib-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-examples", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    return p.parse_args()


def repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def load_tokenized_samples(path: Path, limit: int,
                           seq_len: int) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            ids = obj.get("input_ids") or obj.get("ids") or obj.get("tokens")
            if not isinstance(ids, list):
                continue
            ids = [int(v) for v in ids[:seq_len]]
            if len(ids) < seq_len:
                continue
            samples.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    if len(samples) < limit:
        raise RuntimeError(
            f"only found {len(samples)} usable {seq_len}-token samples in {path}")
    return samples


def replace_kv_projections(model: nn.Module) -> list[tuple[int, str, LinearAct]]:
    wrappers: list[tuple[int, str, LinearAct]] = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        for name in ("k_proj", "v_proj"):
            original = getattr(layer.self_attn, name)
            wrapped = LinearAct(original)
            setattr(layer.self_attn, name, wrapped)
            wrappers.append((layer_idx, name, wrapped))
    return wrappers


def iter_wrappers(wrappers: Iterable[tuple[int, str, LinearAct]],
                  name: str) -> Iterable[tuple[int, LinearAct]]:
    for layer_idx, proj_name, module in wrappers:
        if proj_name == name:
            yield layer_idx, module


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kvquant_repo = Path(args.kvquant_repo)
    calib_jsonl = Path(args.calib_jsonl)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = load_tokenized_samples(calib_jsonl, args.num_examples,
                                     args.seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.config._attn_implementation = args.attn_implementation
    model.requires_grad_(False)
    wrappers = replace_kv_projections(model)
    model.to(args.device)
    model.train()

    fisher_sum: dict[str, torch.Tensor] = {}
    abs_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for sample_idx, input_ids in enumerate(samples):
        input_ids = input_ids.to(args.device)
        model.zero_grad(set_to_none=True)
        for _layer_idx, _name, module in wrappers:
            module.act = None
            module.retain_act_grad = True
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        outputs.loss.backward()
        for layer_idx, name, module in wrappers:
            if module.act is None or module.act.grad is None:
                raise RuntimeError(
                    f"missing activation grad for layer {layer_idx} {name}")
            act = module.act.detach().float().cpu()
            grad2 = module.act.grad.detach().float().pow(2).cpu()
            key = f"{name}_layer_{layer_idx}"
            reduce_dims = tuple(range(grad2.ndim - 1))
            fisher = grad2.sum(dim=reduce_dims)
            abs_act = act.abs().sum(dim=reduce_dims)
            fisher_sum[key] = fisher_sum.get(key, torch.zeros_like(fisher)) + fisher
            abs_sum[key] = abs_sum.get(key, torch.zeros_like(abs_act)) + abs_act
            counts[key] = counts.get(key, 0) + int(grad2.numel() // grad2.shape[-1])
        del outputs
        torch.cuda.empty_cache()
        print(f"calibrated sample {sample_idx + 1}/{len(samples)}", flush=True)

    summary: dict[str, torch.Tensor] = {}
    for key, total in fisher_sum.items():
        denom = max(1, counts[key])
        summary[f"{key}_fisher_mean"] = total / denom
        summary[f"{key}_activation_abs_mean"] = abs_sum[key] / denom

    torch.save(summary, out_dir / "fisher_summary.pt")
    metadata = {
        "source": "kvquant_official_fisher_objective_qwen25_wrapper",
        "kvquant_repo": str(kvquant_repo),
        "kvquant_commit": repo_commit(kvquant_repo),
        "model_path": args.model_path,
        "calib_jsonl": str(calib_jsonl),
        "dataset": "Wikitext-2 prepared from Salesforce/wikitext",
        "num_examples": args.num_examples,
        "seq_len": args.seq_len,
        "max_seq_len": args.max_seq_len,
        "artifact_files": ["fisher_summary.pt", "metadata.json"],
        "wrapper_note": (
            "Qwen2.5 projection modules are wrapped to expose KVQuant-style "
            "activation gradients; the official repository itself is not edited."),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
