#!/usr/bin/env python3
"""Build a KVQuant NUQ artifact for Qwen2.5 from Wikitext-2 calibration.

This driver keeps the official KVQuant simulated quantizer in the loop while
adding the Qwen2.5 projection wrappers needed by this repository.  It writes a
runtime artifact consumed by the vLLM fake-quant hook; it does not edit the
official KVQuant repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LinearAct(nn.Linear):
    """Linear layer that records activation and activation gradients."""

    def __init__(self, source: nn.Linear):
        super().__init__(source.in_features,
                         source.out_features,
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
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--sparsity-threshold", type=float, default=0.99)
    p.add_argument("--first-few-fp16", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args()


def repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
            if len(ids) == seq_len:
                samples.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    if len(samples) < limit:
        raise RuntimeError(
            f"found {len(samples)} usable {seq_len}-token samples in {path}, "
            f"expected {limit}")
    return samples


def replace_kv_projections(model: nn.Module) -> list[tuple[int, str, LinearAct]]:
    wrappers: list[tuple[int, str, LinearAct]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        for name in ("k_proj", "v_proj"):
            original = getattr(layer.self_attn, name)
            wrapped = LinearAct(original)
            setattr(layer.self_attn, name, wrapped)
            wrappers.append((layer_idx, name, wrapped))
    return wrappers


def iter_wrappers(wrappers: Iterable[tuple[int, str, LinearAct]],
                  proj_name: str) -> Iterable[tuple[int, LinearAct]]:
    for layer_idx, name, module in wrappers:
        if name == proj_name:
            yield layer_idx, module


def load_official_simquant(kvquant_repo: Path):
    for subdir in ("quant", "benchmarking", "deployment"):
        root = kvquant_repo / subdir
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from kvquant.simquant_module_quantizer import SimQuant  # type: ignore
            return SimQuant, f"official:{subdir}"
        except Exception:
            sys.modules.pop("kvquant.simquant_module_quantizer", None)
            sys.modules.pop("kvquant", None)
    raise RuntimeError("could not import KVQuant SimQuant from official repo")


def save_shard(path: Path, act: torch.Tensor, fisher: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"act": act.cpu(), "fisher": fisher.cpu()}, path)


def load_concat_shards(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    acts: list[torch.Tensor] = []
    fishers: list[torch.Tensor] = []
    for path in paths:
        obj = torch.load(path, map_location="cpu")
        acts.append(obj["act"].float())
        fishers.append(obj["fisher"].float())
    return torch.cat(acts, dim=0), torch.cat(fishers, dim=0)


def squeeze_threshold(x: object) -> object:
    if isinstance(x, torch.Tensor):
        while x.ndim > 0 and x.shape[0] == 1:
            x = x.squeeze(0)
        while x.ndim > 0 and x.shape[-1] == 1:
            x = x.squeeze(-1)
    return x


def build_entry(SimQuant, bits: int, qchannel: int, act: torch.Tensor,
                fisher: torch.Tensor, sparsity_threshold: float,
                first_few_fp16: int) -> dict:
    dummy = nn.Linear(act.shape[-1], act.shape[-1], bias=False)
    sim = SimQuant(dummy, bits, perchannel=True, qchannel=qchannel)
    sim.out = act.float()
    sim.nsamples = max(1, act.shape[0] // 2048)
    result = sim.quantize(
        include_sparse=True,
        sparsity_threshold=sparsity_threshold,
        nuq=True,
        fisher=fisher.float(),
        norm=False,
        cap_outliers=False,
        first_few_fp16=first_few_fp16,
    )
    upper, lower, centroids = result[:3]
    return {
        "qchannel": qchannel,
        "lut": centroids,
        "outlier_threshold_upper": squeeze_threshold(upper),
        "outlier_threshold_lower": squeeze_threshold(lower),
        "sparsity_threshold": sparsity_threshold,
        "first_few_fp16": first_few_fp16,
        "num_calibration_vectors": int(act.shape[0]),
        "hidden_dim": int(act.shape[-1]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_kvquant_nuq_shards"
    if tmp_dir.exists() and not args.keep_temp:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    kvquant_repo = Path(args.kvquant_repo)
    SimQuant, simquant_source = load_official_simquant(kvquant_repo)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    samples = load_tokenized_samples(Path(args.calib_jsonl),
                                     args.num_examples, args.seq_len)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    shard_paths: dict[str, list[Path]] = {}
    for sample_idx, input_ids in enumerate(samples):
        input_ids = input_ids.to(args.device)
        model.zero_grad(set_to_none=True)
        for _layer_idx, _name, module in wrappers:
            module.act = None
            module.retain_act_grad = True
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        outputs.loss.backward()
        for layer_idx, proj_name, module in wrappers:
            if module.act is None or module.act.grad is None:
                raise RuntimeError(
                    f"missing activation grad for layer {layer_idx} {proj_name}")
            key = f"layer_{layer_idx}_{proj_name}"
            act = module.act.detach().float().reshape(-1, module.act.shape[-1])
            fisher = module.act.grad.detach().float().pow(2).reshape_as(act)
            path = tmp_dir / f"{key}_sample_{sample_idx:04d}.pt"
            save_shard(path, act, fisher)
            shard_paths.setdefault(key, []).append(path)
        del outputs
        torch.cuda.empty_cache()
        print(f"captured sample {sample_idx + 1}/{len(samples)}", flush=True)

    layers: dict[str, dict[str, dict]] = {}
    for layer_idx, _module in iter_wrappers(wrappers, "k_proj"):
        key = f"layer_{layer_idx}_k_proj"
        act, fisher = load_concat_shards(shard_paths[key])
        layers.setdefault(str(layer_idx), {})["key"] = build_entry(
            SimQuant, args.bits, 0, act, fisher, args.sparsity_threshold,
            args.first_few_fp16)
        del act, fisher
        key = f"layer_{layer_idx}_v_proj"
        act, fisher = load_concat_shards(shard_paths[key])
        layers.setdefault(str(layer_idx), {})["value"] = build_entry(
            SimQuant, args.bits, -1, act, fisher, args.sparsity_threshold,
            args.first_few_fp16)
        del act, fisher

    artifact = {
        "format": "atc_kvquant_nuq_v1",
        "artifact_complete": True,
        "nuq_artifact_complete": True,
        "bits": args.bits,
        "sparsity_threshold": args.sparsity_threshold,
        "first_few_fp16": args.first_few_fp16,
        "layers": layers,
    }
    artifact_path = out_dir / "nuq_artifact.pt"
    torch.save(artifact, artifact_path)
    metadata = {
        "source": "kvquant_official_simquant_nuq_qwen25_wrapper",
        "kvquant_repo": str(kvquant_repo),
        "kvquant_commit": repo_commit(kvquant_repo),
        "simquant_source": simquant_source,
        "model_path": args.model_path,
        "calib_jsonl": args.calib_jsonl,
        "calib_jsonl_sha256": sha256_file(Path(args.calib_jsonl)),
        "sample_indices": list(range(args.num_examples)),
        "dataset": "Wikitext-2 prepared from Salesforce/wikitext",
        "num_examples": args.num_examples,
        "seq_len": args.seq_len,
        "bits": args.bits,
        "sparsity_threshold": args.sparsity_threshold,
        "first_few_fp16": args.first_few_fp16,
        "artifact_files": ["nuq_artifact.pt", "metadata.json"],
        "nuq_artifact_sha256": sha256_file(artifact_path),
        "nuq_artifact_complete": True,
        "runtime_note": (
            "Qwen2 pre-RoPE K fake quant consumes the key entries; vLLM "
            "reshape_and_cache consumes the value entries."),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if not args.keep_temp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
