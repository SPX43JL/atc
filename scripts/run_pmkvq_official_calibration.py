#!/usr/bin/env python3
"""Run PM-KVQ official calibration/allocation stages for serving fake quant.

This wrapper keeps the official PM-KVQ repo unchanged.  It adds only runtime
compatibility shims for the local transformers/Qwen2 stack and writes artifacts
in the format consumed by the vLLM fake-quant hook.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmkvq-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-kind",
                        choices=[
                            "redpajama", "redpajama_stream",
                            "qasper_train"
                        ],
                        default="redpajama")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-config", default="arxiv")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--effective-len", type=int, default=8192)
    parser.add_argument("--memory-budget", type=float, default=113.0)
    parser.add_argument("--fbit-choices", default="4,2")
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--sink-tokens", type=int, default=1)
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--protected-bits", type=int, default=16)
    parser.add_argument("--rep-grid", type=int, default=20)
    parser.add_argument("--rep-batch-size", type=int, default=1)
    parser.add_argument("--rep-k-bits", type=int, default=4)
    parser.add_argument("--rep-v-bits", type=int, default=4)
    parser.add_argument("--stages",
                        default="sensitivity,budget,max_keys,rep_scales")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def patch_transformers_qwen2_compat() -> None:
    import transformers.modeling_utils as modeling_utils
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    def eager_attention_forward(module, query, key, value, attention_mask,
                                dropout=0.0, scaling=None, **kwargs):
        repeat_kv = getattr(qwen2, "repeat_kv", None)
        if repeat_kv is None:
            repeat_kv = getattr(llama, "repeat_kv")
        n_rep = getattr(module, "num_key_value_groups", 1)
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)
        scale = scaling
        if scale is None:
            scale = query.shape[-1] ** -0.5
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout and module.training:
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=dropout, training=True)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    if not hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        modeling_utils.ALL_ATTENTION_FUNCTIONS = {}
    modeling_utils.ALL_ATTENTION_FUNCTIONS.setdefault(
        "eager", eager_attention_forward)
    modeling_utils.ALL_ATTENTION_FUNCTIONS.setdefault(
        "sdpa", eager_attention_forward)
    if not hasattr(llama, "ALL_ATTENTION_FUNCTIONS"):
        llama.ALL_ATTENTION_FUNCTIONS = modeling_utils.ALL_ATTENTION_FUNCTIONS
    if not hasattr(llama, "eager_attention_forward"):
        llama.eager_attention_forward = eager_attention_forward
    if not hasattr(qwen2, "ALL_ATTENTION_FUNCTIONS"):
        qwen2.ALL_ATTENTION_FUNCTIONS = modeling_utils.ALL_ATTENTION_FUNCTIONS
    if not hasattr(qwen2, "eager_attention_forward"):
        qwen2.eager_attention_forward = eager_attention_forward


def pmkvq_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def infer_kv_hidden_size(model_path: Path) -> int:
    cfg = AutoConfig.from_pretrained(model_path)
    heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    hidden = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", hidden // cfg.num_attention_heads))
    return heads * head_dim


def iter_qasper_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for paper in data.values():
        parts = [paper.get("title", ""), paper.get("abstract", "")]
        for section in paper.get("full_text", []):
            parts.extend(section.get("paragraphs", []))
        text = "\n".join(part for part in parts if part)
        if text:
            yield text


def build_qasper_dataset(path: Path, tokenizer, n_samples: int,
                         seq_len: int) -> Dataset:
    all_tokens: list[int] = []
    for text in iter_qasper_texts(path):
        all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(all_tokens) >= n_samples * seq_len:
            break
    total = min(n_samples, len(all_tokens) // seq_len)
    if total <= 0:
        raise RuntimeError(f"No {seq_len}-token Qasper calibration samples")
    samples = [{
        "input_ids": all_tokens[i * seq_len:(i + 1) * seq_len]
    } for i in range(total)]
    return Dataset.from_list(samples)


def build_redpajama_stream_dataset(path: str, config: str, tokenizer,
                                   n_samples: int, seq_len: int) -> Dataset:
    from datasets import load_dataset

    stream = load_dataset(
        path,
        config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    all_tokens: list[int] = []
    samples: list[dict[str, list[int]]] = []
    for example in stream:
        text = example.get("text", "")
        if not text:
            continue
        all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(all_tokens) >= seq_len and len(samples) < n_samples:
            samples.append({"input_ids": all_tokens[:seq_len]})
            del all_tokens[:seq_len]
        if len(samples) >= n_samples:
            break
    if len(samples) < n_samples:
        raise RuntimeError(
            f"Built only {len(samples)} RedPajama samples; need {n_samples}")
    return Dataset.from_list(samples)


def build_dataset(args: argparse.Namespace, tokenizer) -> Dataset:
    if args.dataset_kind == "redpajama":
        from pm_kvq.datasets.calib_dataset import get_calib_redpajama

        return get_calib_redpajama(args.dataset_path, args.n_samples,
                                   args.seq_len, tokenizer)
    if args.dataset_kind == "redpajama_stream":
        return build_redpajama_stream_dataset(args.dataset_path,
                                              args.dataset_config, tokenizer,
                                              args.n_samples, args.seq_len)
    return build_qasper_dataset(Path(args.dataset_path), tokenizer,
                                args.n_samples, args.seq_len)


def load_model(args: argparse.Namespace, single_device: bool = False):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    if single_device:
        kwargs["device_map"] = None
    else:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if single_device:
        model.to("cuda:0")
    model.eval()
    model.config._attn_implementation = "eager"
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = layer.self_attn
        if not hasattr(attn, "scaling"):
            attn.scaling = attn.head_dim ** -0.5
    return model


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_three_attention_outputs(fn):
    def wrapped(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1], None
        return out

    return wrapped


def patch_pmkvq_qwen2_forward_tables() -> None:
    from pm_kvq.utils.modeling_utils import ModelType

    try:
        import pm_kvq.quantization.methods.pm_kvq.allocation.sensitivity as sens

        fn = sens.GRADKV_FORWARD.get(ModelType.QWEN2)
        if fn is not None and not getattr(fn, "_atc_three_outputs", False):
            wrapped = ensure_three_attention_outputs(fn)
            wrapped._atc_three_outputs = True
            sens.GRADKV_FORWARD[ModelType.QWEN2] = wrapped
    except Exception:
        pass

    try:
        import pm_kvq.quantization.methods.pre_attn_rtn as rtn

        fn = rtn.RTN_ATTENTION_FORWARD.get(ModelType.QWEN2)
        if fn is not None and not getattr(fn, "_atc_three_outputs", False):
            wrapped = ensure_three_attention_outputs(fn)
            wrapped._atc_three_outputs = True
            rtn.RTN_ATTENTION_FORWARD[ModelType.QWEN2] = wrapped
    except Exception:
        pass


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "sensitivity": out_dir / "sensitivity.pt",
        "budget": out_dir / "budget_fbit_4_2.pt",
        "max_keys": out_dir / "max_keys.pt",
        "rep_scales": out_dir / "rep_scales_k4v4.pt",
        "calib_dataset": out_dir / "calib_dataset_tokenized.jsonl",
        "metadata": out_dir / "metadata.json",
    }


def save_budget(args: argparse.Namespace, paths: dict[str, Path],
                hidden_size: int, source: str) -> dict[str, object]:
    from pm_kvq.quantization.methods.pm_kvq.allocation.allocation import (
        allocate_memory_budget,
    )

    sensitivity = torch.load(paths["sensitivity"], map_location="cpu")
    fbit_choices = [int(x) for x in args.fbit_choices.split(",") if x.strip()]
    layer_bits, official_budgets = allocate_memory_budget(
        fbit_choices=fbit_choices,
        k_sensitivity=sensitivity["k_sensitivity"],
        v_sensitivity=sensitivity["v_sensitivity"],
        memory_budget=args.memory_budget,
        hidden_size=hidden_size,
        max_len=args.max_len,
        save_path=None,
    )
    layer_bits = [int(x) for x in layer_bits]
    official_budgets = [float(x) for x in official_budgets]
    protected = min(args.max_len, max(0, args.sink_tokens) +
                    max(0, args.window_tokens))
    body = max(0, args.max_len - protected)
    elements_per_token = hidden_size * 2
    runtime_budgets = [
        elements_per_token *
        (protected * args.protected_bits + body * int(bits)) /
        (8 * 1024 * 1024)
        for bits in layer_bits
    ]
    torch.save(runtime_budgets, paths["budget"])
    return {
        "source": source,
        "budget_path": str(paths["budget"]),
        "memory_budget_mb": args.memory_budget,
        "fbit_choices": fbit_choices,
        "hidden_size": hidden_size,
        "elements_per_token": elements_per_token,
        "max_len": args.max_len,
        "sink_tokens": args.sink_tokens,
        "window_tokens": args.window_tokens,
        "protected_bits": args.protected_bits,
        "official_layer_budgets_mb_without_protected_overhead":
        official_budgets,
        "runtime_budgets_mb_with_protected_overhead": runtime_budgets,
        "layer_bits": layer_bits,
        "avg_layer_bulk_bits": sum(layer_bits) / max(1, len(layer_bits)),
    }


def run_sensitivity(args: argparse.Namespace, dataset, paths: dict[str, Path]):
    patch_pmkvq_qwen2_forward_tables()
    from pm_kvq.quantization.methods.pm_kvq.allocation import sensitivity

    model = load_model(args)
    try:
        sensitivity.get_kv_sensitivity(model, dataset, args.effective_len,
                                       paths["sensitivity"])
    finally:
        unload_model(model)


def run_max_keys(args: argparse.Namespace, dataset, paths: dict[str, Path]):
    from pm_kvq.quantization.methods.pm_kvq.smoothattention.apply_smoothattention import (
        get_max_keys,
    )

    model = load_model(args)
    try:
        try:
            get_max_keys(model, dataset, args.effective_len, paths["max_keys"])
        except AttributeError as exc:
            if "key_cache" not in str(exc):
                raise
            get_max_keys_compat(model, dataset, args.effective_len,
                                paths["max_keys"])
    finally:
        unload_model(model)


@torch.no_grad()
def get_max_keys_compat(model, calib_dataset, effective_len: int | None,
                        save_path: Path):
    max_keys = []
    if effective_len is not None:
        seq_len = len(calib_dataset[0]["input_ids"])
        scale = effective_len // seq_len
        position_ids = torch.arange(
            0, effective_len, scale, device=model.device).reshape(1, -1)
    else:
        position_ids = None
    for example in calib_dataset:
        outputs = model(
            torch.tensor([example["input_ids"]], device=model.device),
            position_ids=position_ids,
            use_cache=True,
        )
        past = outputs.past_key_values
        key_cache = getattr(past, "key_cache", None)
        if key_cache is None:
            key_cache = [kv[0] for kv in past]
        for i, key in enumerate(key_cache):
            key_max = torch.amax(
                key.abs().reshape(*key.shape[:-2], -1, key.shape[-1] // 2),
                dim=-2,
                keepdim=True,
            )
            if len(max_keys) == i:
                max_keys.append(key_max)
            else:
                max_keys[i] = torch.maximum(key_max, max_keys[i])
    torch.save([max_key.cpu() for max_key in max_keys], save_path)
    return max_keys


def run_rep_scales(args: argparse.Namespace, dataset, paths: dict[str, Path]):
    patch_pmkvq_qwen2_forward_tables()
    from pm_kvq.quantization.methods.pm_kvq.smoothattention import searching_scales

    max_keys = torch.load(paths["max_keys"], map_location="cpu")
    k_config = {
        "n_bits": args.rep_k_bits,
        "granularity": "per_group",
        "symmetric": False,
        "group_size": 128,
        "round_zeros": False,
    }
    v_config = {
        "n_bits": args.rep_v_bits,
        "granularity": "per_group",
        "symmetric": False,
        "group_size": 128,
        "round_zeros": False,
    }
    model = load_model(args, single_device=True)
    try:
        searching_scales.search_rep_scales(
            model,
            k_config=k_config,
            v_config=v_config,
            dataset=dataset,
            max_keys=max_keys,
            grid=args.rep_grid,
            batch_size=args.rep_batch_size,
            effective_len=args.effective_len,
            save_path=paths["rep_scales"],
        )
    finally:
        unload_model(model)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(args.out_dir)
    stages = {x.strip() for x in args.stages.split(",") if x.strip()}

    patch_transformers_qwen2_compat()
    sys.path.insert(0, str(args.pmkvq_repo))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if paths["calib_dataset"].exists() and not args.force:
        from datasets import load_dataset

        dataset = load_dataset(
            "json",
            data_files=str(paths["calib_dataset"]),
            split="train",
        )
    else:
        dataset = build_dataset(args, tokenizer)
        dataset.to_json(paths["calib_dataset"])
    hidden_size = args.hidden_size or infer_kv_hidden_size(args.model_path)

    metadata: dict[str, object] = {
        "source": (
            "pmkvq_official_redpajama_calibration" if
            args.dataset_kind == "redpajama" else
            "pmkvq_redpajama_arxiv_stream_calibration" if
            args.dataset_kind == "redpajama_stream" else
            "pmkvq_qasper_train_surrogate_diagnostics"),
        "pmkvq_repo": str(args.pmkvq_repo),
        "pmkvq_commit": pmkvq_commit(args.pmkvq_repo),
        "model_path": str(args.model_path),
        "dataset_kind": args.dataset_kind,
        "dataset_path": args.dataset_path,
        "dataset_config": args.dataset_config,
        "n_samples_requested": args.n_samples,
        "n_samples_built": len(dataset),
        "seq_len": args.seq_len,
        "effective_len": args.effective_len,
        "stages": sorted(stages),
        "artifacts": {name: str(path) for name, path in paths.items()
                      if name != "metadata"},
        "status": "started",
        "notes": [],
    }

    if "sensitivity" in stages:
        if paths["sensitivity"].exists() and not args.force:
            metadata["notes"].append("reused existing sensitivity.pt")
        else:
            run_sensitivity(args, dataset, paths)

    if "budget" in stages:
        if not paths["sensitivity"].exists():
            raise FileNotFoundError(paths["sensitivity"])
        if paths["budget"].exists() and not args.force:
            metadata["notes"].append("reused existing budget artifact")
        else:
            metadata["budget"] = save_budget(args, paths, hidden_size,
                                             str(metadata["source"]))

    if "max_keys" in stages:
        if paths["max_keys"].exists() and not args.force:
            metadata["notes"].append("reused existing max_keys.pt")
        else:
            run_max_keys(args, dataset, paths)

    if "rep_scales" in stages:
        if not paths["max_keys"].exists():
            raise FileNotFoundError(paths["max_keys"])
        if paths["rep_scales"].exists() and not args.force:
            metadata["notes"].append("reused existing rep_scales.pt")
        else:
            run_rep_scales(args, dataset, paths)
        rep_scales = torch.load(paths["rep_scales"], map_location="cpu")
        metadata["rep_scales"] = {
            "path": str(paths["rep_scales"]),
            "k_bits": args.rep_k_bits,
            "v_bits": args.rep_v_bits,
            "grid": args.rep_grid,
            "batch_size": args.rep_batch_size,
            "num_layers": len(rep_scales),
            "scale_shapes": [list(x.shape) for x in rep_scales],
        }

    metadata["status"] = "complete"
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
