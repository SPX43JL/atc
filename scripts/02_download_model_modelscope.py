#!/usr/bin/env python3
import argparse
import inspect
import os
from pathlib import Path
from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-7B-Instruct from ModelScope with resume support.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target-dir", default="/root/atc_vllm_sched/models/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", "/root/atc_vllm_sched/.modelscope_cache")

    print(f"Downloading {args.model_id} to {target}")
    kwargs = {"local_dir": str(target)}
    params = inspect.signature(snapshot_download).parameters
    if "resume_download" in params:
        kwargs["resume_download"] = True
    if "local_dir_use_symlinks" in params:
        kwargs["local_dir_use_symlinks"] = False
    snapshot_download(args.model_id, **kwargs)

    checks = {
        "config.json": list(target.glob("config.json")),
        "tokenizer": list(target.glob("tokenizer*")),
        "safetensors": list(target.glob("*.safetensors")),
    }
    missing = [name for name, files in checks.items() if not files]
    if missing:
        raise SystemExit(f"Download finished but missing expected files: {missing}. Existing files: {[p.name for p in target.iterdir()]}")

    print("Model download verified:")
    for name, files in checks.items():
        print(f"  {name}: {len(files)} file(s)")
    print(f"  path: {target}")


if __name__ == "__main__":
    main()
