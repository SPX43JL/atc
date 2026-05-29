#!/usr/bin/env python3
"""Write a reproducibility manifest for PM-KVQ calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "/root/atc_vllm_sched/artifacts/pmkvq/"
            "redpajama_arxiv_stream_n512_l2048_eff8192"),
    )
    parser.add_argument("--output-name", default="calibration_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    metadata_path = artifact_dir / "metadata.json"
    dataset_path = artifact_dir / "calib_dataset_tokenized.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    lengths: list[int] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            lengths.append(len(record.get("input_ids", [])))

    artifact_hashes: dict[str, str] = {}
    for name in [
        "calib_dataset_tokenized.jsonl",
        "metadata.json",
        "sensitivity.pt",
        "budget_fbit_4_2.pt",
        "budget_fbit_4_2_mem90.pt",
        "max_keys.pt",
        "rep_scales_k4v4.pt",
    ]:
        path = artifact_dir / name
        if path.exists():
            artifact_hashes[name] = sha256_file(path)

    manifest = {
        "source": metadata.get("source"),
        "pmkvq_commit": metadata.get("pmkvq_commit"),
        "model_path": metadata.get("model_path"),
        "dataset_kind": metadata.get("dataset_kind"),
        "dataset_path": metadata.get("dataset_path"),
        "dataset_config": metadata.get("dataset_config"),
        "streaming": metadata.get("dataset_kind") == "redpajama_stream",
        "shuffle": False,
        "seed": None,
        "seed_note": "not used; streaming subset was consumed in deterministic order",
        "selection_policy": (
            "first_512_contiguous_2048_token_blocks_from_redpajama_arxiv_stream"
        ),
        "sample_block_indices": list(range(len(lengths))),
        "n_samples": len(lengths),
        "seq_len": metadata.get("seq_len"),
        "effective_len": metadata.get("effective_len"),
        "all_tokenized_lengths": sorted(set(lengths)),
        "calib_dataset_jsonl": str(dataset_path),
        "artifact_hashes": artifact_hashes,
    }
    out_path = artifact_dir / args.output_name
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
