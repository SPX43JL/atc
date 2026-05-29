#!/usr/bin/env python3
"""Build a PM-KVQ layer budget artifact from an official sensitivity file.

The PM-KVQ official allocation code chooses one bulk KV bit-width per layer.
The serving fake-quant adapter stores the corresponding per-layer memory
budgets so the runtime can reproduce that progressive budget decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmkvq-repo", type=Path, required=True)
    parser.add_argument("--sensitivity-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--memory-budget", type=float, required=True)
    parser.add_argument("--fbit-choices", default="4,2")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=7500)
    parser.add_argument("--sink-tokens", type=int, default=1)
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--protected-bits", type=int, default=16)
    parser.add_argument("--source", default="pmkvq_official_allocation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pmkvq_repo))
    from pm_kvq.quantization.methods.pm_kvq.allocation.allocation import (  # noqa: PLC0415
        allocate_memory_budget,
    )

    sensitivity = torch.load(args.sensitivity_path, map_location="cpu")
    fbit_choices = [int(x) for x in args.fbit_choices.split(",") if x.strip()]
    layer_bits, official_budgets = allocate_memory_budget(
        fbit_choices=fbit_choices,
        k_sensitivity=sensitivity["k_sensitivity"],
        v_sensitivity=sensitivity["v_sensitivity"],
        memory_budget=args.memory_budget,
        hidden_size=args.hidden_size,
        max_len=args.max_len,
        save_path=None,
    )
    layer_bits = [int(x) for x in layer_bits]
    official_budgets = [float(x) for x in official_budgets]

    protected = min(args.max_len, max(0, args.sink_tokens) +
                    max(0, args.window_tokens))
    body = max(0, args.max_len - protected)
    elements_per_token = args.hidden_size * 2
    protected_budgets = [
        elements_per_token *
        (protected * args.protected_bits + body * int(bits)) /
        (8 * 1024 * 1024)
        for bits in layer_bits
    ]

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(protected_budgets, args.save_path)
    metadata = {
        "source": args.source,
        "pmkvq_repo": str(args.pmkvq_repo),
        "sensitivity_path": str(args.sensitivity_path),
        "memory_budget_mb": args.memory_budget,
        "fbit_choices": fbit_choices,
        "hidden_size": args.hidden_size,
        "elements_per_token": elements_per_token,
        "max_len": args.max_len,
        "sink_tokens": args.sink_tokens,
        "window_tokens": args.window_tokens,
        "protected_bits": args.protected_bits,
        "official_layer_budgets_mb_without_protected_overhead":
        official_budgets,
        "layer_bits": layer_bits,
        "avg_layer_bulk_bits": sum(layer_bits) / max(1, len(layer_bits)),
        "runtime_budgets_mb_with_protected_overhead": protected_budgets,
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
