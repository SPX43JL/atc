# ATC vLLM KV Fake Quant Serving Snapshot

This repository is a clean, public snapshot of the useful ATC project files for
reproducing and auditing the current formal KV fake-quant serving experiments.

The scope is intentionally narrow: this project evaluates Python-level
fake-quant / quantize-dequantize approximations inside a vLLM serving path. It
does not implement packed low-bit KV cache storage, real KV-cache memory
reduction, CUDA/Triton kernel acceleration, or real low-bit serving kernels.

## What Is Included

- Current ATC fake-quant code and vLLM integration points.
- Serving, benchmark, manifest, artifact-building, and summarization scripts.
- Latest useful KIVI, KVTuner, KVQuant, and PM-KVQ method configs/metadata.
- Method-specific fake-quant implementations split under
  `vllm_src/vllm/attention/ops/atc_kv_fake_quant/methods/`, with `core.py`
  kept as the unified vLLM hook entry point.
- Minimal official reference scripts for the four methods under
  `methods/*/reference_scripts/`, so reviewers can inspect the paper/repo
  mechanism that the ATC fake-quant path approximates.
- Formal LongBench6 + MATH500 + GSM8K summary tables.
- Documentation and the 2026-05-29 audit report.

Large or sensitive inputs are not included: model weights, dataset caches,
raw JSONL manifests, per-example predictions, large logs, `.pt` artifacts,
virtualenvs, backups, and credentials.

## Repository Layout

```text
docs/                 Reports and audit notes.
methods/              Method-specific configs, metadata, and alignment notes.
manifests/            Manifest summary only; raw JSONL slices are regenerated.
results/formal/       Current formal summary tables and combined total table.
router/               Simple OpenAI-compatible round-robin router.
scripts/              Current useful serving/eval/artifact/reproduction scripts.
vllm_src/             Modified vLLM files needed by the ATC fake-quant path.
```

## Formal Experiment Scope

- Model: `Qwen2.5-7B-Instruct`.
- Endpoint: OpenAI-compatible chat endpoint.
- Decoding: `temperature=0.0`.
- Workload: burst.
- Max concurrency: `8`.
- Samples: first 200 examples per dataset.
- LongBench tasks: `qasper`, `narrativeqa`, `hotpotqa`,
  `passage_retrieval_en`, `passage_count`, `qmsum`.
- Math tasks: `HuggingFaceH4/MATH-500` test first 200 and
  `openai/gsm8k` main/test first 200.
- Methods in the current formal table: `baseline`, `KIVI-2`, `KIVI-4`,
  `KVTuner per-token C4.00`, `KVTuner-KIVI C3.92`, `KVQuant NUQ3`,
  `PM-KVQ cachewide mem132`.

MixKVQ is not part of the current formal main table because there is no
official reference repo and the previous runs are diagnostic/exploratory.

## Current Formal Results

Use this table first:

- `results/formal/combined_longbench_math_gsm_20260529/combined_total_table_longbench_math_gsm_20260529.md`
- Same table as CSV/JSON in the same directory.

Supporting summaries:

- `results/formal/longbench6_200_chat_c8_repaired_20260528/summary_merged_with_pm.{md,csv,json}`
- `results/formal/math500_gsm8k_200_chat_c8_repaired_20260528/summary_merged_with_best_baseline_pm.{md,csv,json}`

Audit status: the current table is usable with caution. The table values were
checked against source summary files, but reports should explicitly state the
fake-quant-only scope, KVQuant effective-bit accounting, MATH/GSM baseline
selection policy, QMSum evaluator approximation, and the few LongBench bit
statistics backfilled from pre-retry trace backups.

## Reproduction Notes

This snapshot is not a complete vLLM source tree or an environment image. To
rerun the experiments, use a full ATC/vLLM serving environment and overlay the
files in `vllm_src/`, `scripts/`, and `router/`.

Expected server-side ingredients:

- vLLM 0.6.6-style source tree with the included modified files applied.
- Qwen2.5-7B-Instruct model weights.
- Hugging Face datasets available for LongBench, MATH-500, GSM8K, Wikitext-2,
  and RedPajama arXiv calibration as needed.
- Official method repos checked out at the commits listed in `methods/*/repo_meta.json`.
- KVQuant and PM-KVQ binary `.pt` artifacts regenerated or restored locally;
  only their public metadata is committed here.

Typical flow on the original ATC server layout:

```bash
cd /root/atc_vllm_sched
source .venv_dev/bin/activate

python scripts/prepare_multitask_eval_manifests.py
bash scripts/run_formal_multitask_pipeline.sh
python scripts/summarize_multitask_results.py --help
```

The formal pipeline starts vLLM workers, starts the router, runs smoke/formal
requests, and writes result summaries. GPU availability is required for reruns.

Method-specific reproduction helpers:

- KIVI/KVTuner: current variants are configured in
  `scripts/run_formal_multitask_pipeline.sh` and implemented in
  `vllm_src/vllm/attention/ops/atc_kv_fake_quant/methods/`.
- KVQuant: use `scripts/run_kvquant_nuq_artifact.py` for the current NUQ
  artifact path; `scripts/run_kvquant_official_calibration.py` is kept as the
  useful Fisher/official-wrapper reference used during alignment.
- PM-KVQ: use `scripts/run_pmkvq_official_calibration.py`,
  `scripts/build_pmkvq_budget_artifact.py`, and
  `scripts/write_pmkvq_calibration_manifest.py` for calibration/budget
  reproduction metadata.
- Official method repo excerpts are in `methods/*/reference_scripts/`.

## Public-Snapshot Safety

The repository intentionally excludes:

- `models/`, `logs/`, `data/`, `downloads/`, `backups/`, `.venv*`,
  `.modelscope_cache/`, full `references/`, and large `artifacts/`.
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.so`, `*.whl`,
  `*.jsonl`, `*.log`, `*.pid`, `*.pyc`, `*.orig`, `.env*`.
- Per-example result JSON files containing prompts, answers, or model outputs.

Before committing this snapshot, run the sensitive-string and large-file checks
described in `docs/audit_report_20260529.md`.
