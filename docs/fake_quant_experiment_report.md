# vLLM KV Fake Quant Serving Experiment Report

This report is for Python-level fake quantization only. It does not claim real KV cache memory saving, packed INT storage, CUDA acceleration, or kernel speedup.

## Formal Configuration

- Dataset: `THUDM/LongBench`, config `qasper`, split `test`, first 200 official-order examples.
- Prompt/max generation: official LongBench config from KIVI/KVTuner repos; Qasper `max_gen=128`.
- Context truncation: token-based middle truncation; `max_input_tokens=7500` unless documented otherwise in each run.
- Serving concurrency/workload: framework serving extension, not specified in the papers.

## Formal Results

| workload | concurrency | method | ok/total | LongBench F1 | EM | output trunc | prompt trunc | avg latency(s) | p95 latency(s) | queue wait(s) | throughput(req/s) | avg_k_bits | avg_v_bits | avg_kv_bits | precision_distribution | PM cache-wide coverage |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| burst | 8 | pmkvq_cachewide | 200/200 | 44.08 | 0.1900 | 0.0000 | 0.1050 | 5.9163 | 17.4091 | 71.2794 | 1.3195 | 6.27 | 6.27 | 6.27 | 4:0.57, 8:0.36, 16:0.07 | 1.0000 |

Result directory:

`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260522_pmkvq_defer_current_mem132_fast_c8/`

PM-KVQ artifact inputs:

- Calibration: `/root/atc_vllm_sched/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192/`.
- Data source: `togethercomputer/RedPajama-Data-1T`, `arxiv` streaming subset, deterministic first 512 token blocks.
- Calibration manifest: `n_samples=512`, `seq_len=2048`, `effective_len=8192`, JSONL sha256 `a97b062b5e83d54bce99b3cc12771cee856a87d64c459e338c0992a77b267a68`.
- Budget: `budget_fbit_8_4_mem132.pt`, official allocation over `fbit_choices=[8,4]`; layer bulk average `4.71` bit.
- Rep-scale: `rep_scales_k4v4.pt`, strict shape checked for Qwen2.5 28 layers x 4 KV heads x 64 half-head channels.

The reported `avg_kv_bits=6.27` is residual/protected-token effective precision from serving traces. It is higher than the layer bulk average because PM-KVQ keeps sink/window tokens at 16-bit.

## PM-KVQ Diagnostics

- RedPajama official 4/2 and all-4-ish budgets remain unstable on this Qwen2.5/Qasper setup: HF official probe and vLLM cache-wide diagnostics both show severe F1 drop and/or truncation.
- Official `8,4` allocation `mem128` is still too aggressive; `mem132` is the lowest stable point found so far.
- The earlier slow 200-sample run was caused by Python-level cache-wide rewrite overhead, not dataset/evaluator settings. The implementation now uses batched tensor gather/scatter and formula-based bit counting; Qasper-200 c8 runtime dropped from roughly 35 minutes to `151.57s`.
- A transient single-sample 502 should be repaired with `--indices <idx>` or `EVAL_INDICES=<idx>` instead of rerunning the whole table. Full rerun is only needed when latency/throughput changes.

## Pressure Trace

PMKVQ and MixKVQ write precision decisions to `logs/fake_quant/<method>_kv_fake_quant.jsonl` and router pressure events to `logs/fake_quant/<method>_router_trace.jsonl`.

## Known Limits

- All methods are mapped into fake quantize/dequantize before vLLM writes normal FP16/BF16 KV cache.
- Current request-level pressure is approximated through a router state file, so per-layer traces are aligned with serving pressure but not a perfect per-request scheduler integration.
- KVQuant now has a Qwen2-specific pre-RoPE K fake-quant hook; runs without a complete NUQ artifact remain diagnostics only.
- PM-KVQ `pmkvq_cachewide` rewrites normal FP16/BF16 vLLM cache slots in place to emulate progressive shrinking. It does not store packed low-bit KV and does not claim real memory saving or acceleration.

## 2026-05-26 Multitask Formal Extension

Result directory:

`/root/atc_vllm_sched/results/fake_quant/formal/multitask_longbench_math_200_chat_c8_20260526/`

Manifest metadata:

`/root/atc_vllm_sched/data/eval_manifests/20260526_multitask_200/manifest_summary.json`

The run completed `8 datasets x 7 methods x 200 examples = 11200` serving requests, burst c8, chat endpoint, temperature 0.0. Full tables are in:

- `summary.md`
- `summary.csv`
- `summary.json`

Key Qasper rows:

| method | score | failed | trunc | total elapsed(s) | avg_kv_bits | precision_distribution |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 44.43 | 0 | 0/200 | 42.3 | BF16 | BF16/FP16 |
| kivi2 | 41.74 | 0 | 1/200 | 55.8 | 2.40 | 2:0.97, 16:0.03 |
| kivi4 | 44.60 | 0 | 0/200 | 56.9 | 4.34 | 4:0.97, 16:0.03 |
| kvtuner | 44.61 | 0 | 0/200 | 55.8 | 4.09 | 2:0.57, 4:0.17, 8:0.23, 16:0.03 |
| kvquant | 43.39 | 0 | 0/200 | 101.0 | 3.00 | 3:1.00 |
| pmkvq_cachewide_mem132 | 44.13 | 0 | 0/200 | 142.2 | 6.28 | 4:0.57, 8:0.36, 16:0.07 |
| mixkvq_c2_7 | 44.00 | 0 | 0/200 | 65.9 | 2.73 | 2:0.94, 4:0.01, 16:0.05 |

Observed long-generation behavior:

- KVQuant is the slowest on MATH500: score `61.50`, total elapsed `2591.8s`, avg latency `102.30s`, trunc `28/200`.
- PM-KVQ cache-wide is also slow on MATH500: score `64.50`, total elapsed `1947.8s`, avg latency `76.87s`, trunc `29/200`.
- MixKVQ C2.7 MATH500 score is `65.00`, total elapsed `1199.2s`, trunc `23/200`.
- Baseline MATH500 score is `65.50`, total elapsed `360.6s`, trunc `29/200`.

These latency numbers are fake-quant Python overhead and cache rewrite overhead; they are not kernel-speed results. The complete summary should be used for cross-task comparison.

## 2026-05-27 Cache Semantics Repair

No GPU was available for a formal rerun at the time of this repair. The following changes are code-level/paper-alignment fixes, so the 2026-05-26 multitask rows for KIVI-2, KIVI-4, KVTuner, KVQuant, and MixKVQ should be treated as pre-repair diagnostics until rerun.

- KIVI/KVTuner-KIVI mode now distinguish K and V residual semantics from the KIVI paper/repo: K keeps only the unflushed `seq_len % R` residual group in FP16; V keeps the latest `R` tokens in FP16.
- KVTuner default is now the official Qwen2.5-7B-Instruct per-token-asym C4.0 preset with `group_size=-1` and `residual_length=0`; KVTuner repo `vanilla_quantizer.py` resolves `q_group_size=-1` to the full last dimension. KIVI-mode presets remain supported with `group_size=32` and `residual_length=32` for explicit diagnostics.
- KVQuant attention sink is now global first token only (`first_tokens_fp16=1`), matching local `kvquant.pdf` Section 3.5.
- MixKVQ now uses `sink=32` plus lazy residual buffer modulo `R=128`, matching local `MixKVQ.pdf` Appendix D/E, instead of a permanent sliding last-128 FP16 window.

## 2026-05-27 KVQuant/MixKVQ Artifact Repair

No GPU was available for new calibration or formal evaluation. This pass added the code needed for the next GPU run:

- KVQuant: added `scripts/run_kvquant_nuq_artifact.py`, official `quant_fn_nuq_recon` adapter, Qwen2 pre-RoPE K fake-quant hook, strict artifact metadata, and cache-write K skip when pre-RoPE was applied. Formal should require `kvquant_mode=prerope_nuq`.
- MixKVQ: added `scripts/search_mixkvq_thresholds.py` and strict `ATC_MIXKVQ_THRESHOLDS_PATH` runtime. Formal should require `thresholds_source=artifact`; old no-artifact C2.7 ratio assignment is diagnostics only.
- Static validation passed: `py_compile`, `bash -n`, `scripts/test_fake_quant_methods.py`, KVQuant driver `--help`, and MixKVQ threshold-search dry-run.

Static verification passed on atc: `python -m py_compile` for the fake-quant core/tests/summary scripts, `bash -n` for the formal pipelines, and `PYTHONPATH=/root/atc_vllm_sched/vllm_src .venv_dev/bin/python scripts/test_fake_quant_methods.py`.

## 2026-05-28 Six-Variant Preparation

This pass prepares methods and calibration artifacts only; it does not rerun the full 8-dataset formal table.

- Default formal variants are now `baseline,kivi2,kivi4,kvtuner_pertoken_c4_00,kvtuner_kivi_c3_92,kvquant,pmkvq_cachewide_mem132`.
- `kvtuner_pertoken_c4_00` uses the official Qwen2.5 per-token-asym C4.00 preset with `group_size=-1` and `residual_length=0`.
- `kvtuner_kivi_c3_92` uses the official Qwen2.5 KIVI-mode C3.92 preset with `group_size=32` and `residual_length=32`.
- MixKVQ is removed from the default formal matrix because it has no official repo; its code path remains diagnostics-only.
- KVQuant nuq3-1% strict formal is now prepared with Wikitext-2 16x2048 artifact `/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048_nuq/nuq_artifact.pt` (`sha256=636ff8f76e063009d328256ee74b808bf92b0eef46802f1982c4ae398a0e0d94`), pre-RoPE K hook, global first-token FP16, and trace `kvquant_mode=prerope_nuq`.
