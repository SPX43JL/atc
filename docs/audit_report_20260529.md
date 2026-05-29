# ATC Fake Quant Serving Audit Report - 2026-05-29

## Verdict

Current formal results are usable with caution.

There is no fatal issue for the stated project goal: vLLM serving-side Python
fake quant / fake-quant approximation. The results should not be described as
packed low-bit KV cache, real KV-cache memory saving, kernel acceleration, or
production low-bit serving.

GPU was unavailable during this audit, so the checks were static, result-file
based, configuration based, and reproducibility oriented.

## Result Integrity

Checked source files:

- Combined table: `combined_total_table_longbench_math_gsm_20260529.{md,csv,json}`.
- LongBench6 summary: `summary_merged_with_pm.json`.
- MATH500/GSM8K summary: `summary_merged_with_best_baseline_pm.json`.
- Current docs, scripts, fake-quant runtime, and Qwen2 pre-RoPE hook files.

Findings:

- Combined table rows match the referenced summary/result rows for score,
  failed count, truncation, latency, throughput, endpoint, temperature,
  concurrency, workload, average tokens, and max generation length.
- Endpoint is consistently chat, temperature is `0.0`, workload is burst, and
  max concurrency is `8`.
- LongBench prompt/max generation follows KIVI/LongBench-style config:
  Qasper 128, NarrativeQA 128, HotpotQA 32, passage retrieval/count 32, QMSum
  512. Token middle truncation uses `max_input_tokens=7500`.
- MATH500 and GSM8K use identical prompts, generation lengths, answer
  extraction, and exact-match logic across baseline and fake-quant methods.
- QMSum uses an in-project LCS ROUGE-L implementation rather than the external
  official `rouge` package. This is internally comparable but should be labeled
  as an evaluator approximation if making official LongBench claims.

Important reporting notes:

- MATH/GSM baseline uses the better score from old/rerun baseline summaries:
  MATH500 uses the rerun score `66.0`; GSM8K uses the older score `85.5`.
- KVQuant bit accounting now uses effective trace accounting: NUQ 3-bit bulk
  plus first-token FP16 plus 1% sparse outliers charged as FP16-equivalent
  storage. This yields about LongBench `3.13`, MATH500 `3.15`, GSM8K `3.14`
  average KV bits and precision distribution around `3:0.99,16:0.01`.
- Keep both nominal/bulk bit and effective bit columns in paper/report tables
  when possible.
- Ten LongBench KIVI-4 / KVTuner-KIVI bit rows are backfilled from pre-retry
  trace backups. Scores/latencies come from the current merged result. This is
  acceptable for a cautious report because the backed-up score rows match the
  current rows, but GPU rerun is the cleanest future fix.

## Method Alignment

KIVI:

- Reference repo: `jy-yuan/KIVI`, commit `876b4d2`.
- Current mapping: K/V 2-bit and 4-bit variants, group size `G=32`, residual
  length `R=128`, K per-channel and V per-token fake quant.
- Serving status: fake quant only; no packed cache or KIVI CUDA kernels.

KVTuner:

- Reference repo: `cmd2001/KVTuner`, commit `96dd05e`.
- Current mapping: Qwen2.5-7B-Instruct official preset files for per-token
  C4.00 and KIVI-mode C3.92.
- KVTuner-KIVI uses KVTuner preset residual/group settings, not pure KIVI.
- Serving status: uses preset layer-wise K/V bits; no new search/calibration
  is included in this snapshot.

KVQuant:

- Reference repo: `SqueezeAILab/KVQuant`, commit `57a2383`.
- Current mapping: NUQ3, Wikitext-2 16 x 2048 calibration artifact metadata,
  global first-token FP16, 1% sparse outlier accounting, Qwen2 pre-RoPE K hook.
- The Qwen2 hook applies fake quant after QKV split and before rotary
  embedding. Cache write avoids double-quantizing K when pre-RoPE K is marked.
- Serving status: fake quant only; no dense/sparse packed storage or kernels.

PM-KVQ:

- Reference repo: `thu-nics/PM-KVQ`, commit `c8a2bff`.
- Current mapping: RedPajama arXiv 512 x 2048 calibration metadata,
  `effective_len=8192`, rep scales, budget `mem132`, cache-wide progressive
  rewrite approximation.
- MATH/GSM rows show `16:1.00` because the mem132 budget does not trigger
  low-bit rewrite on those shorter effective cache lengths. This is expected
  and should not be reported as a low-bit math/GSM saving.
- Serving status: Python in-place rewrite of normal FP16/BF16 vLLM cache slots;
  no packed cache or real memory reduction.

## Anomaly Notes

- KIVI-2 higher than KIVI-4 on a few rows is small-slice/tie-level variation
  plus generation/truncation sensitivity. It is not evidence that the config is
  wrong.
- KVQuant occasionally above baseline on NarrativeQA/MATH500 is also
  tie-level variation/quantization perturbation on 200-example subsets.
- PM-KVQ `16:1.00` on MATH/GSM is a budget-trigger outcome, not a trace bug.
- `bit_source=pre_retry_trace_backup` rows should be footnoted and rerun later
  when GPU is stable.

## Suggested Future Reruns

Minimum GPU rerun list:

- LongBench rows whose bit stats are `pre_retry_trace_backup`:
  KIVI-4 and KVTuner-KIVI on Qasper/HotpotQA/Passage Count/QMSum, plus the
  other backed-up rows listed in the combined table JSON.
- Optional: recompute QMSum with the external LongBench `rouge` dependency if
  the report must claim exact official LongBench evaluator parity.

## Snapshot Safety Checks

Before committing/pushing this public snapshot:

```bash
git status --short
git diff --stat
find . -type f -size +20M -print
```

Then run a credential scanner for common token/key/password/authorization
assignment patterns and common provider token value prefixes. The current
prepared snapshot was checked with equivalent PowerShell commands before
staging.
