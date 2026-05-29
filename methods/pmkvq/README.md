# PM-KVQ Current Snapshot

Current ATC use:

- Formal method: `PM-KVQ cachewide mem132`.
- Calibration metadata: RedPajama arXiv stream, 512 samples, 2048 tokens,
  `effective_len=8192`.
- Budget: `budget_fbit_8_4_mem132`.
- Serving mapping: Python cache-wide rewrite of normal FP16/BF16 vLLM KV cache
  slots to approximate progressive mixed-precision behavior.

Included files:

- `artifact_metadata/metadata.json`
- `artifact_metadata/calibration_manifest.json`
- `artifact_metadata/budget_fbit_8_4_mem132.json`
- `reference_scripts/`: minimal official PM-KVQ allocation, sensitivity,
  progressive cache, SmoothAttention rep-scale, quantizer, and calibration
  dataset scripts used to audit ATC's cache-wide approximation.
- `repo_meta.json`

Not included:

- Full PM-KVQ repo.
- `sensitivity.pt`, `rep_scales_k4v4.pt`, `max_keys.pt`, or any other `.pt`
  artifact.
- Real packed KV cache, real bit-width shrinking, or memory-saving kernels.

Note: MATH500/GSM8K PM-KVQ rows can show `16:1.00` because this budget does
not trigger low-bit rewrite for those shorter effective cache lengths.
