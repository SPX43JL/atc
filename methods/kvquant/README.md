# KVQuant Current Snapshot

Current ATC use:

- Formal method: `KVQuant NUQ3`.
- Accounting: NUQ 3-bit bulk plus global first-token FP16 and 1% sparse
  outliers charged as FP16-equivalent effective bits.
- Expected effective average KV bits: about LongBench `3.13`, MATH500 `3.15`,
  GSM8K `3.14`.
- Qwen2 pre-RoPE K hook is enabled in the ATC vLLM integration: K is fake
  quantized before RoPE, and cache write avoids applying K quantization again.

Included files:

- `artifact_metadata/metadata.json` for Wikitext-2 n16 l2048 NUQ artifact.
- `reference_scripts/`: minimal official KVQuant activation-cache,
  SimQuant/NUQ/outlier, model-parse, data-loading, and deployment-hook scripts
  used to audit ATC's pre-RoPE NUQ mapping.
- `repo_meta.json`

Not included:

- Full KVQuant repo.
- `nuq_artifact.pt` or any `.pt` calibration artifact.
- Packed dense/sparse low-bit storage or CUDA kernels.
