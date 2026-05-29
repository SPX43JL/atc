# KIVI Current Snapshot

Current ATC use:

- Methods in formal table: `KIVI-2` and `KIVI-4`.
- Group size: `G=32`.
- Residual length: `R=128`.
- K cache: per-channel asymmetric fake quant.
- V cache: per-token fake quant.
- Residual/cache-wide accounting follows the current ATC vLLM fake-quant path:
  K residual uses the unflushed `seq_len % R` group, V residual keeps recent
  `R` tokens.

Included files:

- `config/dataset2prompt.json`
- `config/dataset2maxlen.json`
- `config/model2maxlen.json`
- `reference_scripts/`: minimal official KIVI LongBench/evaluator/cache
  quantization scripts used to audit ATC's fake-quant mapping.
- `repo_meta.json`

Not included:

- Full KIVI repo.
- Packed low-bit KV cache or CUDA kernels.
- Old exploratory Qasper-only results.
