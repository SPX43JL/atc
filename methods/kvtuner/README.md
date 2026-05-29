# KVTuner Current Snapshot

Current ATC use:

- `kvtuner_pertoken_c4_00`: Qwen2.5-7B-Instruct per-token asymmetric C4.00
  preset.
- `kvtuner_kivi_c3_92`: Qwen2.5-7B-Instruct KIVI-mode C3.92 preset.
- The KVTuner-KIVI variant uses the KVTuner preset group/residual settings,
  not the pure KIVI K2/K4 settings.

Included files:

- `presets/Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml`
- `presets/Qwen2.5-7B-Instruct_kivi_KVTuner4_0.yaml`
- `reference_scripts/`: minimal official KVTuner preset parser, LongBench/GSM
  evaluation helpers, and flexible-quant cache/quantizer code used for
  alignment review.
- `repo_meta.json`

Not included:

- Full KVTuner repo.
- New KVTuner search/calibration outputs.
- Old or superseded presets.
