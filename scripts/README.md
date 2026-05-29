# Current Reproduction Scripts

These are the project scripts kept for the current useful ATC fake-quant
snapshot. They are the scripts reviewers should inspect first.

Core formal run:

- `prepare_multitask_eval_manifests.py`
- `run_formal_multitask_pipeline.sh`
- `run_multitask_serving_eval.py`
- `summarize_multitask_results.py`
- `start_vllm_fake_quant_gpu0.sh`
- `start_vllm_fake_quant_gpu1.sh`
- `start_router_fake_quant.sh`
- `wait_openai_endpoint.py`
- `check_openai_chat_ports.py`

Method/artifact helpers:

- KVQuant: `run_kvquant_nuq_artifact.py`,
  `run_kvquant_official_calibration.py`.
- PM-KVQ: `run_pmkvq_official_calibration.py`,
  `build_pmkvq_budget_artifact.py`,
  `write_pmkvq_calibration_manifest.py`.
- Shared fake-quant checks: `test_fake_quant_methods.py`.

Environment/serving helpers:

- `00_check_gpu_env.sh`
- `01_setup_env.sh`
- `02_download_model_modelscope.py`
- `03_prepare_datasets.sh`
- `benchmark_common.sh`
- `monitor_gpu.sh`
- `run_fake_quant_serving_workload.py`
- `simple_openai_chat_benchmark.py`

`run_fake_quant_smoke_all.sh` is included only as a quick single-GPU smoke
helper. The current formal results come from the multitask pipeline.
