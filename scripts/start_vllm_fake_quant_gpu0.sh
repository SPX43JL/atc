#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
source "$PROJECT_DIR/.venv_dev/bin/activate"
mkdir -p "$PROJECT_DIR/logs/fake_quant"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Keep the hook on the Python cache-write path; this does not modify kernels.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

METHOD="${ATC_KV_FAKE_QUANT_METHOD:-${METHOD:-none}}"
export ATC_KV_FAKE_QUANT_METHOD="$METHOD"
export ATC_KV_FAKE_QUANT_LOG_EVERY="${ATC_KV_FAKE_QUANT_LOG_EVERY:-200}"
export ATC_KV_FAKE_QUANT_LOG_PATH="${ATC_KV_FAKE_QUANT_LOG_PATH:-$PROJECT_DIR/logs/fake_quant/${METHOD}_kv_fake_quant.jsonl}"
export ATC_SERVING_STATE_PATH="${ATC_SERVING_STATE_PATH:-$PROJECT_DIR/logs/fake_quant/${METHOD}_serving_state.json}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8100}"

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --dtype auto \
  --kv-cache-dtype auto \
  --enforce-eager \
  2>&1 | tee -a "$PROJECT_DIR/logs/fake_quant/vllm_gpu${GPU_ID}_${METHOD}.log"
