#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
source "$PROJECT_DIR/.venv/bin/activate"
mkdir -p "$PROJECT_DIR/logs/benchmark" "$PROJECT_DIR/results/benchmark"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
DATASET_NAME="${DATASET_NAME:-burstgpt}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
REQUEST_RATE="${REQUEST_RATE:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
case "$DATASET_NAME" in
  sharegpt) DATASET_PATH="${DATASET_PATH:-$PROJECT_DIR/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}" ;;
  burstgpt) DATASET_PATH="${DATASET_PATH:-$PROJECT_DIR/data/burstgpt/BurstGPT_1.csv}" ;;
  *) DATASET_PATH="${DATASET_PATH:-}" ;;
esac
run_bench() {
  local base_url="$1"
  local label="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local result="$PROJECT_DIR/results/benchmark/${label}_${DATASET_NAME}_${stamp}.json"
  local log="$PROJECT_DIR/logs/benchmark/${label}_${DATASET_NAME}_${stamp}.log"
  if vllm bench serve --help >/dev/null 2>&1; then
    vllm bench serve \
      --backend openai-chat \
      --endpoint /v1/chat/completions \
      --base-url "$base_url" \
      --model "$MODEL_NAME" \
      --tokenizer "$PROJECT_DIR/models/Qwen2.5-7B-Instruct" \
      --dataset-name "$DATASET_NAME" \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$REQUEST_RATE" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --save-result \
      --result-dir "$PROJECT_DIR/results/benchmark" \
      2>&1 | tee "$log"
  else
    python "$PROJECT_DIR/scripts/simple_openai_chat_benchmark.py" \
      --base-url "$base_url" \
      --model "$MODEL_NAME" \
      --dataset-name "$DATASET_NAME" \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$REQUEST_RATE" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --output "$result" \
      2>&1 | tee "$log"
  fi
}
