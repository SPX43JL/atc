#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-$PROJECT_DIR/.venv_dev/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL="${SERVED_MODEL:-Qwen2.5-7B-Instruct}"

RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/results/fake_quant/formal/multitask_longbench_math_200_chat_c8_20260526}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/fake_quant_multitask_20260526}"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/data/eval_manifests/20260526_multitask_200}"

DATASETS="${DATASETS:-qasper,narrativeqa,hotpotqa,passage_retrieval_en,passage_count,qmsum,math500,gsm8k}"
VARIANTS="${VARIANTS:-baseline,kivi2,kivi4,kvtuner_pertoken_c4_00,kvtuner_kivi_c3_92,kvquant,pmkvq_cachewide_mem132}"
LIMIT="${LIMIT:-200}"

HOST="${HOST:-127.0.0.1}"
PORT0="${PORT0:-8100}"
PORT1="${PORT1:-8101}"
ROUTER_PORT="${ROUTER_PORT:-9100}"
FORMAL_CONCURRENCY="${FORMAL_CONCURRENCY:-8}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-2}"
WORKLOAD="${WORKLOAD:-burst}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ENDPOINT="${ENDPOINT:-chat}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL="${RUN_FORMAL:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
TRACE_EVERY="${TRACE_EVERY:-16}"

KVTUNER_PERTOKEN_PRESET="${KVTUNER_PERTOKEN_PRESET:-$PROJECT_DIR/references/kv_methods/KVTuner/calibration_presets/Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml}"
KVTUNER_KIVI_PRESET="${KVTUNER_KIVI_PRESET:-$PROJECT_DIR/references/kv_methods/KVTuner/calibration_presets/Qwen2.5-7B-Instruct_kivi_KVTuner4_0.yaml}"
KVTUNER_PRESET="${KVTUNER_PRESET:-$KVTUNER_PERTOKEN_PRESET}"
KVQUANT_ARTIFACT="${KVQUANT_ARTIFACT:-$PROJECT_DIR/artifacts/kvquant/wikitext2_qwen25_n16_l2048_official}"
KVQUANT_NUQ_ARTIFACT="${KVQUANT_NUQ_ARTIFACT:-$PROJECT_DIR/artifacts/kvquant/wikitext2_qwen25_n16_l2048_nuq/nuq_artifact.pt}"
MIXKVQ_THRESHOLDS="${MIXKVQ_THRESHOLDS:-$PROJECT_DIR/artifacts/mixkvq/gsm8k_train_seed0_c2_7/thresholds.json}"
PMKVQ_ARTIFACT_DIR="${PMKVQ_ARTIFACT_DIR:-$PROJECT_DIR/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192}"
PMKVQ_BUDGET="${PMKVQ_BUDGET:-$PMKVQ_ARTIFACT_DIR/budget_fbit_8_4_mem132.pt}"
PMKVQ_REP_SCALES="${PMKVQ_REP_SCALES:-$PMKVQ_ARTIFACT_DIR/rep_scales_k4v4.pt}"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$MANIFEST_DIR"

VLLM_PIDS=()
ROUTER_PID=""
CURRENT_VARIANT=""

cleanup_stack() {
  if [[ -n "${ROUTER_PID:-}" ]]; then
    kill "$ROUTER_PID" >/dev/null 2>&1 || true
    wait "$ROUTER_PID" >/dev/null 2>&1 || true
    ROUTER_PID=""
  fi
  if [[ "${#VLLM_PIDS[@]}" -gt 0 ]]; then
    for pid in "${VLLM_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${VLLM_PIDS[@]}"; do
      wait "$pid" >/dev/null 2>&1 || true
    done
    VLLM_PIDS=()
  fi
  pkill -TERM -f "[v]llm serve .*--port $PORT0" 2>/dev/null || true
  pkill -TERM -f "[v]llm serve .*--port $PORT1" 2>/dev/null || true
  pkill -TERM -f "[s]imple_round_robin_router.py" 2>/dev/null || true
  sleep 4
}
trap cleanup_stack EXIT

split_csv() {
  local value="$1"
  value="${value//,/ }"
  echo "$value"
}

manifest_path() {
  local dataset="$1"
  case "$dataset" in
    math500) echo "$MANIFEST_DIR/math500_first${LIMIT}.jsonl" ;;
    gsm8k) echo "$MANIFEST_DIR/gsm8k_first${LIMIT}.jsonl" ;;
    *) echo "$MANIFEST_DIR/longbench_${dataset}_first${LIMIT}.jsonl" ;;
  esac
}

wait_http() {
  local url="$1"
  local label="$2"
  local max_attempts="${3:-180}"
  for _ in $(seq 1 "$max_attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[ready] $label $url"
      return 0
    fi
    sleep 2
  done
  echo "[error] timed out waiting for $label $url" >&2
  return 1
}

reset_quant_env() {
  unset ATC_KV_FAKE_QUANT_METHOD
  unset ATC_KIVI_K_BITS ATC_KIVI_V_BITS ATC_KIVI_GROUP_SIZE ATC_KIVI_RESIDUAL_TOKENS
  unset ATC_KVTUNER_CONFIG_PATH ATC_KVTUNER_PRESET_PATH ATC_KVTUNER_GROUP_SIZE ATC_KVTUNER_RESIDUAL_TOKENS ATC_KVTUNER_QUANT_MODE ATC_KVTUNER_VARIANT_LABEL
  unset ATC_KVQUANT_BITS ATC_KVQUANT_GROUP_SIZE ATC_KVQUANT_OUTLIER_RATIO ATC_KVQUANT_OUTLIER_RATE ATC_KVQUANT_USE_NUQ ATC_KVQUANT_FIRST_TOKENS_FP16 ATC_KVQUANT_ARTIFACT_PATH ATC_KVQUANT_NUQ_ARTIFACT_PATH ATC_KVQUANT_PREROPE ATC_KVQUANT_STRICT_ARTIFACT ATC_KVQUANT_SPARSITY_THRESHOLD
  unset ATC_PMKVQ_BUDGET_PATH ATC_PMKVQ_REP_SCALES_PATH ATC_PMKVQ_FBIT_CHOICES ATC_PMKVQ_BUDGET_EVAL_LEN ATC_PMKVQ_CACHEWIDE_TIMING ATC_PMKVQ_MIN_BITS
  unset ATC_MIXKVQ_TARGET_BITS ATC_MIXKVQ_CONFIG ATC_MIXKVQ_BF16_TAU ATC_MIXKVQ_INT4_TAU ATC_MIXKVQ_GROUP_SIZE ATC_MIXKVQ_RESIDUAL_TOKENS ATC_MIXKVQ_SINK_TOKENS ATC_MIXKVQ_THRESHOLDS_PATH ATC_MIXKVQ_STRICT_THRESHOLDS
}

configure_variant() {
  local variant="$1"
  reset_quant_env
  export ATC_KV_FAKE_QUANT_LOG_PATH="$LOG_DIR/${variant}_kv_fake_quant.jsonl"
  export ATC_ROUTER_TRACE_PATH="$LOG_DIR/${variant}_router_trace.jsonl"
  export ATC_SERVING_STATE_PATH="$LOG_DIR/${variant}_serving_state.json"
  export ATC_KV_FAKE_QUANT_LOG_EVERY="$TRACE_EVERY"

  case "$variant" in
    baseline)
      export RUNTIME_METHOD="none"
      export ATC_KV_FAKE_QUANT_METHOD="none"
      ;;
    kivi2)
      export RUNTIME_METHOD="kivi"
      export ATC_KV_FAKE_QUANT_METHOD="kivi"
      export ATC_KIVI_K_BITS=2
      export ATC_KIVI_V_BITS=2
      export ATC_KIVI_GROUP_SIZE=32
      export ATC_KIVI_RESIDUAL_TOKENS=128
      ;;
    kivi4)
      export RUNTIME_METHOD="kivi"
      export ATC_KV_FAKE_QUANT_METHOD="kivi"
      export ATC_KIVI_K_BITS=4
      export ATC_KIVI_V_BITS=4
      export ATC_KIVI_GROUP_SIZE=32
      export ATC_KIVI_RESIDUAL_TOKENS=128
      ;;
    kvtuner|kvtuner_pertoken_c4_00)
      export RUNTIME_METHOD="kvtuner"
      export ATC_KV_FAKE_QUANT_METHOD="kvtuner"
      export ATC_KVTUNER_PRESET_PATH="$KVTUNER_PERTOKEN_PRESET"
      export ATC_KVTUNER_GROUP_SIZE=-1
      export ATC_KVTUNER_RESIDUAL_TOKENS=0
      export ATC_KVTUNER_QUANT_MODE=pertoken
      export ATC_KVTUNER_VARIANT_LABEL="KVTuner per-token-asym C4.00"
      ;;
    kvtuner_kivi_c3_92)
      export RUNTIME_METHOD="kvtuner"
      export ATC_KV_FAKE_QUANT_METHOD="kvtuner"
      export ATC_KVTUNER_PRESET_PATH="$KVTUNER_KIVI_PRESET"
      export ATC_KVTUNER_GROUP_SIZE=32
      export ATC_KVTUNER_RESIDUAL_TOKENS=32
      export ATC_KVTUNER_QUANT_MODE=kivi
      export ATC_KVTUNER_VARIANT_LABEL="KVTuner-KIVI C3.92"
      ;;
    kvquant)
      export RUNTIME_METHOD="kvquant"
      export ATC_KV_FAKE_QUANT_METHOD="kvquant"
      export ATC_KVQUANT_BITS=3
      export ATC_KVQUANT_GROUP_SIZE=32
      export ATC_KVQUANT_OUTLIER_RATE=0.01
      export ATC_KVQUANT_SPARSITY_THRESHOLD=0.99
      export ATC_KVQUANT_USE_NUQ=1
      export ATC_KVQUANT_FIRST_TOKENS_FP16=1
      export ATC_KVQUANT_ARTIFACT_PATH="$KVQUANT_ARTIFACT"
      export ATC_KVQUANT_NUQ_ARTIFACT_PATH="$KVQUANT_NUQ_ARTIFACT"
      export ATC_KVQUANT_PREROPE=1
      export ATC_KVQUANT_STRICT_ARTIFACT=1
      ;;
    pmkvq_cachewide_mem132)
      export RUNTIME_METHOD="pmkvq_cachewide"
      export ATC_KV_FAKE_QUANT_METHOD="pmkvq_cachewide"
      export ATC_PMKVQ_BUDGET_PATH="$PMKVQ_BUDGET"
      export ATC_PMKVQ_REP_SCALES_PATH="$PMKVQ_REP_SCALES"
      export ATC_PMKVQ_FBIT_CHOICES="8,4"
      export ATC_PMKVQ_BUDGET_EVAL_LEN=8192
      export ATC_PMKVQ_CACHEWIDE_TIMING=defer_current
      export ATC_PMKVQ_MIN_BITS=4
      ;;
    mixkvq_c2_7)
      export RUNTIME_METHOD="mixkvq"
      export ATC_KV_FAKE_QUANT_METHOD="mixkvq"
      export ATC_MIXKVQ_TARGET_BITS=2.7
      export ATC_MIXKVQ_CONFIG="C2.7"
      export ATC_MIXKVQ_BF16_TAU=2.0
      export ATC_MIXKVQ_GROUP_SIZE=32
      export ATC_MIXKVQ_RESIDUAL_TOKENS=128
      export ATC_MIXKVQ_SINK_TOKENS=32
      export ATC_MIXKVQ_THRESHOLDS_PATH="$MIXKVQ_THRESHOLDS"
      export ATC_MIXKVQ_STRICT_THRESHOLDS=1
      ;;
    *)
      echo "[error] unknown variant: $variant" >&2
      exit 2
      ;;
  esac
}

start_vllm_one() {
  local gpu="$1"
  local port="$2"
  local log="$LOG_DIR/${CURRENT_VARIANT}_vllm_gpu${gpu}_port${port}.log"
  echo "[start] vLLM gpu=$gpu port=$port variant=$CURRENT_VARIANT runtime=$RUNTIME_METHOD"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export GPU_ID="$gpu"
    export PORT="$port"
    export PROJECT_DIR="$PROJECT_DIR"
    export MODEL_PATH="$MODEL_PATH"
    export MAX_MODEL_LEN="$MAX_MODEL_LEN"
    export GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
    export PYTHONPATH="$PROJECT_DIR/vllm_src:${PYTHONPATH:-}"
    exec bash "$PROJECT_DIR/scripts/start_vllm_fake_quant_gpu${gpu}.sh"
  ) >"$log" 2>&1 &
  VLLM_PIDS+=("$!")
}

start_router() {
  local log="$LOG_DIR/${CURRENT_VARIANT}_router.log"
  echo "[start] router variant=$CURRENT_VARIANT"
  (
    export PROJECT_DIR="$PROJECT_DIR"
    export ROUTER_PORT="$ROUTER_PORT"
    export VLLM_BACKENDS="http://127.0.0.1:$PORT0/v1/chat/completions,http://127.0.0.1:$PORT1/v1/chat/completions"
    export PYTHONPATH="$PROJECT_DIR/vllm_src:${PYTHONPATH:-}"
    exec bash "$PROJECT_DIR/scripts/start_router_fake_quant.sh"
  ) >"$log" 2>&1 &
  ROUTER_PID="$!"
  wait_http "http://$HOST:$ROUTER_PORT/health" "router" 60
}

start_stack() {
  local variant="$1"
  cleanup_stack
  CURRENT_VARIANT="$variant"
  configure_variant "$variant"
  rm -f "$ATC_KV_FAKE_QUANT_LOG_PATH" "$ATC_ROUTER_TRACE_PATH" "$ATC_SERVING_STATE_PATH"
  start_vllm_one 0 "$PORT0"
  start_vllm_one 1 "$PORT1"
  wait_http "http://$HOST:$PORT0/v1/models" "vLLM gpu0" 240
  wait_http "http://$HOST:$PORT1/v1/models" "vLLM gpu1" 240
  start_router
}

run_dataset() {
  local variant="$1"
  local dataset="$2"
  local indices="$3"
  local concurrency="$4"
  local output="$5"
  local manifest
  manifest="$(manifest_path "$dataset")"
  if [[ ! -f "$manifest" ]]; then
    echo "[error] manifest not found: $manifest" >&2
    exit 1
  fi
  if [[ -s "$output" ]]; then
    echo "[skip] existing $output"
    return 0
  fi
  echo "[eval] variant=$variant dataset=$dataset concurrency=$concurrency indices=${indices:-all}"
  "$PYTHON" "$PROJECT_DIR/scripts/run_multitask_serving_eval.py" \
    --base-url "http://$HOST:$ROUTER_PORT" \
    --model "$SERVED_MODEL" \
    --manifest "$manifest" \
    --dataset "$dataset" \
    --method "$RUNTIME_METHOD" \
    --method-variant "$variant" \
    --workload "$WORKLOAD" \
    --max-concurrency "$concurrency" \
    --endpoint "$ENDPOINT" \
    --temperature "$TEMPERATURE" \
    --indices "$indices" \
    --output "$output"
}

if [[ "$RUN_STATIC" == "1" ]]; then
  echo "[static] py_compile and smoke unit tests"
  "$PYTHON" -m py_compile \
    "$PROJECT_DIR/scripts/prepare_multitask_eval_manifests.py" \
    "$PROJECT_DIR/scripts/run_multitask_serving_eval.py" \
    "$PROJECT_DIR/scripts/run_kvquant_nuq_artifact.py" \
    "$PROJECT_DIR/scripts/summarize_multitask_results.py" \
    "$PROJECT_DIR/router/simple_round_robin_router.py" \
    "$PROJECT_DIR/vllm_src/vllm/model_executor/models/qwen2.py" \
    "$PROJECT_DIR/vllm_src/vllm/attention/ops/atc_kv_fake_quant/adapters.py" \
    "$PROJECT_DIR/vllm_src/vllm/attention/ops/atc_kv_fake_quant/core.py" \
    "$PROJECT_DIR/vllm_src/vllm/attention/ops/atc_kv_fake_quant/runtime.py" \
    "$PROJECT_DIR/vllm_src/vllm/attention/ops/atc_kv_fake_quant/trace.py"
  bash -n "$PROJECT_DIR/scripts/run_formal_multitask_pipeline.sh"
  PYTHONPATH="$PROJECT_DIR/vllm_src:${PYTHONPATH:-}" "$PYTHON" "$PROJECT_DIR/scripts/test_fake_quant_methods.py"
fi

if [[ "$RUN_PREPARE" == "1" ]]; then
  echo "[prepare] manifests limit=$LIMIT datasets=$DATASETS"
  "$PYTHON" "$PROJECT_DIR/scripts/prepare_multitask_eval_manifests.py" \
    --out-dir "$MANIFEST_DIR" \
    --limit "$LIMIT" \
    --datasets "$DATASETS"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[smoke] baseline all datasets first 5, then quant variants qasper/qmsum first 10"
  start_stack baseline
  for dataset in $(split_csv "$DATASETS"); do
    run_dataset baseline "$dataset" "0-4" "$SMOKE_CONCURRENCY" \
      "$RESULT_DIR/smoke_baseline_${dataset}_c${SMOKE_CONCURRENCY}.json"
  done
  cleanup_stack

  for variant in $(split_csv "$VARIANTS"); do
    [[ "$variant" == "baseline" ]] && continue
    start_stack "$variant"
    run_dataset "$variant" qasper "0-9" "$SMOKE_CONCURRENCY" \
      "$RESULT_DIR/smoke_${variant}_qasper_c${SMOKE_CONCURRENCY}.json"
    run_dataset "$variant" qmsum "0-9" "$SMOKE_CONCURRENCY" \
      "$RESULT_DIR/smoke_${variant}_qmsum_c${SMOKE_CONCURRENCY}.json"
    cleanup_stack
  done
fi

if [[ "$RUN_FORMAL" == "1" ]]; then
  echo "[formal] variants=$VARIANTS datasets=$DATASETS"
  for variant in $(split_csv "$VARIANTS"); do
    start_stack "$variant"
    for dataset in $(split_csv "$DATASETS"); do
      run_dataset "$variant" "$dataset" "" "$FORMAL_CONCURRENCY" \
        "$RESULT_DIR/${variant}_${dataset}_burst_c${FORMAL_CONCURRENCY}.json"
    done
    cleanup_stack
  done
fi

"$PYTHON" "$PROJECT_DIR/scripts/summarize_multitask_results.py" \
  --results-dir "$RESULT_DIR" \
  --trace-dir "$LOG_DIR" \
  --output-json "$RESULT_DIR/summary.json" \
  --output-md "$RESULT_DIR/summary.md" \
  --output-csv "$RESULT_DIR/summary.csv"

echo "[done] $RESULT_DIR"
