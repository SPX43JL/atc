#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
source "$PROJECT_DIR/.venv_dev/bin/activate"
mkdir -p "$PROJECT_DIR/logs/fake_quant"

METHOD="${ATC_KV_FAKE_QUANT_METHOD:-${METHOD:-none}}"
export ATC_KV_FAKE_QUANT_METHOD="$METHOD"
export ROUTER_PORT="${ROUTER_PORT:-9100}"
export VLLM_BACKENDS="${VLLM_BACKENDS:-http://127.0.0.1:8100/v1/chat/completions,http://127.0.0.1:8101/v1/chat/completions}"
export ATC_SERVING_STATE_PATH="${ATC_SERVING_STATE_PATH:-$PROJECT_DIR/logs/fake_quant/${METHOD}_serving_state.json}"
export ATC_ROUTER_TRACE_PATH="${ATC_ROUTER_TRACE_PATH:-$PROJECT_DIR/logs/fake_quant/${METHOD}_router_trace.jsonl}"

exec python "$PROJECT_DIR/router/simple_round_robin_router.py" \
  2>&1 | tee -a "$PROJECT_DIR/logs/fake_quant/router_${METHOD}.log"
