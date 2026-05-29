#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
METHODS="${METHODS:-none kivi kvtuner kvquant pmkvq pmkvq_cachewide mixkvq}"
PORT="${PORT:-8100}"
LIMIT="${LIMIT:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-6000}"
MAX_TOKENS="${MAX_TOKENS:-64}"

stop_server() {
  ps -eo pid,args \
    | awk '/vllm serve/ && /--port '"$PORT"'/ && !/awk/ {print $1}' \
    | xargs -r kill || true
  ps -eo pid,args \
    | awk '/start_vllm_fake_quant_gpu0.sh/ && !/awk/ {print $1}' \
    | xargs -r kill || true
  sleep 5
}

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -120 "$PROJECT_DIR/logs/fake_quant/vllm_gpu0_${1}.log" || true
  return 1
}

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs/fake_quant"

for method in $METHODS; do
  echo "=== method: $method ==="
  stop_server
  METHOD="$method" PORT="$PORT" nohup bash scripts/start_vllm_fake_quant_gpu0.sh \
    > "$PROJECT_DIR/logs/fake_quant/launcher_${method}.log" 2>&1 &
  wait_ready "$method"
  METHOD="$method" LIMIT="$LIMIT" MAX_CONCURRENCY="$MAX_CONCURRENCY" \
    MAX_CONTEXT_CHARS="$MAX_CONTEXT_CHARS" MAX_TOKENS="$MAX_TOKENS" \
    BASE_URL="http://127.0.0.1:${PORT}" \
    bash scripts/run_fake_quant_qasper_eval.sh
done

stop_server
