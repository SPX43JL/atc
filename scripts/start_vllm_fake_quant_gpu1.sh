#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-8101}"
export GPU_ID PORT

exec "$PROJECT_DIR/scripts/start_vllm_fake_quant_gpu0.sh"
