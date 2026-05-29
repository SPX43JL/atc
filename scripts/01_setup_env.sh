#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$PROJECT_DIR/.venv"
mkdir -p "$PROJECT_DIR/logs"

# China-friendly defaults. Override these if another mirror is faster.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Pin vLLM to a CUDA-12-era release for NVIDIA driver 550 / CUDA 12.4.
# Newer vLLM releases may pull torch CUDA 13 wheels, which require newer drivers.
python -m pip install \
  "vllm==0.6.6.post1" \
  modelscope \
  "transformers==4.47.1" \
  openai \
  aiohttp \
  pandas \
  numpy \
  tqdm \
  matplotlib

echo "== Installed versions =="
python - <<'PY'
mods = ["torch", "vllm", "modelscope", "transformers", "openai", "aiohttp", "pandas", "numpy", "tqdm", "matplotlib"]
for m in mods:
    mod = __import__(m)
    print(f"{m}: {getattr(mod, '__version__', 'installed')}")
PY
