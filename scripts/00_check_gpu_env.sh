#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
echo "== Host =="
hostname || true
whoami || true
date -Is || true
echo

echo "== GPU / CUDA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
else
  echo "nvidia-smi not found"
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found; vLLM pip wheels can still work if driver/CUDA runtime is compatible."
fi
echo

echo "== Python =="
command -v python3 || true
python3 --version || true
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  "$PROJECT_DIR/.venv/bin/python" --version
  "$PROJECT_DIR/.venv/bin/python" - <<'PY' || true
mods = ["torch", "vllm", "modelscope", "transformers", "openai", "aiohttp", "pandas", "numpy", "tqdm", "matplotlib"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'installed')}")
    except Exception as e:
        print(f"{m}: missing ({e})")
PY
fi
echo

echo "== Disk =="
df -h "$PROJECT_DIR" /root /tmp 2>/dev/null || df -h
echo

echo "== Network quick check =="
python3 - <<'PY' || true
import urllib.request
for url in ["https://modelscope.cn", "https://pypi.org"]:
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            print(url, r.status)
    except Exception as e:
        print(url, "FAILED", repr(e))
PY
