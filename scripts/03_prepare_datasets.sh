#!/usr/bin/env bash
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/atc_vllm_sched}"
mkdir -p "$PROJECT_DIR/data/burstgpt" "$PROJECT_DIR/data/sharegpt" "$PROJECT_DIR/logs"
log() { echo "[$(date -Is)] $*"; }

download_try() {
  local url="$1"
  local out="$2"
  log "Trying $url"
  if command -v wget >/dev/null 2>&1; then
    wget -c --timeout=60 --tries=3 -O "$out" "$url" && return 0
    wget --no-check-certificate -c --timeout=60 --tries=3 -O "$out" "$url" && return 0
  fi
  curl -k -L --retry 3 --retry-delay 5 -C - -o "$out" "$url" && return 0
  return 1
}

log "Preparing BurstGPT"
BURST_OUT="$PROJECT_DIR/data/burstgpt/BurstGPT_1.csv"
if [ ! -s "$BURST_OUT" ]; then
  rm -f "$BURST_OUT"
  download_try "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv" "$BURST_OUT" || \
  download_try "https://gh-proxy.com/https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv" "$BURST_OUT" || \
  download_try "https://mirror.ghproxy.com/https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv" "$BURST_OUT" || true
fi
if [ -s "$BURST_OUT" ]; then
  head -n 1001 "$BURST_OUT" > "$PROJECT_DIR/data/burstgpt/BurstGPT_1_first1000.csv"
  python3 - <<PY
import csv
p = "$BURST_OUT"
with open(p, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    n = gpt4 = 0
    for row in r:
        n += 1
        try:
            ok = row.get('Model') == 'GPT-4' and float(row.get('Response tokens') or 0) > 0
        except ValueError:
            ok = False
        gpt4 += int(ok)
print(f"BurstGPT rows={n}, usable_gpt4_rows={gpt4}, path={p}")
PY
else
  cat > "$PROJECT_DIR/data/burstgpt/README_DOWNLOAD_FAILED.md" <<'MSG'
BurstGPT was not downloaded automatically because the server could not reach GitHub raw/proxy URLs.
Expected file: BurstGPT_1.csv
Source: https://github.com/HPMLL/BurstGPT/blob/main/data/BurstGPT_1.csv
MSG
  log "BurstGPT download failed; wrote data/burstgpt/README_DOWNLOAD_FAILED.md"
fi

log "Preparing ShareGPT"
SHARE_OUT="$PROJECT_DIR/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -s "$SHARE_OUT" ]; then
  rm -f "$SHARE_OUT"
  download_try "https://modelscope.cn/datasets/otavia/ShareGPT_Vicuna_unfiltered/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json" "$SHARE_OUT" || \
  download_try "https://www.modelscope.cn/datasets/otavia/ShareGPT_Vicuna_unfiltered/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json" "$SHARE_OUT" || \
  download_try "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" "$SHARE_OUT" || true
fi
if [ -s "$SHARE_OUT" ]; then
  python3 - <<PY
import json
p = "$SHARE_OUT"
with open(p, encoding='utf-8') as f:
    data = json.load(f)
print(f"ShareGPT conversations={len(data)}, path={p}")
PY
else
  cat > "$PROJECT_DIR/data/sharegpt/README_DOWNLOAD_FAILED.md" <<'MSG'
ShareGPT was not downloaded automatically.
Expected file: ShareGPT_V3_unfiltered_cleaned_split.json
Preferred sources:
  ModelScope: https://modelscope.cn/datasets/otavia/ShareGPT_Vicuna_unfiltered
  Hugging Face: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
MSG
  log "ShareGPT download failed; wrote data/sharegpt/README_DOWNLOAD_FAILED.md"
fi
