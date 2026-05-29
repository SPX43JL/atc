# vLLM KV Fake Quant Serving 实验说明

## 当前方向

本阶段只做 Python/vLLM 上层 fake quant 评估，不做真实低精度 KV cache 存储，不修改 CUDA/Triton/PagedAttention kernel，也不声称显存下降或 kernel 加速。

目标是在同一个 vLLM serving 场景下，把已有 KV cache 量化论文方法抽象成统一 fake quant 策略，比较输出质量和 serving 压力下的行为，为后续新 serving 动态精度方法提供 related-work baseline。

## 正式对齐状态（2026-05-18）

上一轮基于 Qasper dev、本地字符截断、自定 `max_tokens=64/128`、自定 `max_context_chars=6000/12000` 的结果全部只保留为 **pre-alignment exploratory**。这些结果不能作为正式 related-work 对比，也不会进入正式结果表。

正式实验入口已改为 LongBench 官方 Qasper：

- 数据集：`THUDM/LongBench`，config `qasper`，split `test`，官方顺序前 200 条。
- Prompt：来自 `references/kv_methods/KIVI/config/dataset2prompt.json`。
- Generation length：来自 `references/kv_methods/KIVI/config/dataset2maxlen.json`，Qasper 为 `128`。
- Context 截断：按 KIVI/KVTuner repo 中 LongBench 脚本的 token-based middle truncation。
- 正式 evaluator：`scripts/run_longbench_qasper_eval.py`。
- 正式 pipeline：`scripts/run_formal_longbench_qasper_pipeline.sh`，默认使用 OpenAI chat endpoint，让 Qwen2.5-Instruct 走 chat template。
- 方法/参数来源报告：`docs/paper_repo_config_alignment_report.md`。
- 正式实验报告：`docs/fake_quant_experiment_report.md`。

serving 侧的 burst/time-series、max concurrency 8/16、router round-robin 是本项目的 `framework serving extension / not specified in paper`，只用于描述 serving 压力，不用于声称 fake quant 真实加速。

## 2026-05-21 修复状态

最新可信输出目录：

`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8_globalpos_pm8/`

KIVI-4 补跑目录：

`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8_kivi4/`

KVTuner effective-bit 补跑目录：

`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8_kvtuner_effectivebits/`

PM-KVQ 低比特诊断目录：

`/root/atc_vllm_sched/results/fake_quant/diagnostic/pmkvq_qasper_surrogate_official_n8_all4_c8_l40/`

`/root/atc_vllm_sched/results/fake_quant/diagnostic/pmkvq_qasper_surrogate_official_n8_mem90_c8_l40/`

本轮修复内容：

- 修正 asymmetric fake quant 的 min-offset 公式，避免旧 zero-point clamp 破坏全正/全负 group。
- `reshape_and_cache` / `reshape_and_cache_flash` 均传入 `slot_mapping` 和 block size；fake quant 按 request/sequence segment 处理，而不是把整个 flattened batch 当一条序列。
- attention context 传入 `attn_metadata`，可读取 `query_start_loc`、`context_lens_tensor`、`seq_lens_tensor`。
- PMKVQ 增加官方 allocation artifact 构建脚本和 Qasper-train surrogate rep-scale 构建脚本；当前正式 PM 使用保守 8-bit bulk budget、`max_len=8192`、`ATC_PMKVQ_PREFILL_WINDOW_MODE=defer`、`ATC_PMKVQ_BUDGET_EVAL_LEN=8192`。
- PMKVQ 的 `defer` / `budget_eval_len` 是 serving fake approximation：用于在 write-only hook 中近似官方 progressive shrink 后的最终 cache occupancy，避免 chunked prefill 每个 chunk 永久保留 16-bit window。它不等同于真实 PM-KVQ cache-wide bit-width shrinking。
- KIVI/KVTuner trace 增加 residual-aware effective precision summary，同时保留 nominal bit 字段；KIVI-4 的 avg KV bits 因 recent residual window 计入 16-bit，为 `4.38` 而不是 nominal `4.00`。
- 新增 `scripts/run_pmkvq_official_calibration.py`，在不修改官方 PM-KVQ repo 的前提下调用官方 sensitivity/allocation/max-key/rep-scale 代码，并在 wrapper 内处理当前 Qwen2.5/transformers 兼容问题。

最新 200 条 Qasper burst c8 结果：baseline F1 `44.45`；KIVI `41.74`，output trunc `1/200`；KVTuner `44.71`；KVQuant `43.31`，output trunc `0`；PMKVQ `44.09`，avg KV bits `8.00`，output trunc `0`；MixKVQ `43.68`。

该旧 KVQuant 高分不是完整 calibrated KVQuant 结论：当时 Fisher/NUQ calibration artifact 仍缺失，结果是 official quantizer-backed closest fake-quant approximation；2026-05-28 已补齐完整 Wikitext-2 NUQ artifact，但尚未重跑 formal。KIVI 剩余的唯一 trunc 样本是 index `127`，输出在模型名列表上重复到 `max_gen=128`；baseline 同样样本 41 token 正常停止。若通过提高 KIVI bit 或增大 residual 强行消除该 trunc，将不再是 KIVI paper K2V2/G32/R128 baseline。

KIVI-4 补跑使用 `K4V4/G32/R128`，Qasper-200 c8 F1 `44.91`、output trunc `0/200`、residual-aware avg KV bits `4.38`。这与 KIVI repo LongBench 文档中对 MQA/GQA 模型推荐 KIVI-4 更稳的说明一致；KIVI-2 仍保留为更激进压缩点。

KVTuner 为刷新 residual-aware bit 统计补跑到新目录：Qasper-200 c8 F1 `44.53`、output trunc `0/200`、effective avg KV bits `4.07`、distribution `2:0.53,4:0.20,8:0.26,16:0.01`。旧主目录 KVTuner F1 `44.71` 与该补跑差异很小，仍按 tie-level variation 解释。

## 2026-05-22 PM/KVQuant 继续对齐状态

当前硬件已切换为双 NVIDIA A800-SXM4-40GB。

PM-KVQ 的 RedPajama 前置准备已从不可访问的 `togethercomputer/RedPajama-Data-1T-Sample` 改为可访问的 `togethercomputer/RedPajama-Data-1T` arXiv streaming subset。根据本地 `PM-KVQ.pdf`，论文 calibration 使用 RedPajama arXiv subset、512 samples、2048 tokens，并用 positional interpolation 近似 8192 context；当前 artifact 目录为：

`/root/atc_vllm_sched/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192/`

已完成 `sensitivity.pt`、`budget_fbit_4_2.pt`、`max_keys.pt`、`rep_scales_k4v4.pt` 和 `calibration_manifest.json`。manifest 记录：`n_samples=512`，每条 tokenized 长度 `2048`，`effective_len=8192`，JSONL 为 `/root/atc_vllm_sched/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192/calib_dataset_tokenized.jsonl`，sha256 为 `a97b062b5e83d54bce99b3cc12771cee856a87d64c459e338c0992a77b267a68`。注意：`Data-1T/arxiv` 是对不可访问 Sample repo 的最接近替代，且当前 deterministic stream first-512 与官方 seed=42 shuffle 不完全同源。

PM-KVQ 还新增 `pmkvq_cachewide` 路径：旧 `pmkvq` 仍是 write-only pre-write approximation；`pmkvq_cachewide` 在 vLLM 写入 FP16/BF16 cache 后，读取 attention metadata 中的 `block_tables`、`seq_lens`、`context_lens`，按官方 progressive sink/window/bulk budget 语义回写历史 KV cache slots。trace 新增 `cache_wide_coverage`、`rewritten_slots`、`skipped_slots`、`slot_target_bit_distribution`。这仍是 fake quant，不是 packed low-bit cache，但解决了旧 hook 不能回写旧 KV 的主要语义缺口。2026-05-22 又修复了 cache-wide 性能问题：旧实现逐 slot Python gather/scatter 且每步构造整段 GPU bit-map，200 条 Qasper 需要约 35 分钟；现在改为批量 tensor gather/scatter 和公式化 bit-count summary，优化后 200 条 formal `total_elapsed_s=151.57`。

PM-KVQ 当前稳定低比特 formal 目录：

`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260522_pmkvq_defer_current_mem132_fast_c8/`

该结果使用 RedPajama arXiv n512 artifact、官方 allocation 生成的 `8,4` choices `budget_fbit_8_4_mem132.pt`、`rep_scales_k4v4.pt`、`pmkvq_cachewide`、`ATC_PMKVQ_CACHEWIDE_TIMING=defer_current`。Qasper-200 burst c8：F1 `44.08`，failed `0/200`，output trunc `0/200`，cache-wide coverage `1.0`。budget artifact 的 layer bulk average 为 `4.71` bit；serving trace 的 residual/protected-token effective avg KV bits 为 `6.27`，distribution `4:0.57, 8:0.36, 16:0.07`。更激进的 4/2 和 all-4 diagnostics 在 HF official path 与 vLLM cache-wide path 都出现明显崩分或 trunc，当前不进入 formal 主表。

KVQuant 的 Wikitext-2 前置准备已完成，并新增 Fisher calibration driver：

`/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048_official/`

该目录包含 `fisher_summary.pt` 和 `metadata.json`，使用本地 Wikitext-2 16×2048 samples、KVQuant repo commit `57a2383`、本地 `kvquant.pdf` 中的 Fisher/sensitivity calibration 目标。Qwen2.5 没有官方 fork 的 `.act` projection module，因此 wrapper 只替换 K/V projection 以捕获 activation gradient，不修改官方 repo。该 Fisher-only artifact 仍为历史 diagnostics；正式 KVQuant 路径改为 pre-RoPE K hook + 完整 NUQ artifact。

## 已清理的旧方向

重置前已备份当前项目到：

`/root/atc_vllm_sched/backups/20260518_113451_before_fake_quant_reset/`

已从活动代码路径中清理或隔离：

- `vllm_src/csrc/atc_kivi_int4/`
- `vllm/attention/ops/atc_kivi_int4_cuda.py`
- packed INT4 KV cache allocation / decode / state 管理逻辑
- `cache_engine.py`、`xformers.py`、`selector.py` 中为 packed INT4 修改的逻辑
- `--kv-cache-dtype atc_kivi_int4` 和 `atc_fp8_e4m3` 这些旧实验 dtype
- CUDA quick test、packed decode test、KIVI INT4 启动脚本

现在 fake quant 仍使用官方普通 KV cache dtype：`--kv-cache-dtype auto`。

## 框架入口

核心入口在 vLLM Python 层：

`vllm_src/vllm/_custom_ops.py`

在调用官方：

`torch.ops._C_cache_ops.reshape_and_cache(...)`

之前，会调用：

`vllm.attention.ops.atc_kv_fake_quant.maybe_fake_quant_kv(...)`

这个 hook 只对 key/value 做 quantize-dequantize 扰动，然后仍写入普通 FP16/BF16 KV cache。它不改变 cache block 形状，不改变 cache dtype，不改变 CUDA kernel。

## 方法切换

通过环境变量选择方法：

```bash
export ATC_KV_FAKE_QUANT_METHOD=none      # baseline
export ATC_KV_FAKE_QUANT_METHOD=kivi
export ATC_KV_FAKE_QUANT_METHOD=kvtuner
export ATC_KV_FAKE_QUANT_METHOD=kvquant
export ATC_KV_FAKE_QUANT_METHOD=pmkvq
export ATC_KV_FAKE_QUANT_METHOD=pmkvq_cachewide
export ATC_KV_FAKE_QUANT_METHOD=mixkvq
```

通用参数：

```bash
export ATC_KV_FAKE_QUANT_BITS=4
export ATC_KV_FAKE_QUANT_GROUP_SIZE=32
export ATC_KV_FAKE_QUANT_PRESSURE=auto    # low / medium / high / auto
```

当前实现状态：

- `kivi`：论文对齐的 K per-channel、V per-token fake quant，主配置 `K2V2/G32/R128`；另补 `K4V4/G32/R128` 作为 GQA/MQA 稳定性对照。
- `kvtuner`：兼容别名，等价于 `kvtuner_pertoken_c4_00`；正式表同时提供 `kvtuner_pertoken_c4_00` 和 `kvtuner_kivi_c3_92` 两个 variant。
- `kvquant`：正式路径要求完整 Wikitext-2 NUQ artifact；K 在 Qwen2 `qkv_proj` split 后、`rotary_emb` 前 fake quant，V 在 cache-write hook fake quant，first token 按全序列位置保留 FP16。
- `pmkvq`：write-only progressive memory-budget fake quant approximation，保留用于和旧结果对照。
- `pmkvq_cachewide`：写入后回写历史 vLLM cache slots，按官方 PM-KVQ progressive sink/window/bulk budget 语义做 cache-wide fake quant；formal 低比特 PM-KVQ 应优先使用该方法名。
- `mixkvq`：按论文 query-aware salience 做 BF16/UINT4/UINT2 key channel mixed precision，V per-token 2-bit；默认 C2.7 为论文配置，threshold search 缺失处标注为 approximation。

## 单卡启动

当前资源拥挤时只用 GPU0：

```bash
cd /root/atc_vllm_sched
METHOD=none PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
```

切换方法时重启 vLLM：

```bash
METHOD=kivi PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
METHOD=kvtuner PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
METHOD=kvquant PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
METHOD=pmkvq PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
METHOD=mixkvq PORT=8100 bash scripts/start_vllm_fake_quant_gpu0.sh
```

日志：

```bash
tail -f /root/atc_vllm_sched/logs/fake_quant/vllm_gpu0_<method>.log
tail -f /root/atc_vllm_sched/logs/fake_quant/<method>_kv_fake_quant.jsonl
```

## Qasper 小样本质量评估

下面这一节是旧 smoke test 入口，只能用于快速检查服务，不再作为正式结果。

baseline：

```bash
cd /root/atc_vllm_sched
METHOD=none LIMIT=10 MAX_CONCURRENCY=2 bash scripts/run_fake_quant_qasper_eval.sh
```

某个方法：

```bash
METHOD=kivi LIMIT=10 MAX_CONCURRENCY=2 bash scripts/run_fake_quant_qasper_eval.sh
```

结果保存到：

`/root/atc_vllm_sched/results/fake_quant/qasper/`

## LongBench Qasper 正式评估

正式单次 evaluator：

```bash
cd /root/atc_vllm_sched
source .venv_dev/bin/activate
python scripts/run_longbench_qasper_eval.py \
  --base-url http://127.0.0.1:9100 \
  --method none \
  --workload sequential \
  --max-concurrency 1 \
  --limit 200 \
  --max-input-tokens 7500 \
  --output results/fake_quant/formal/longbench_qasper_200/formal_none_sequential_c1.json
```

如果只需要重跑某个 failed 样本，可用 `--indices` 保持官方顺序中的原始 index，不必全量重跑。例如 index 19：

```bash
python scripts/run_longbench_qasper_eval.py \
  --base-url http://127.0.0.1:9100 \
  --method pmkvq_cachewide \
  --workload burst \
  --max-concurrency 1 \
  --endpoint chat \
  --limit 200 \
  --indices 19 \
  --max-input-tokens 7500 \
  --output results/fake_quant/diagnostic/retry_index19.json
```

pipeline 也支持 `EVAL_INDICES=19` 透传。单样本 retry 适合修复 transient 502/timeout 的 F1/failed 记录；若要重新报告 latency/throughput，仍需完整 formal。

双卡双实例正式 pipeline 当前默认只跑 burst workload（max concurrency 8/16）。正式结果必须使用 `ENDPOINT=chat`。旧的 raw completions 结果只作为 endpoint 诊断记录，不作为正式结论。

```bash
cd /root/atc_vllm_sched
source .venv_dev/bin/activate
bash scripts/run_formal_longbench_qasper_pipeline.sh
```

只跑部分方法时：

```bash
METHODS="none kivi" LIMIT=200 bash scripts/run_formal_longbench_qasper_pipeline.sh
```

如果以后需要重新打开 time-series，可显式设置：

```bash
WORKLOADS="burst time-series" bash scripts/run_formal_longbench_qasper_pipeline.sh
```

正式结果位置：

- JSON：`results/fake_quant/formal/longbench_qasper_200/`
- Markdown 报告：`docs/fake_quant_experiment_report.md`
- vLLM/router 日志：`logs/fake_quant/`
- precision trace：`logs/fake_quant/<method>_kv_fake_quant.jsonl`

## serving workload

burst：

```bash
python scripts/run_fake_quant_serving_workload.py \
  --base-url http://127.0.0.1:8100 \
  --method kivi \
  --dataset qasper \
  --dataset-path /root/atc_vllm_sched/data/qasper/qasper-dev-v0.3.json \
  --workload burst \
  --num-prompts 10 \
  --max-concurrency 2 \
  --output /root/atc_vllm_sched/results/fake_quant/workload/kivi_burst_mc2.json
```

time-series：

```bash
python scripts/run_fake_quant_serving_workload.py \
  --base-url http://127.0.0.1:8100 \
  --method kivi \
  --dataset qasper \
  --dataset-path /root/atc_vllm_sched/data/qasper/qasper-dev-v0.3.json \
  --workload time-series \
  --request-rate 1 \
  --num-prompts 10 \
  --max-concurrency 2 \
  --output /root/atc_vllm_sched/results/fake_quant/workload/kivi_timeseries_mc2.json
```

这些 latency、throughput、queue wait 指标只用于描述 serving 压力，不作为 fake quant 带来真实加速的证据。

## 单卡和双卡边界

单卡现在可以完成：

- 代码清理和 fake quant hook 验证
- baseline / 五种方法的小样本 Qasper 质量评估
- max concurrency 1/2/4 的 burst 和 time-series 小规模 smoke test

后续必须等双卡可用再做：

- 双实例 serving
- router 下 max concurrency 8/16 的正式 burst/time-series 对比
- 两张 4090 上的更完整 serving 压力实验

下一步需要双卡时，应先切换到双卡资源后再继续。

## 当前验证结果

验证时间：2026-05-18

验证环境：

- 服务器：`atc`
- GPU：单卡 RTX 4090，`CUDA_VISIBLE_DEVICES=0`
- 模型：`/root/atc_vllm_sched/models/Qwen2.5-7B-Instruct`
- vLLM：`/root/atc_vllm_sched/vllm_src`，`pip install -e` 开发环境
- 数据集：`/root/atc_vllm_sched/data/qasper/qasper-dev-v0.3.json`
- Qasper 参数：`limit=10`，`max_concurrency=1`，`max_context_chars=6000`，`max_tokens=64`，`temperature=0`

静态检查：

- `py_compile` 通过。
- 活动 `vllm/` 和 `csrc/` 中不再残留 `atc_kivi_int4`、`atc_fp8_e4m3`、`atc_kivi_int4_cuda`、`reshape_and_cache_atc`、packed INT4/KIVI CUDA extension 标记。
- `scripts/test_fake_quant_methods.py` 已在 CUDA 上通过，`none/kivi/kvtuner/kvquant/pmkvq/mixkvq` 都能对 `[17, 4, 32]` KV tensor 输出同 shape、同 dtype 的 fake quant 结果。

Qasper 小样本结果：

| 方法 | ok/total | avg token F1 | avg latency(s) |
| --- | ---: | ---: | ---: |
| none baseline | 10/10 | 0.0988 | 0.8416 |
| kivi | 10/10 | 0.0723 | 1.0721 |
| kvtuner | 10/10 | 0.0280 | 1.4253 |
| kvquant | 10/10 | 0.0459 | 1.3345 |
| pmkvq | 10/10 | 0.0738 | 1.3639 |
| mixkvq | 10/10 | 0.0718 | 2.0038 |

这些数字只说明框架已经能在 vLLM serving 路径中切换方法并生成质量结果；样本数很小，不能作为论文结论。

serving workload smoke test：

| 方法 | workload | num prompts | max concurrency | ok/total | throughput(req/s) | avg latency(s) | avg queue wait(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pmkvq | burst | 4 | 2 | 4/4 | 0.7853 | 2.4351 | 1.3201 |
| pmkvq | time-series | 4 | 2 | 4/4 | 0.7084 | 2.3501 | 0.1807 |

实现注意：

- `mixkvq` 的 Python 动态 head 选择逻辑不能被 CUDA graph capture 捕获，因此 fake quant 启动脚本默认使用 `--enforce-eager`。
- 当前五种方法是 serving fake quant 的第一版可运行抽象，尚未逐行对齐本地桌面 PDF 中每篇论文的所有细节；后续正式实验前应再根据论文和开源代码细化默认配置。
- 当前阶段没有真实低精度存储，也不会降低 `nvidia-smi` 显存占用；这是符合本阶段目标的。

## 2026-05-21 Related-Work Baseline Fix

当前硬件已切换为双 NVIDIA A100-SXM4-40GB。正式 related-work 结果目录：

```text
/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8
```

关键修正：

- asymmetric fake quant 改为 KIVI/KVQuant 官方 min-offset 语义，修复 all-positive/all-negative group 被 zero-point clamp 压坏的问题。
- KVQuant 尽量走官方 `kvquant` simquant/outlier 函数；如果官方 calibration artifact 不存在，则仅标为 fake-quant approximation。
- `reshape_and_cache_flash` 也补上 fake quant hook；仍不修改 CUDA/Triton/PagedAttention kernel。
- 正式评测只使用 LongBench `qasper` test 前 200 条、chat endpoint、burst concurrency 8、`max_gen=128`。

修正后 c8 结果摘要：

| method | LongBench F1 | output trunc | avg_kv_bits | note |
| --- | ---: | ---: | ---: | --- |
| none | 44.45 | 0.000 | BF16 | baseline |
| kivi K2V2 | 41.74 | 0.005 | 2.00 | K2V2/G32/R128; old summary before residual-aware stats |
| kivi K4V4 | 44.91 | 0.000 | 4.38 | K4V4/G32/R128; residual-aware stats |
| kvtuner | 44.53 | 0.000 | 4.07 | official Qwen2.5 preset; effective-bit rerun |
| kvquant | 43.31 | 0.000 | 3.00 | official quantizer path, no Fisher artifact |
| pmkvq | 44.09 | 0.000 | 8.00 | high-bit serving fake approximation |
| mixkvq | 43.68 | 0.000 | 2.73 | paper-derived C2.7 approximation |

Calibration caveats:

- KVQuant Fisher probe failed at Wikitext2 download timeout; no local Wikitext2 cache exists.
- PM-KVQ official wrapper now gets through sensitivity, allocation, max-key, and searched rep-scale generation on Qasper-train surrogate artifacts. It does this by runtime shimming Qwen2/Llama attention compatibility in `scripts/run_pmkvq_official_calibration.py`, not by editing the official PM-KVQ repo.
- PM-KVQ RedPajama official calibration remains blocked because `togethercomputer/RedPajama-Data-1T-Sample` cannot be accessed from atc and no local cache exists.
- PM-KVQ Qasper-train surrogate low-bit diagnostics remain unusable as formal: all-4 bulk 40条 c8 F1 `9.45`、trunc `7/40`、avg KV bits `4.00`; 4/2 mixed 40条 c8 F1 `7.03`、trunc `9/40`、avg KV bits `3.22`。
- KVTuner's 44.71 vs baseline 44.45 was rechecked against the formal JSON configs: same data/order/chat endpoint/`max_gen=128`/`temperature=0.0`/evaluator, and all baseline/KVTuner outputs stopped normally. Treat this as a small tie-level variation on the 200-example Qasper slice, not as a serving/evaluator mismatch.

The corrected KIVI/KIVI-4/KVTuner/KVQuant/MixKVQ rows are usable as current serving fake-quant related-work baselines with the documented approximations. PM-KVQ should be cited as a conservative high-bit serving fake approximation plus failed low-bit diagnostics until official RedPajama artifacts are generated and the cache-wide progressive shrink semantics can be represented more faithfully.

## 2026-05-26 Multitask Formal Extension

New multitask formal scripts:

- `scripts/prepare_multitask_eval_manifests.py`
- `scripts/run_multitask_serving_eval.py`
- `scripts/summarize_multitask_results.py`
- `scripts/run_formal_multitask_pipeline.sh`

Manifest directory:

```text
/root/atc_vllm_sched/data/eval_manifests/20260526_multitask_200/
```

Result directory:

```text
/root/atc_vllm_sched/results/fake_quant/formal/multitask_longbench_math_200_chat_c8_20260526/
```

Formal matrix:

- Datasets: LongBench `qasper`, `narrativeqa`, `hotpotqa`, `passage_retrieval_en`, `passage_count`, `qmsum`; plus `HuggingFaceH4/MATH-500` and `openai/gsm8k`, first 200 examples each.
- Methods: `baseline`, `kivi2`, `kivi4`, `kvtuner_pertoken_c4_00`, `kvtuner_kivi_c3_92`, `kvquant`, `pmkvq_cachewide_mem132`. `MixKVQ` is no longer in the default formal matrix because it has no official repo and its threshold search remains a paper-derived diagnostics path only.
- Serving: same dual-instance vLLM + round-robin router, chat endpoint, burst c8, temperature 0.0.
- LongBench prompt/max_gen: KIVI/LongBench official `dataset2prompt.json` and `dataset2maxlen.json`; token middle truncation with `max_input_tokens=7500`.
- Math extension: MATH500 uses boxed-answer CoT prompt with `max_gen=1024`; GSM8K uses KVTuner local 8-shot CoT prompt and `####` numeric extraction with `max_gen=512`.

Full table: `summary.md`, `summary.csv`, `summary.json` in the result directory. All 56 formal JSON files were generated, with failed count `0` for every method/dataset pair.

Important interpretation notes:

- Baseline itself has nonzero output truncation on some tasks, e.g. LongBench retrieval tasks with official `max_gen=32` and MATH500 with `max_gen=1024`; compare quantized methods against the same-task baseline instead of treating truncation as automatically quantization-induced.
- KVQuant and PM-KVQ cache-wide are much slower than baseline on long-generation math tasks because this is Python-level fake quant/rewrite, not packed low-bit kernel execution.
- Effective bit statistics are residual/protected-token aware. Short prompts in MATH500/GSM8K keep a large fraction of tokens in residual/full-precision windows, so effective bits can be much higher than nominal bulk bits.

## 2026-05-27 Cache Semantics Repair

This repair was made after re-checking the local PDFs and official repos, not as a score-tuning change. The affected rows in `multitask_longbench_math_200_chat_c8_20260526` should be treated as pre-repair diagnostics until GPU is available for rerun.

- KIVI: local `KIVI.pdf` Algorithm 1 and repo `models/llama_kivi.py` show different residual semantics for K and V. Key residual is the unflushed `seq_len % R` group; Value residual is the most recent `R` tokens. The cache-wide fake path now records/rewrite targets as `K{bit}V16`, `K16V16`, etc. instead of assuming both K and V always keep the last `R` tokens in FP16.
- KVTuner: local `KVtunner.pdf` Table 11 lists Qwen2.5-7B-Instruct `per-token-asym` C4.0 and KIVI-mode C3.92/C5.96 presets. The default formal pipeline now uses the per-token-asym C4.0 preset with `group_size=-1` and residual `0`; KVTuner repo `vanilla_quantizer.py` resolves `q_group_size=-1` to the full last dimension. The adapter still supports KIVI-mode presets, where repo README uses `group_size=32`, `residual_length=32`.
- KVQuant: local `kvquant.pdf` Section 3.5 says only the first token is kept in FP16 for attention-sink-aware quantization, while the repo README gives `eg. 5` as an implementation option. The project default and formal scripts are now `ATC_KVQUANT_FIRST_TOKENS_FP16=1`, applied to global sequence position only.
- MixKVQ: local `MixKVQ.pdf` states `G=32`, `R=128`; Appendix D.1 describes lazy residual updates where new KV is kept in a full-precision residual buffer and quantized only when the buffer reaches `R`; Appendix E fixes sink length at `32` in the ablation setup. The adapter now uses `ATC_MIXKVQ_SINK_TOKENS=32` and lazy-buffer modulo residual semantics instead of a permanent sliding "last 128 tokens" FP16 window.

No CUDA/Triton/PagedAttention kernel behavior changed. The repair only changes Python fake quant target selection, in-place cache rewrite, and trace/effective-bit accounting.

## 2026-05-27 KVQuant/MixKVQ Artifact Alignment

- KVQuant now has an env-gated Qwen2 pre-RoPE key fake-quant hook (`ATC_KVQUANT_PREROPE=1`): Qwen2 applies K fake quant between `qkv_proj` split and `rotary_emb`, then the cache-write hook skips K and handles V/trace, avoiding double quantization.
- New `scripts/run_kvquant_nuq_artifact.py` builds the full runtime NUQ artifact with official KVQuant `SimQuant.quantize(... include_sparse=True, sparsity_threshold=0.99, nuq=True, first_few_fp16=1)`. The old `fisher_summary.pt` remains a diagnostic artifact, not a complete KVQuant artifact.
- KVQuant repaired formal should set `ATC_KVQUANT_STRICT_ARTIFACT=1`, `ATC_KVQUANT_NUQ_ARTIFACT_PATH=.../nuq_artifact.pt`, and expect traces with `kvquant_mode=prerope_nuq` and `pre_rope_applied=true`.
- MixKVQ now supports strict threshold artifacts via `ATC_MIXKVQ_THRESHOLDS_PATH` and `ATC_MIXKVQ_STRICT_THRESHOLDS=1`; strict formal fails fast without a threshold-search artifact instead of silently using the old C2.7 ratio approximation.
- New `scripts/search_mixkvq_thresholds.py` writes `thresholds.json` with `tau_BF16`, `tau_UINT4`, trial history, Pareto frontier, seed, dataset/split, and target bit. It uses Optuna TPE if available and a deterministic fallback for dry-run/static validation.
- MixKVQ threshold artifact path remains diagnostics-only and is no longer part of the default formal matrix.

## 2026-05-28 Six-Variant Ready State

The default formal matrix is now prepared as six quantized variants plus baseline:

- `kivi2`: KIVI K2V2, `G=32`, `R=128`.
- `kivi4`: KIVI K4V4, `G=32`, `R=128`.
- `kvtuner_pertoken_c4_00`: KVTuner Qwen2.5 per-token-asym C4.00 preset, `group_size=-1`, `residual_length=0`.
- `kvtuner_kivi_c3_92`: KVTuner Qwen2.5 KIVI-mode C3.92 preset, `group_size=32`, `residual_length=32`.
- `kvquant`: KVQuant nuq3-1% strict path. Full Wikitext-2 16x2048 NUQ artifact is generated at `/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048_nuq/nuq_artifact.pt` with sha256 `636ff8f76e063009d328256ee74b808bf92b0eef46802f1982c4ae398a0e0d94`; calibration JSONL sha256 is `f20f63e4f8e20fb92a4af7fba003930d6b876a5da075e4bc2d7c599ebfd4baec`. Runtime uses pre-RoPE K fake quant and global first-token FP16.
- `pmkvq_cachewide_mem132`: PM-KVQ RedPajama arXiv n512 artifacts, `budget_fbit_8_4_mem132.pt`, `rep_scales_k4v4.pt`, cache-wide rewrite.

`scripts/run_formal_multitask_pipeline.sh` and `scripts/run_formal_longbench_qasper_pipeline.sh` both use explicit KVTuner variants. The old `kvtuner` alias remains a per-token C4.00 compatibility alias, but new reports should use the explicit variant names. KVQuant adapter now passes `sparsity_threshold=0.99` directly to the official `get_outliers_dynamic`, matching the repo's 1% dense-and-sparse semantics instead of applying an extra tail conversion.
