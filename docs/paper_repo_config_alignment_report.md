# Paper/Repo Config Alignment Report

本报告是正式实验的配置来源说明。`results/fake_quant/sensitivity/` 和早先
`results/fake_quant/formal/` 中基于 Qasper dev、字符截断、手写 prompt、
`max_tokens=64/128`、`max_context_chars=6000/12000` 的结果，全部标记为
**pre-alignment exploratory**，不进入正式 related-work 对比。

## Shared Formal Serving/Eval Setup

- 模型：`/root/atc_vllm_sched/models/Qwen2.5-7B-Instruct`。
- 正式数据集：`THUDM/LongBench`，config `qasper`，split `test`，官方顺序前 200 条。
- Prompt：`references/kv_methods/KIVI/config/dataset2prompt.json` 中 `qasper` 模板，KVTuner repo 使用同一模板。
- Generation length：`references/kv_methods/KIVI/config/dataset2maxlen.json` 中 `qasper=128`。
- Context truncation：KIVI/KVTuner LongBench 脚本中的 token-based middle truncation。
- Decoding：官方脚本为 greedy；Qwen2.5-Instruct 正式 serving evaluator 使用 OpenAI chat endpoint，`temperature=0.0`。
- Serving workload：burst、max concurrency 8、router round-robin 是本项目的 `framework serving extension / not specified in paper`；concurrency 16 不再作为当前正式目标。
- Fake quant hook：`vllm/_custom_ops.py` 中 `reshape_and_cache` 和 `reshape_and_cache_flash` 调用前做 quantize-dequantize，正常 FP16/BF16 KV cache 写入；PM-KVQ 新增 `pmkvq_cachewide` 写入后 Python in-place rewrite 历史 FP16/BF16 cache slots。两者都不改 CUDA/Triton/PagedAttention kernel，不做 packed KV cache。
- 2026-05-21 修正：公共 asymmetric fake quant 已改为 KIVI/KVQuant min-offset 公式，避免旧 zero-point clamp 导致正/负 group 坍缩。
- 2026-05-21 统计修正：KIVI/KVTuner trace 增加 residual-aware effective precision summary，bulk token 按配置 bit 计，recent residual window 按 16-bit 计；同时保留 `nominal_k_bits` / `nominal_v_bits` 字段。
- 2026-05-27 cache 语义修正：重新核对本地 PDF 和官方 repo 后，KIVI/KVTuner/MixKVQ 的 residual/sink/window 不再按单一 write chunk 或统一 "last R KV FP16" 近似。KIVI/KVTuner-KIVI mode 的 K residual 按未 flush 的 `seq_len % R` group 计算，V residual 按最近 `R` 计算；MixKVQ 使用 lazy residual buffer modulo `R`，并显式记录 sink length。KVQuant attention sink 默认改为论文 Section 3.5 的 first token `1`，不是 repo README 示例中的 `eg. 5`。2026-05-26 multitask formal 表中的受影响方法需在 GPU 恢复后重跑。

## KIVI

- PDF：`C:\Users\16867\Desktop\KIVI.pdf`。
- Repo：`/root/atc_vllm_sched/references/kv_methods/KIVI`，origin `https://github.com/jy-yuan/KIVI.git`，commit `876b4d2`。
- 论文方法：tuning-free asymmetric 2-bit KV cache；Key cache per-channel，Value cache per-token；近期 residual cache 保持全精度。
- 论文/repo 参数：主配置 K/V 2-bit；LongBench 相关实现使用官方 prompt/max_gen；论文中 LongBench 公平设置使用 group size `G=32`、residual length `R=128`。KIVI repo 的 `docs/long_bench.md` 对 vanilla MHA 推荐 KIVI-2，对 MQA/GQA 模型推荐 KIVI-4 作为更稳配置；Qwen2.5-7B-Instruct 是 GQA/MQA-style KV-head sharing 模型。
- 当前 fake quant 映射：K 用 token group 内 per-channel asymmetric fake quant；V 用 per-token/channel-group asymmetric fake quant；主配置保留 `K2V2/G32/R128`，另补跑 `K4V4/G32/R128`。2026-05-27 后 cache-wide fake path 按论文 Algorithm 1/官方实现区分 K/V residual：K 只保留未凑满 `R` 的 `seq_len % R` group 为 FP16，V 保留最近 `R` 个 token 为 FP16。
- 2026-05-21 KIVI-4 补跑：`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8_kivi4/`，Qasper-200 c8 F1 `44.91`，output trunc `0/200`，residual-aware avg KV bits `4.38`，distribution `4:0.97,16:0.03`。KIVI-2 仍保留为更激进压缩点，F1 `41.74`，output trunc `1/200`。
- 跳过内容：真实 2-bit 存储、CUDA kernel、packing、真实显存下降。

## KVTuner

- PDF：`C:\Users\16867\Desktop\KVtunner.pdf`。
- Repo：`/root/atc_vllm_sched/references/kv_methods/KVTuner`，origin `https://github.com/cmd2001/KVTuner.git`，commit `96dd05e`。
- 论文方法：离线搜索 layer-wise K/V precision pairs；不是 runtime serving pressure 动态策略。
- Repo 参数：`calibration_presets/Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml` 对应本地 `KVtunner.pdf` Table 11 中 Qwen2.5-7B-Instruct per-token-asym C4.0 配置；`Qwen2.5-7B-Instruct_kivi_KVTuner4_0.yaml` 对应 KIVI-mode 低 bit 对照。README 示例 KIVI scheme 使用 `axis_key=1`、`axis_value=0`、`group_size=32`、`residual_length=32`；per-token quantization 使用 `residual_length=0`，官方 `vanilla_quantizer.py` 将 `q_group_size=-1` 解析为最后一维整维度分组。
- 当前 fake quant 映射：正式矩阵保留两个显式 KVTuner variants。`kvtuner_pertoken_c4_00` 读取 `Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml`，对应 Table 11 的 per-token-asym C4.00：`group_size=-1`、`residual_length=0`。`kvtuner_kivi_c3_92` 读取 `Qwen2.5-7B-Instruct_kivi_KVTuner4_0.yaml`，对应 Table 11 的 KIVI-mode C3.92：`group_size=32`、`residual_length=32`。二者都只使用官方 preset 中的 layer-wise K/V bit，不读取 queue length 或 serving pressure。
- Calibration 状态：未重新运行 KVTuner calibration/search；正式文档中标注为使用官方 preset。
- 2026-05-21 复查：正式 `none/kvtuner/pmkvq` JSON 均为 chat endpoint、`temperature=0.0`、`max_gen=128`、同一 Qasper test 前 200 条、同一 evaluator。主目录 KVTuner F1 `44.71` 高于 baseline `44.45` 的幅度较小；本地 `KVtunner.pdf` 的 LongBench 表中也存在 mixed precision 与 BF16 接近或略高的设置，因此当前解释为 200 条 Qasper slice 上的 tie/小幅量化扰动，不视为评测配置错误。
- 2026-05-21 effective-bit 补跑：`/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260521_c8_kvtuner_effectivebits/`，F1 `44.53`，output trunc `0/200`，residual-aware avg KV bits `4.07`，distribution `2:0.53,4:0.20,8:0.26,16:0.01`。
- 跳过内容：官方 cache class、真实低精度存储、重新校准。

## KVQuant

- PDF：`C:\Users\16867\Desktop\kvquant.pdf`。
- Repo：`/root/atc_vllm_sched/references/kv_methods/KVQuant`，origin `https://github.com/SqueezeAILab/KVQuant.git`，commit `57a2383`。
- 论文方法：Pre-RoPE Key per-channel quantization、non-uniform quantization、Fisher/sensitivity calibration、dense-and-sparse outlier handling、attention sink token 保留。
- 论文/repo 参数：论文中校准使用 Wikitext-2 train 的 16 个 2K 长度样本；repo 支持 `bits`、`nuq`、`include_sparse`、`first_few_fp16`、outlier/capping 相关配置；LongBench 常见对比包含 KV3/1% outlier。
- 当前 fake quant 映射：默认 `bits=3`、`outlier_ratio=0.01`、`first_tokens_fp16=1`。2026-05-27 后新增 Qwen2 pre-RoPE K hook：`qkv_proj` split 后、`rotary_emb` 前调用 KVQuant fake quant；cache-write hook 若看到 pre-RoPE marker 则跳过 K、只处理 V，避免二次量化。无完整 NUQ artifact 时仍可显式跑 post-RoPE diagnostics，但 strict formal 要求 `kvquant_mode=prerope_nuq`。
- Calibration 状态：Wikitext-2 已准备为 `/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048/`，来自 `Salesforce/wikitext` 的 `wikitext-2-raw-v1`。`scripts/run_kvquant_official_calibration.py` 生成的 `fisher_summary.pt` 只保留为历史诊断；`scripts/run_kvquant_nuq_artifact.py` 已复用官方 `SimQuant.quantize(... include_sparse=True, sparsity_threshold=0.99, nuq=True, first_few_fp16=1)` 生成完整 artifact：`/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048_nuq/nuq_artifact.pt`。metadata 记录 `num_examples=16`、`seq_len=2048`、`bits=3`、`sparsity_threshold=0.99`、`first_few_fp16=1`、calib JSONL sha256 `f20f63e4f8e20fb92a4af7fba003930d6b876a5da075e4bc2d7c599ebfd4baec`、artifact sha256 `636ff8f76e063009d328256ee74b808bf92b0eef46802f1982c4ae398a0e0d94`；runtime loader 校验为 `28` 层，每层 K/V 均有 LUT/signposts 和 outlier thresholds。
- 2026-05-22 adapter 修复：KVQuant official quantizer import 现在优先尝试 repo `benchmarking/kvquant`、`deployment/kvquant`、`quant/kvquant`，trace 会标注 `official:<subdir>` 或 fallback 原因；新增 `ATC_KVQUANT_ARTIFACT_PATH` 记录 Fisher artifact metadata。
- 2026-05-21 复查：sequence-aware `first_tokens_fp16` 后，最新 Qasper-200 c8 KVQuant output trunc 为 `0/200`，F1 `43.31`。这个分数不能解读为完整 KVQuant paper baseline，因为之前没有 Fisher artifact；2026-05-22 后的下一轮 diagnostics/formal 应使用 `ATC_KVQUANT_ARTIFACT_PATH=/root/atc_vllm_sched/artifacts/kvquant/wikitext2_qwen25_n16_l2048_official`。2026-05-27 后 first-token scope 明确为全序列位置 `<1`，decode 第 100 步不会再被误当成 sink token。
- 2026-05-28 修正：adapter 直接把 `sparsity_threshold=0.99` 传入官方 `get_outliers_dynamic`。官方函数内部会做双尾阈值换算；项目不再额外换算，避免把论文/repo 的 1% outlier 语义变成更小比例。
- 重要 approximation：即使完整 NUQ artifact 可用，当前仍是 fake quantize/dequantize；真实 dense/sparse storage、CSR/CSC sparse outlier cache 和 CUDA kernels 不复现。
- 跳过内容：真实 dense/sparse storage、CUDA kernel、packed cache。

## PM-KVQ

- PDF：`C:\Users\16867\Desktop\PM-KVQ.pdf`。
- Repo：`/root/atc_vllm_sched/references/kv_methods/PM-KVQ`，origin `https://github.com/thu-nics/PM-KVQ.git`，commit `c8a2bff`。
- 论文方法：progressive mixed-precision KV cache；通过 calibration allocation 和 memory budget 逐步降低 bulk KV bit-width；不是 serving queue pressure heuristic。
- Repo 参数：`n_sink_token=1`、`n_sink_token_bits=16`、`n_window_token=128`、`n_window_token_bits=16`、`n_init_kv_bits=16`；progressive cache 中 bulk bits 按 budget 16->8->4->2 shrink；group-wise asymmetric quantizer `group_size=128`。
- 当前 fake quant 映射：按 sink/window 全精度、bulk 根据官方 progressive budget 语义 shrink；若提供 `ATC_PMKVQ_BUDGET_PATH`，按官方 `allocate_memory.py` 产生的 per-layer MB budget 选择 bulk bit；若提供 `ATC_PMKVQ_REP_SCALES_PATH`，在 K fake quant 前应用 SmoothAttention-style key scale inverse/restore。最新低比特 formal 使用 RedPajama arXiv n512 artifact 上官方 allocation 生成的 `budget_fbit_8_4_mem132.pt`，layer bulk bits 为 `[8, 4 x 23, 8 x 4]`，平均 bulk bit `4.71`；不是旧 Qasper-train surrogate all-8 approximation。
- Calibration 状态：新增 `scripts/run_pmkvq_official_calibration.py`，优先调用官方 repo 源码接口 `get_kv_sensitivity`、`allocate_memory_budget`、`get_max_keys`、`search_rep_scales`。wrapper 只做当前 `transformers==4.47.1` / Qwen2.5 兼容 shim：补 `ALL_ATTENTION_FUNCTIONS`、`eager_attention_forward`、`self.scaling`、Qwen2 attention 3-return、legacy tuple `past_key_values`。本地 `PM-KVQ.pdf` 指明 calibration 使用 RedPajama arXiv subset、512 samples、2048 tokens、positional interpolation 到 8192；由于 `Data-1T-Sample` 不可访问，当前使用 `togethercomputer/RedPajama-Data-1T` 的 `arxiv` streaming subset 作为最接近替代，artifact 目录为 `/root/atc_vllm_sched/artifacts/pmkvq/redpajama_arxiv_stream_n512_l2048_eff8192/`。
- 2026-05-22 PM cache-wide 修复：新增 `ATC_KV_FAKE_QUANT_METHOD=pmkvq_cachewide`。旧 `pmkvq` 只在 pre-write tensor 上做 fake quant；`pmkvq_cachewide` 在 vLLM 写入后读取 `block_tables`/`seq_lens`/`context_lens`，恢复当前序列历史 slots，并按官方 progressive 语义对 sink、modulo window 和 bulk region 计算 target bit，对未处理或 target bit 改变的旧 slots 做 in-place fake quant rewrite。trace 记录 `cache_wide_coverage`、`rewritten_slots`、`skipped_slots`、`slot_target_bit_distribution`。这解决 write-only hook 不能回写旧 KV 的主要语义缺口，但仍不是真实 packed low-bit cache。
- 2026-05-22 calibration 状态：RedPajama arXiv n512 artifact 已有 `sensitivity.pt`、`budget_fbit_4_2.pt`、`max_keys.pt`、`rep_scales_k4v4.pt` 和 `calibration_manifest.json`。manifest 确认 512 条来自 `togethercomputer/RedPajama-Data-1T` 的 `arxiv` streaming subset，每条 tokenized 长度 2048，`effective_len=8192`，JSONL sha256 `a97b062b5e83d54bce99b3cc12771cee856a87d64c459e338c0992a77b267a68`。与官方 repo 的差异是：`Data-1T-Sample` 仍不可访问，当前使用 deterministic stream first-512，而不是官方 seed=42 shuffle；这记录为 data-source approximation。
- HF official path 诊断：在 wrapper 中修正官方 `apply_fake_pmkvq` 对 `apply_smoothattention_rep` 的位置参数误用，改为显式 `apply_smoothattention_rep(model, rep_scales=...)` 后再 `apply_progressive(...)`。即使如此，RedPajama 4/2-ish budget 在 HF official probe 上 20 条 Qasper 仍崩分，说明主要问题不是 vLLM adapter，而是该模型/任务/预算组合过激。
- 低比特 diagnostics：RedPajama official 4/2 或 all-4-ish budget 在 vLLM `pmkvq_cachewide` 40 条 c8 上 F1 约 `18.77` 且有 trunc；official `8,4` allocation 中 mem128 仍崩，mem132 成为当前最低稳定点。最终 formal 目录 `/root/atc_vllm_sched/results/fake_quant/formal/longbench_qasper_200_chat_fix_20260522_pmkvq_defer_current_mem132_fast_c8/`，Qasper-200 burst c8 F1 `44.08`，failed `0/200`，output trunc `0/200`，cache-wide coverage `1.0`，trace effective avg KV bits `6.27`，distribution `4:0.57,8:0.36,16:0.07`。由于 trace 统计计入 sink/window protected 16-bit tokens，effective avg 高于 layer bulk avg `4.71`。
- 2026-05-22 性能修复：`pmkvq_cachewide` 旧版用逐 slot Python gather/scatter，并在每层每步构造整段 GPU bit-map 后搬到 CPU 做 trace，导致 Qasper-200 需要约 35 分钟且偶发 router 600s timeout。已改为批量 tensor gather/scatter、公式计算 bit target 和 count summary；优化后同一 200 条 `total_elapsed_s=151.57`。只为 transient 502 补样本时可用 evaluator `--indices` 或 pipeline `EVAL_INDICES=19` 单样本重跑，不需要全量重跑。
- 重要 approximation：`pmkvq_cachewide` 通过 Python in-place rewrite FP16/BF16 cache 模拟 progressive shrinking，不能真实保存 packed low-bit KV，也不声明显存节省；若 cache layout 或 metadata 缺失导致 `cache_wide_coverage < 99%`，只能标 diagnostics/blocker。
- 跳过内容：packed cache、bit_width_shrinking 的真实整数重打包、真实显存节省。

## MixKVQ

- PDF：`C:\Users\16867\Desktop\MixKVQ.pdf`。
- Repo：无官方开源代码。
- 论文方法：query-aware mixed precision；Key channel salience `A_d = I_d * S_d`，其中 `I_d` 来自 query 重要性，`S_d=(max(k_d)-min(k_d))/(2^B-1)`；高 salience BF16，中 salience UINT4，低 salience UINT2；Value cache per-token quantization；full precision residual buffer；论文公平设置 `G=32`、`R=128`；Appendix D.1 使用 lazy update，buffer 长度达到 `R` 时才执行 channel selection/outlier extraction/bit packing 并 merge 到 main cache；Appendix E ablation 固定 sink length `32`。报告配置如 MixKVQ-C2.7/C2.3。
- 当前 fake quant 映射：用 attention query 和 key range 计算 salience；按 target average bit（默认 C2.7）在 salience 排序下分配 BF16/UINT4/UINT2；V 使用 per-token 2-bit fake quant。2026-05-27 后不再使用永久 sliding last-128 FP16 近似，而是使用 `sink=32` 加 lazy residual buffer `((seq_len - sink) % R)` 的 global-position-aware cache-wide rewrite。
- Calibration 状态：因无官方 repo，MixKVQ 不再进入默认 related-work formal 矩阵。`scripts/search_mixkvq_thresholds.py` 和 runtime threshold loader 保留作 diagnostics；旧无 artifact C2.7 ratio 分配不能作为 paper-aligned formal。
- 跳过内容：真实 UINT storage；默认 formal 不再跑 MixKVQ。

## 2026-05-26 Multitask Formal Extension

本节记录多数据集扩展的配置来源，不改变前述算法对齐状态。

- LongBench 任务：`qasper`, `narrativeqa`, `hotpotqa`, `passage_retrieval_en`, `passage_count`, `qmsum`，均来自 `THUDM/LongBench` test split，官方顺序前 200 条。
- LongBench prompt/max_gen：继续使用 `/root/atc_vllm_sched/references/kv_methods/KIVI/config/dataset2prompt.json` 和 `dataset2maxlen.json`；metric 映射沿用 KIVI/LongBench evaluator：QA F1、retrieval/count、ROUGE-L。
- Math extension：`HuggingFaceH4/MATH-500` test first 200，boxed answer exact match；`openai/gsm8k` `main` test first 200，KVTuner repo 的本地 8-shot CoT prompt 和 `####` numeric exact match。二者是本项目 serving extension，不是 LongBench 官方任务。
- Manifest 固化路径：`/root/atc_vllm_sched/data/eval_manifests/20260526_multitask_200/`，`manifest_summary.json` 记录每个 JSONL 的 dataset id、config、split、first-200 顺序和 sha256。
- Formal result：旧目录 `/root/atc_vllm_sched/results/fake_quant/formal/multitask_longbench_math_200_chat_c8_20260526/` 汇总的是 2026-05-26 的 56 个 method-dataset formal JSON。2026-05-28 后默认矩阵改为 baseline plus six variants：`kivi2`, `kivi4`, `kvtuner_pertoken_c4_00`, `kvtuner_kivi_c3_92`, `kvquant`, `pmkvq_cachewide_mem132`；本轮只准备方法与 calibration artifact，不重跑完整 multitask formal。
- Serving 设置：仍为本项目 framework extension，双 vLLM 实例、round-robin router、chat endpoint、burst c8、temperature 0.0；不修改 CUDA/Triton/PagedAttention，不声明真实显存节省或加速。
- 解释口径：短 prompt/math 任务中 residual/protected 16-bit token 占比高，因此 residual-aware effective bits 可能明显高于 nominal bulk bits；baseline 在部分官方 max_gen 较短任务上也会出现 output truncation。
