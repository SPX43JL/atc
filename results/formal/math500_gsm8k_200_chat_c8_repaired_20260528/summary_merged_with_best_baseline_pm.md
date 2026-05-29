# MATH500/GSM8K Repaired Formal Summary

| dataset | metric | method | score | ok/total | failed | trunc | avg latency(s) | p95 latency(s) | total elapsed(s) | throughput(req/s) | avg_k_bits | avg_v_bits | avg_kv_bits | precision_distribution |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| math500 | math_exact | baseline | 66.00 | 200/200 | 0 | 27 (13.50%) | 12.717 | 22.488 | 325.3 | 0.615 | BF16 | BF16 | BF16 | BF16/FP16 |
| math500 | math_exact | kivi2 | 67.50 | 200/200 | 0 | 20 (10.00%) | 57.276 | 114.967 | 1459.3 | 0.137 | 3.90 | 5.77 | 4.83 | 2:0.80, 16:0.20 |
| math500 | math_exact | kivi4 | 64.50 | 200/200 | 0 | 27 (13.50%) | 58.404 | 121.634 | 1483.9 | 0.135 | 5.61 | 7.19 | 6.40 | 4:0.80, 16:0.20 |
| math500 | math_exact | kvtuner_pertoken_c4_00 | 63.00 | 200/200 | 0 | 18 (9.00%) | 49.949 | 96.948 | 1264.6 | 0.158 | 5.14 | 3.43 | 4.29 | 2:0.14, 4:0.71, 8:0.14 |
| math500 | math_exact | kvtuner_kivi_c3_92 | 67.00 | 200/200 | 0 | 18 (9.00%) | 46.926 | 88.736 | 1199.4 | 0.167 | 4.40 | 4.83 | 4.62 | 2:0.54, 4:0.14, 8:0.27, 16:0.05 |
| math500 | math_exact | kvquant | 67.50 | 200/200 | 0 | 21 (10.50%) | 96.759 | 191.501 | 2459.4 | 0.081 | 3.02 | 3.02 | 3.02 | 3:1.00, 16:0.00 |
| math500 | math_exact | pmkvq_cachewide_mem132 | 64.50 | 200/200 | 0 | 29 (14.50%) | 76.869 | 156.510 | 1947.8 | 0.103 | 16.00 | 16.00 | 16.00 | 16:1.00 |
| gsm8k | gsm8k_exact | baseline | 85.50 | 200/200 | 0 | 0 (0.00%) | 4.527 | 8.545 | 116.7 | 1.714 | BF16 | BF16 | BF16 | BF16/FP16 |
| gsm8k | gsm8k_exact | kivi2 | 84.00 | 200/200 | 0 | 1 (0.50%) | 19.833 | 36.329 | 503.0 | 0.398 | 2.96 | 3.99 | 3.48 | 2:0.89, 16:0.11 |
| gsm8k | gsm8k_exact | kivi4 | 83.50 | 200/200 | 0 | 0 (0.00%) | 19.318 | 37.409 | 489.3 | 0.409 | 4.82 | 5.72 | 5.27 | 4:0.89, 16:0.11 |
| gsm8k | gsm8k_exact | kvtuner_pertoken_c4_00 | 82.00 | 200/200 | 0 | 0 (0.00%) | 16.172 | 33.093 | 411.3 | 0.486 | 5.14 | 3.43 | 4.29 | 2:0.14, 4:0.71, 8:0.14 |
| gsm8k | gsm8k_exact | kvtuner_kivi_c3_92 | 83.00 | 200/200 | 0 | 0 (0.00%) | 16.329 | 32.412 | 414.6 | 0.482 | 4.21 | 4.43 | 4.32 | 2:0.56, 4:0.14, 8:0.28, 16:0.03 |
| gsm8k | gsm8k_exact | kvquant | 84.00 | 200/200 | 0 | 1 (0.50%) | 31.773 | 64.281 | 806.9 | 0.248 | 3.01 | 3.01 | 3.01 | 3:1.00, 16:0.00 |
| gsm8k | gsm8k_exact | pmkvq_cachewide_mem132 | 85.00 | 200/200 | 0 | 0 (0.00%) | 24.619 | 46.716 | 623.9 | 0.321 | 16.00 | 16.00 | 16.00 | 16:1.00 |

Notes:
- Baseline is selected per dataset from available baseline runs by higher score.
- PM-KVQ mem132 is reused from the unchanged 20260526 formal run.
- All quantized rows except PM come from the repaired 20260528 rerun.
