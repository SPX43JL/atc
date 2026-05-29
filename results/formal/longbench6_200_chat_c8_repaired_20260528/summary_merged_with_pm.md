# LongBench6 repaired formal summary

Baseline is freshly rerun in this directory. PM-KVQ mem132 rows are reused from 20260526 because PM method/artifacts were not changed in this rerun.

| dataset | method | score | failed | trunc | total(s) | avg KV bits | precision dist | source |
|---|---|---:|---:|---:|---:|---:|---|---|
| qasper | baseline | 44.49 | 0 | 0 | 45.8 | BF16 | BF16/FP16 | new_20260528 |
| qasper | kivi2 | 42.07 | 0 | 0 | 160.7 | 2.28 | 2:0.98, 16:0.02 | new_20260528 |
| qasper | kivi4 | 44.91 | 0 | 0 | 165.3 |  |  | new_20260528 |
| qasper | kvtuner_pertoken_c4_00 | 43.76 | 0 | 0 | 151.6 | 4.29 | 2:0.14, 4:0.71, 8:0.14 | new_20260528 |
| qasper | kvtuner_kivi_c3_92 | 44.13 | 0 | 0 | 165.3 |  |  | new_20260528 |
| qasper | kvquant | 41.48 | 0 | 0 | 115.7 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| qasper | pmkvq_cachewide_mem132 | 44.13 | 0 | 0 | 142.2 | 6.28 | 4:0.57, 8:0.36, 16:0.07 | reused_20260526_pm |
| narrativeqa | baseline | 23.72 | 0 | 0 | 62.8 | BF16 | BF16/FP16 | new_20260528 |
| narrativeqa | kivi2 | 24.10 | 0 | 0 | 213.7 | 2.22 | 2:0.98, 16:0.02 | new_20260528 |
| narrativeqa | kivi4 | 23.68 | 0 | 0 | 212.3 | 4.19 | 4:0.98, 16:0.02 | new_20260528 |
| narrativeqa | kvtuner_pertoken_c4_00 | 24.47 | 0 | 0 | 200.9 | 4.28 | 2:0.14, 4:0.72, 8:0.14 | new_20260528 |
| narrativeqa | kvtuner_kivi_c3_92 | 24.14 | 0 | 0 | 219.2 |  |  | new_20260528 |
| narrativeqa | kvquant | 25.16 | 0 | 0 | 117.5 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| narrativeqa | pmkvq_cachewide_mem132 | 24.28 | 0 | 0 | 171.5 | 4.88 | 4:0.81, 8:0.18, 16:0.01 | reused_20260526_pm |
| hotpotqa | baseline | 49.42 | 0 | 15 | 62.4 | BF16 | BF16/FP16 | new_20260528 |
| hotpotqa | kivi2 | 49.80 | 0 | 14 | 206.8 | 2.21 | 2:0.98, 16:0.02 | new_20260528 |
| hotpotqa | kivi4 | 49.11 | 0 | 13 | 212.3 |  |  | new_20260528 |
| hotpotqa | kvtuner_pertoken_c4_00 | 48.57 | 0 | 15 | 194.6 | 4.28 | 2:0.14, 4:0.72, 8:0.14 | new_20260528 |
| hotpotqa | kvtuner_kivi_c3_92 | 48.48 | 0 | 14 | 210.7 |  |  | new_20260528 |
| hotpotqa | kvquant | 48.72 | 0 | 13 | 116.5 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| hotpotqa | pmkvq_cachewide_mem132 | 49.32 | 0 | 14 | 168.7 | 4.89 | 4:0.81, 8:0.18, 16:0.01 | reused_20260526_pm |
| passage_retrieval_en | baseline | 67.50 | 0 | 34 | 67.1 | BF16 | BF16/FP16 | new_20260528 |
| passage_retrieval_en | kivi2 | 65.00 | 0 | 31 | 222.6 | 2.21 | 2:0.99, 16:0.01 | new_20260528 |
| passage_retrieval_en | kivi4 | 67.50 | 0 | 34 | 230.2 |  |  | new_20260528 |
| passage_retrieval_en | kvtuner_pertoken_c4_00 | 65.50 | 0 | 34 | 210.8 | 4.28 | 2:0.14, 4:0.72, 8:0.14 | new_20260528 |
| passage_retrieval_en | kvtuner_kivi_c3_92 | 66.00 | 0 | 35 | 225.8 | 4.00 | 2:0.57, 4:0.15, 8:0.28, 16:0.00 | new_20260528 |
| passage_retrieval_en | kvquant | 64.00 | 0 | 33 | 129.4 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| passage_retrieval_en | pmkvq_cachewide_mem132 | 66.50 | 0 | 32 | 182.0 | 4.86 | 4:0.81, 8:0.18, 16:0.01 | reused_20260526_pm |
| passage_count | baseline | 6.50 | 0 | 0 | 63.1 | BF16 | BF16/FP16 | new_20260528 |
| passage_count | kivi2 | 6.50 | 0 | 0 | 168.2 | 2.22 | 2:0.98, 16:0.02 | new_20260528 |
| passage_count | kivi4 | 6.50 | 0 | 0 | 173.1 |  |  | new_20260528 |
| passage_count | kvtuner_pertoken_c4_00 | 6.50 | 0 | 0 | 153.4 | 4.28 | 2:0.14, 4:0.72, 8:0.14 | new_20260528 |
| passage_count | kvtuner_kivi_c3_92 | 6.50 | 0 | 0 | 180.4 |  |  | new_20260528 |
| passage_count | kvquant | 5.00 | 0 | 0 | 95.0 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| passage_count | pmkvq_cachewide_mem132 | 6.50 | 0 | 0 | 136.2 | 4.87 | 4:0.81, 8:0.18, 16:0.01 | reused_20260526_pm |
| qmsum | baseline | 17.73 | 0 | 0 | 106.2 | BF16 | BF16/FP16 | new_20260528 |
| qmsum | kivi2 | 17.24 | 0 | 0 | 466.3 | 2.18 | 2:0.99, 16:0.01 | new_20260528 |
| qmsum | kivi4 | 17.91 | 0 | 0 | 431.2 |  |  | new_20260528 |
| qmsum | kvtuner_pertoken_c4_00 | 17.25 | 0 | 0 | 395.2 | 4.29 | 2:0.14, 4:0.71, 8:0.14 | new_20260528 |
| qmsum | kvtuner_kivi_c3_92 | 17.59 | 0 | 0 | 425.3 |  |  | new_20260528 |
| qmsum | kvquant | 17.43 | 0 | 0 | 414.4 | 3.00 | 3:1.00, 16:0.00 | new_20260528 |
| qmsum | pmkvq_cachewide_mem132 | 17.65 | 0 | 0 | 457.9 | 4.81 | 4:0.81, 8:0.18, 16:0.01 | reused_20260526_pm |
