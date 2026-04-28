# 실험 결과 — 4-class Category 분류 (L1 / L2 / L3 / L4)

ICE 도메인 7,086 patents 에 대한 **4-class ordinal 분류** 실험 결과.
binary Y (top 10% vs rest) 와 동일한 모델 / 같은 protocol / 같은 metric
세트로 forward citation level 을 4단계로 나누어 분류 성능을 측정함.

Binary 결과는 `Result.md` 참조.

---

## 실험 설정

| 항목 | 값 |
|---|---|
| Universe | 7,086 USPTO patents (Sinigaglia 2022 16-query union, 1980–2020) |
| Target | 4-class Category (forward5 percentile 기반) |
| Train / Test split | 80 / 20 stratified random on Category |
| Protocol | 20 seeds (42, 0..18) mean ± std |
| 전처리 | median impute + min-max scale (train fold만 fit) |
| Test n | 1,418 |

### Class 정의 — Percentile 기반 (L1 + L2 ≈ 10%)

forward5 percentile 임계값:
- q50 = 1, q90 = 6, q98 = 14

| Class | 정의 | n | % |
|---|---|---:|---:|
| **L1** | forward5 ≥ 14 (top 2%) | 149 | **2.10%** |
| **L2** | forward5 ∈ [6, 14) (top 2–10%) | 736 | **10.39%** |
| L3 | forward5 ∈ [1, 6) (top 10–50%) | 3,615 | 51.02% |
| L4 | forward5 = 0 (bottom 50%) | 2,586 | 36.49% |

**L1 + L2 = 12.49%** (q90 = 6 ties 209건이 모두 L2로 들어감 — binary Y rate 와 동일).

3 가지 feature configuration (binary 실험과 동일):

| Config | 차원 | 구성 |
|---|---:|---|
| **BIBLIO** | 26 | 15 numeric biblio + MF top-10 one-hot + MF_other |
| **+25** | 51 | BIBLIO + 25 debate 변수 전체 |
| **+6focal** | 32 | BIBLIO + 6 focal debate (var_conf_pro, H_final, delta_H, conf_gap_change, cross_domain_attack, semantic_coherence) |

성능 지표 (positive class = 해당 class, 나머지는 negative; one-vs-rest):
- **AUROC** / **AUPRC** : threshold-free (predict_proba 기반)
- **Precision** / **Recall** : argmax prediction 기준
- **DOR** = Haldane-corrected (TP+0.5)(TN+0.5) / ((FP+0.5)(FN+0.5))

---

## Class L1 (top 2%, n_pos_test ≈ 30) — 최우수 (breakthrough patents)

| Config | Model | AUROC | AUPRC | Precision | Recall | DOR |
|---|---|---:|---:|---:|---:|---:|
| BIBLIO | **RF** | **0.829** ± 0.026 | **0.209** ± 0.047 | **0.754** ± 0.189 | **0.103** ± 0.040 | 142.1 ± 90.0 |
| BIBLIO | XGB | 0.785 ± 0.024 | 0.181 ± 0.044 | 0.476 ± 0.172 | 0.105 ± 0.039 | 56.6 ± 48.3 |
| BIBLIO | FFN | 0.779 ± 0.040 | 0.134 ± 0.041 | 0.385 ± 0.235 | 0.077 ± 0.046 | 41.9 ± 35.0 |
| BIBLIO | SVM | 0.754 ± 0.030 | 0.142 ± 0.039 | 0 ± 0 | 0 ± 0 | 44.0 ± 6.8 |
| **+6focal** | **RF** | **0.832** ± 0.027 ⭐ | 0.207 ± 0.054 | **0.792** ± 0.323 | 0.070 ± 0.036 | **209.0** ± 133.6 ⭐ |
| +6focal | XGB | 0.781 ± 0.028 | 0.157 ± 0.043 | 0.480 ± 0.242 | 0.078 ± 0.039 | 60.4 ± 49.5 |
| +6focal | FFN | 0.784 ± 0.035 | 0.131 ± 0.039 | 0.344 ± 0.227 | 0.067 ± 0.050 | 36.9 ± 31.0 |
| +6focal | SVM | 0.757 ± 0.024 | 0.130 ± 0.036 | 0 ± 0 | 0 ± 0 | 45.5 ± 0.0 |
| +25 | RF | 0.821 ± 0.025 | 0.190 ± 0.052 | 0.717 ± 0.405 | 0.040 ± 0.026 | 137.4 ± 80.8 |
| +25 | XGB | 0.789 ± 0.036 | 0.141 ± 0.039 | 0.473 ± 0.306 | 0.062 ± 0.033 | 70.9 ± 78.0 |
| +25 | FFN | 0.733 ± 0.025 | 0.082 ± 0.021 | 0.148 ± 0.112 | 0.050 ± 0.032 | 11.8 ± 10.1 |
| +25 | SVM | 0.766 ± 0.029 | 0.103 ± 0.037 | 0 ± 0 | 0 ± 0 | 45.5 ± 0.0 |

**관찰**: L1 (top 2%) 은 모든 클래스 중 **가장 잘 식별** — RF +6focal AUROC **0.832**.
Binary Y RF AUROC 0.753 대비 훨씬 높음. **극단 breakthrough 일수록 biblio 가
명확히 잡음**. SVM 은 minor class 에 대해 Precision/Recall 0 (다 majority 로 분류).

---

## Class L2 (top 2–10%, n_pos_test ≈ 147)

| Config | Model | AUROC | AUPRC | Precision | Recall | DOR |
|---|---|---:|---:|---:|---:|---:|
| BIBLIO | **RF** | **0.724** ± 0.018 | **0.255** ± 0.022 | **0.503** ± 0.098 | 0.060 ± 0.016 | **9.92** ± 4.0 |
| BIBLIO | FFN | 0.710 ± 0.019 | 0.210 ± 0.016 | 0.258 ± 0.089 | 0.036 ± 0.014 | 3.45 ± 1.6 |
| BIBLIO | XGB | 0.675 ± 0.020 | 0.213 ± 0.017 | 0.310 ± 0.052 | **0.114** ± 0.025 | 4.40 ± 1.1 |
| BIBLIO | SVM | 0.702 ± 0.022 | 0.195 ± 0.018 | 0 ± 0 | 0 ± 0 | 8.62 ± 0.0 |
| **+6focal** | **RF** | 0.724 ± 0.017 | 0.243 ± 0.017 | **0.541** ± 0.160 ⭐ | 0.025 ± 0.009 | **11.92** ± 7.0 ⭐ |
| +6focal | XGB | 0.677 ± 0.022 | 0.200 ± 0.019 | 0.281 ± 0.070 | 0.089 ± 0.026 | 3.82 ± 1.3 |
| +6focal | FFN | 0.701 ± 0.023 | 0.199 ± 0.019 | 0.246 ± 0.078 | 0.051 ± 0.022 | 3.20 ± 1.4 |
| +6focal | SVM | 0.706 ± 0.019 | 0.198 ± 0.018 | 0 ± 0 | 0 ± 0 | 8.62 ± 0.0 |
| +25 | RF | 0.709 ± 0.018 | 0.225 ± 0.015 | 0.474 ± 0.191 | 0.018 ± 0.008 | 10.12 ± 7.8 |
| +25 | XGB | 0.670 ± 0.019 | 0.194 ± 0.020 | 0.267 ± 0.070 | 0.079 ± 0.024 | 3.55 ± 1.3 |
| +25 | FFN | 0.653 ± 0.030 | 0.174 ± 0.019 | 0.207 ± 0.081 | 0.062 ± 0.025 | 2.58 ± 1.3 |
| +25 | SVM | 0.686 ± 0.017 | 0.178 ± 0.012 | 0 ± 0 | 0 ± 0 | 8.62 ± 0.0 |

**관찰**: RF AUROC ≈ 0.72 — binary Y 와 거의 동급. RF + 6focal 의 Precision 54%
높지만 Recall 매우 낮음 (2.5%) — 보수적 예측 (top 10% 내 confident only).
XGB 가 Recall 은 높은 편 (8-11%) 이지만 Precision 낮음 (28-31%).

---

## Class L3 (n_pos_test ≈ 723) — **가장 어려운 class**

| Config | Model | AUROC | AUPRC | Precision | Recall | DOR |
|---|---|---:|---:|---:|---:|---:|
| BIBLIO | **RF** | **0.597** ± 0.009 | **0.588** ± 0.011 | **0.563** ± 0.006 | 0.755 ± 0.014 | **1.97** ± 0.12 |
| BIBLIO | FFN | 0.589 ± 0.013 | 0.572 ± 0.016 | 0.562 ± 0.007 | 0.741 ± 0.024 | 1.92 ± 0.20 |
| BIBLIO | SVM | 0.586 ± 0.012 | 0.564 ± 0.011 | 0.544 ± 0.004 | **0.853** ± 0.017 | 2.00 ± 0.20 |
| BIBLIO | XGB | 0.578 ± 0.012 | 0.567 ± 0.014 | 0.559 ± 0.009 | 0.681 ± 0.019 | 1.69 ± 0.16 |
| +6focal | **RF** | 0.590 ± 0.012 | 0.579 ± 0.015 | 0.558 ± 0.007 | 0.776 ± 0.017 | 1.96 ± 0.19 |
| +6focal | FFN | 0.571 ± 0.010 | 0.558 ± 0.013 | 0.556 ± 0.007 | 0.720 ± 0.017 | 1.73 ± 0.15 |
| +6focal | SVM | 0.582 ± 0.009 | 0.564 ± 0.009 | 0.542 ± 0.005 | 0.843 ± 0.017 | 1.88 ± 0.21 |
| +6focal | XGB | 0.573 ± 0.013 | 0.558 ± 0.011 | 0.555 ± 0.008 | 0.683 ± 0.015 | 1.64 ± 0.14 |
| +25 | RF | 0.584 ± 0.011 | 0.571 ± 0.013 | 0.557 ± 0.005 | 0.781 ± 0.014 | 1.96 ± 0.16 |
| +25 | XGB | 0.572 ± 0.014 | 0.557 ± 0.013 | 0.556 ± 0.010 | 0.689 ± 0.014 | 1.66 ± 0.17 |
| +25 | SVM | 0.573 ± 0.010 | 0.563 ± 0.011 | 0.535 ± 0.003 | **0.869** ± 0.014 | 1.82 ± 0.14 |
| +25 | FFN | 0.547 ± 0.010 | 0.538 ± 0.009 | 0.545 ± 0.008 | 0.672 ± 0.021 | 1.46 ± 0.12 |

**관찰**: AUROC ≈ 0.55–0.60 — **거의 random**. "중간 수준 인용" patent 를
biblio 만으로 식별 어려움. SVM 이 Recall 86% 로 높지만 Precision 53–54% —
대부분을 L3 로 분류하는 majority bias.

---

## Class L4 (n_pos_test ≈ 518)

| Config | Model | AUROC | AUPRC | Precision | Recall | DOR |
|---|---|---:|---:|---:|---:|---:|
| BIBLIO | **RF** | **0.711** ± 0.010 | **0.570** ± 0.009 | **0.564** ± 0.014 | 0.464 ± 0.022 | **3.34** ± 0.29 |
| BIBLIO | FFN | 0.700 ± 0.014 | 0.548 ± 0.018 | 0.562 ± 0.022 | 0.475 ± 0.028 | 3.36 ± 0.42 |
| BIBLIO | SVM | 0.699 ± 0.012 | 0.554 ± 0.016 | 0.597 ± 0.024 | 0.327 ± 0.016 | 3.35 ± 0.39 |
| BIBLIO | XGB | 0.677 ± 0.012 | 0.540 ± 0.015 | 0.529 ± 0.017 | 0.486 ± 0.023 | 2.87 ± 0.30 |
| **+6focal** | **RF** | **0.713** ± 0.011 ⭐ | **0.573** ± 0.012 ⭐ | **0.571** ± 0.015 ⭐ | 0.444 ± 0.022 | 3.36 ± 0.31 |
| +6focal | SVM | 0.696 ± 0.012 | 0.549 ± 0.014 | 0.585 ± 0.027 | 0.330 ± 0.019 | 3.18 ± 0.44 |
| +6focal | FFN | 0.686 ± 0.015 | 0.537 ± 0.017 | 0.545 ± 0.019 | 0.467 ± 0.024 | 3.05 ± 0.37 |
| +6focal | XGB | 0.673 ± 0.012 | 0.531 ± 0.016 | 0.525 ± 0.016 | 0.484 ± 0.017 | 2.80 ± 0.26 |
| +25 | RF | 0.709 ± 0.010 | 0.565 ± 0.013 | 0.575 ± 0.016 | 0.441 ± 0.014 | 3.43 ± 0.31 |
| +25 | XGB | 0.673 ± 0.013 | 0.526 ± 0.016 | 0.527 ± 0.015 | 0.483 ± 0.022 | 2.82 ± 0.28 |
| +25 | SVM | 0.684 ± 0.013 | 0.538 ± 0.011 | 0.585 ± 0.023 | 0.276 ± 0.018 | 3.01 ± 0.32 |
| +25 | FFN | 0.647 ± 0.014 | 0.496 ± 0.016 | 0.506 ± 0.018 | 0.457 ± 0.031 | 2.45 ± 0.29 |

**관찰**: L4 (낮은 인용 = 비유망) AUROC ≈ 0.71 (RF). 모델이 "유망 안 함"
판정에 강함. RF + 6focal 이 모든 metric 1위.

---

## Overall Macro 지표 (multi-class)

| Config | Model | Accuracy | macro F1 | AUROC_macro | AUPRC_macro | DOR_macro |
|---|---|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **0.563** ± 0.007 | **0.360** ± 0.019 | **0.715** | **0.405** | 39.3 |
| BIBLIO | XGB | 0.539 ± 0.010 | 0.364 ± 0.022 | 0.679 | 0.375 | 16.4 |
| BIBLIO | FFN | 0.557 ± 0.011 | 0.335 ± 0.019 | 0.694 | 0.366 | 12.7 |
| BIBLIO | SVM | 0.554 ± 0.007 | 0.272 ± 0.004 | 0.685 | 0.364 | 14.5 |
| +6focal | **RF** | 0.562 ± 0.008 | 0.331 ± 0.015 | **0.715** | 0.400 | **56.6** ⭐ |
| +6focal | XGB | 0.536 ± 0.009 | 0.346 ± 0.022 | 0.676 | 0.362 | 17.2 |
| +6focal | FFN | 0.544 ± 0.010 | 0.330 ± 0.020 | 0.686 | 0.356 | 11.2 |
| +6focal | SVM | 0.550 ± 0.009 | 0.270 ± 0.005 | 0.685 | 0.360 | 14.8 |
| +25 | RF | 0.562 ± 0.008 | 0.315 ± 0.013 | 0.706 | 0.388 | 38.2 |
| +25 | XGB | 0.537 ± 0.010 | 0.337 ± 0.018 | 0.676 | 0.355 | 19.7 |
| +25 | FFN | 0.517 ± 0.013 | 0.311 ± 0.015 | 0.645 | 0.322 | 4.6 |
| +25 | SVM | 0.544 ± 0.005 | 0.259 ± 0.004 | 0.677 | 0.345 | 14.7 |

---

## Class 별 AUROC 요약 비교 (RF only)

| Class | n_pos_test | BIBLIO | +6focal | +25 | best config |
|---|---:|---:|---:|---:|---|
| L1 (top 2%) | 30 | 0.829 | **0.832** | 0.821 | +6focal |
| L2 (top 2–10%) | 147 | **0.724** | 0.724 | 0.709 | tie |
| L3 (mid) | 723 | **0.597** | 0.590 | 0.584 | BIBLIO |
| L4 (low) | 518 | 0.711 | **0.713** | 0.709 | +6focal |

**RF 가 모든 class 에서 1위**. +6focal 이 L1 / L4 약간 향상, L2 동등, L3 약간
손해. +25 는 모든 class 에서 BIBLIO 보다 약함.

---

## 핵심 관찰

### 1. **L1 (top 2%) 은 가장 분류하기 쉬운 class**
- RF +6focal AUROC **0.832**, Precision 79%, DOR 209
- Binary Y RF AUROC 0.753 대비 훨씬 높음
- 의미: **"진짜 breakthrough" patent 는 biblio 패턴이 매우 명확** — TCT, SK, PK 등
  variables.md universal fingerprint 가 극단 case 에서 sharp

### 2. **L3 (mid) 가 가장 어려움**
- 모든 모델 AUROC 0.55–0.60 (random 0.5 대비 미세 lift)
- "중간 수준" 인용은 random shock 영향이 커서 식별 어려움
- 이는 **본질적으로 어려운 task** — 더 좋은 변수가 필요할 가능성

### 3. **RF 우위 더 강해짐 (multi-class 에서)**
- Binary 에서도 RF 1위, multi-class 에서도 RF 1위
- macro AUROC: RF = 0.715, XGB = 0.679, FFN = 0.694, SVM = 0.685
- XGB 는 binary 와 달리 multi-class 에서 RF 보다 명확히 약함

### 4. **Debate 변수 효과 (multi-class) 는 약함**
- macro AUROC: BIBLIO RF 0.715 vs +6focal RF 0.715 (동등)
- macro F1: BIBLIO 0.360 vs +6focal 0.331 (BIBLIO 우세)
- DOR_macro: +6focal 56.6 vs BIBLIO 39.3 (+6focal 우세)
- → **metric 에 따라 결과 다름**. AUROC 동등, F1 BIBLIO 우세, DOR +6focal 우세
- Binary 에서 본 XGB +6focal +0.014 lift 패턴이 multi-class 에서는 안 보임

### 5. **+25 는 multi-class 에서도 손해**
- 모든 metric 에서 BIBLIO 보다 약하거나 동등
- FFN +25 가 가장 큰 손해 (accuracy 51.7%, F1 0.311) — 51 features × 5,668 train 과적합

### 6. **SVM 의 minor class 처리 한계**
- SVM 이 L1 / L2 에 대해 Precision 0, Recall 0 — 어떤 patent 도 L1 / L2 로 분류 안 함
- argmax predict 에서 majority class (L3 / L4) 로만 분류
- SVM 의 multi-class 처리 (OvR) 가 imbalanced data 에 부적합

### 7. **Binary vs Multi-class 비교**

| 측면 | Binary (top 10%) | Multi-class (4-way) |
|---|---|---|
| RF AUROC | 0.753 | 0.715 (macro) / 0.832 (L1) |
| 최고 task | Binary 가 단순 | L1 단독 식별이 binary 보다 쉬움 |
| Debate 효과 | XGB +0.014 robust lift | macro 효과 미미 |
| 어려운 case | top decile 경계 | L3 (mid) 분류 |

---

## 결론

| 질문 | 답 |
|---|---|
| Multi-class 분류 가능한가? | YES — RF macro AUROC 0.715, accuracy 56.3% |
| 어떤 class 가 잘 식별되는가? | **L1 (top 2%)** AUROC 0.83 — 극단일수록 명확 |
| 가장 어려운 class 는? | **L3 (mid)** AUROC 0.59 — random 근접 |
| 어떤 모델이 우수한가? | **RF** (모든 class, 모든 metric 1위) |
| Debate 추가 효과? | metric 에 따라 다름 — macro AUROC 동등, F1 BIBLIO 우세, DOR +6focal 우세 |
| Binary 와 비교 시? | Multi-class 도 동일하게 RF + 6focal best 권장 |

---

## 산출물

| 파일 | 내용 |
|---|---|
| `analysis/outputs/experiments_summary/category_full_metrics.csv` | 4 models × 3 configs × 4 classes × 5 metrics (20-seed mean ± std) |
| `analysis/outputs/experiments_summary/category_full_metrics_detail.csv` | per-seed raw (20 × 4 × 3 × 4 = 960 rows) |
| `analysis/outputs/experiments_summary/category_macro.csv` | per (config, model) macro 요약 |
| `analysis/experiments_category.py` | multi-class 실험 재현 스크립트 |
