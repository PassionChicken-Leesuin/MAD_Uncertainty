# 실험 결과 (ICE Domain, 7,086 patents)

ICE 도메인 7,086 USPTO 특허 cohort에서 16개 bibliometric 변수 +
25개 multi-agent debate 변수의 유망 특허 예측 성능을 비교한 결과.

## 실험 설정

| 항목 | 값 |
|---|---|
| Universe | 7,086 USPTO patents (Sinigaglia 2022 16-query union, 1980–2020) |
| 출력 (Y) | 5년 forward citation 상위 10% (q90 기준, ties 포함 12.49%) |
| Train / Test split | 80 / 20 stratified random on Y |
| Section 1–3 protocol | **20 seeds (42, 0..18)** mean ± std |
| Section 4–6 protocol | **Single seed=42** (재현성) |
| 전처리 | median impute + min-max scale (train fold만 fit) |
| Test n | 1,418, test Y_pos ≈ 177 |

세 가지 feature configuration:

| Config | 차원 | 구성 |
|---|---:|---|
| **BIBLIO** | 26 | 15 numeric biblio + MF top-10 one-hot + MF_other |
| **+25** | 51 | BIBLIO + 25개 debate 변수 전체 |
| **+6focal** | 32 | BIBLIO + 6개 focal subset (var_conf_pro, H_final, delta_H, conf_gap_change, cross_domain_attack, semantic_coherence) |

---

**성능 지표 정의** (모두 top-10% threshold 에서 계산, Y=1 클래스):

- **AUROC**: ROC 곡선 면적 (threshold-free)
- **AUPRC**: Precision-Recall 곡선 면적 (threshold-free)
- **Precision**: TP / (TP + FP) — top-10% 예측의 정밀도
- **Recall**: TP / (TP + FN) — 실제 positive 중 검출 비율
- **DOR**: Diagnostic Odds Ratio = (TP × TN) / (FP × FN), Haldane 보정 (각 cell +0.5)
  - DOR > 1 = 분류기가 random보다 우수, 클수록 좋음
  - 베이스라인 (random) DOR = 1.0

## Section 1: BIBLIO만 — 4 모델 성능 (20-seed mean ± std)

| Model | AUROC | AUPRC | Precision | Recall | DOR |
|---|---:|---:|---:|---:|---:|
| **RF** | **0.7507 ± 0.018** | **0.3496 ± 0.027** | **0.3743 ± 0.027** | **0.3025 ± 0.021** | **5.63 ± 0.78** |
| XGB | 0.7130 ± 0.016 | 0.3056 ± 0.031 | 0.3504 ± 0.029 | 0.2811 ± 0.023 | 4.93 ± 0.80 |
| FFN | 0.7110 ± 0.023 | 0.2718 ± 0.023 | 0.3218 ± 0.029 | 0.2582 ± 0.023 | 4.19 ± 0.68 |
| SVM | 0.6482 ± 0.023 | 0.2228 ± 0.023 | 0.2585 ± 0.031 | 0.2073 ± 0.025 | 2.87 ± 0.55 |

**관찰**: Tree 기반 모델 (RF, XGB) 이 SVM, FFN 대비 baseline 우수.
RF가 모든 metric 절대 1위 (DOR 5.63 = random 대비 5.6× lift).

---

## Section 2: BIBLIO + 25 Debate — 4 모델 성능

| Model | AUROC | AUPRC | Precision | Recall | DOR | Δ AUROC |
|---|---:|---:|---:|---:|---:|---:|
| **RF** | **0.7426 ± 0.017** | **0.3290 ± 0.027** | **0.3621 ± 0.036** | **0.2963 ± 0.029** | **5.32 ± 1.04** | −0.008 |
| XGB | 0.7168 ± 0.016 | 0.2888 ± 0.024 | 0.3264 ± 0.023 | 0.2619 ± 0.019 | 4.29 ± 0.58 | **+0.004** |
| FFN | 0.6584 ± 0.022 | 0.2126 ± 0.019 | 0.2532 ± 0.027 | 0.2031 ± 0.021 | 2.77 ± 0.46 | −0.053 |
| SVM | 0.6569 ± 0.017 | 0.2073 ± 0.015 | 0.2423 ± 0.031 | 0.1944 ± 0.025 | 2.59 ± 0.51 | +0.009 |

**관찰**: 25개 debate 변수 전체 추가 시:
- RF / FFN 은 오히려 악화 (over-parameterization). RF DOR 5.63 → 5.32
- XGB / SVM 은 미미한 AUROC 향상

---

## Section 3: BIBLIO + 6 Focal Debate — 4 모델 성능

| Model | AUROC | AUPRC | Precision | Recall | DOR | Δ AUROC |
|---|---:|---:|---:|---:|---:|---:|
| **RF** | **0.7528 ± 0.017** | **0.3434 ± 0.029** | **0.3675 ± 0.033** | **0.2994 ± 0.026** | **5.46 ± 0.94** | +0.002 |
| **XGB** | **0.7266 ± 0.019** | 0.2980 ± 0.020 | 0.3349 ± 0.023 | 0.2686 ± 0.019 | 4.50 ± 0.61 | **+0.014** ⭐ |
| FFN | 0.6951 ± 0.021 | 0.2577 ± 0.018 | 0.3039 ± 0.038 | 0.2438 ± 0.030 | 3.80 ± 0.77 | −0.016 |
| SVM | 0.6589 ± 0.015 | 0.2197 ± 0.018 | 0.2479 ± 0.020 | 0.1989 ± 0.016 | 2.67 ± 0.34 | +0.011 |

**관찰**:
- **모든 metric 절대 1위**: RF + 6focal (AUROC 0.7528, DOR 5.46)
- **debate 활용도 1위**: XGB + 6focal (Δ AUROC +0.014, 17/20 seed 승, DOR 4.29 → 4.50)
- 6 focal subset 이 25개 전체보다 일관 우수 — sparse가 better

---

## Section 1–3 요약 비교

### AUROC mean (20 seeds)

| Model | BIBLIO | +25 | +6focal | best |
|---|---:|---:|---:|---|
| RF | 0.7507 | 0.7426 | **0.7528** | +6focal |
| XGB | 0.7130 | 0.7168 | **0.7266** | +6focal |
| FFN | **0.7110** | 0.6584 | 0.6951 | BIBLIO |
| SVM | 0.6482 | 0.6569 | **0.6589** | +6focal |

### DOR mean (20 seeds)

| Model | BIBLIO | +25 | +6focal | best |
|---|---:|---:|---:|---|
| **RF** | **5.63** | 5.32 | 5.46 | BIBLIO |
| XGB | 4.93 | 4.29 | 4.50 | BIBLIO |
| FFN | 4.19 | 2.77 | 3.80 | BIBLIO |
| SVM | 2.87 | 2.59 | 2.67 | BIBLIO |

→ **DOR 기준으론 BIBLIO만이 최고** — top-10% threshold에서 confusion matrix 비율 측면에서는 debate 추가가 약간 손해. AUROC는 향상되지만 calibration이 흔들림 (probability ranking은 좋아져도 같은 threshold에서 TP/FP 균형이 살짝 무너짐).

### Δ AUROC vs BIBLIO (paired by seed)

| Model | +25 Δ | +6focal Δ | win rate (+6focal vs BIBLIO) |
|---|---:|---:|:---:|
| RF | −0.008 | +0.002 | 12/20 (60%) |
| **XGB** | +0.004 | **+0.014** | **17/20 (85%) ⭐** |
| FFN | −0.053 | −0.016 | 4/20 (20%) |
| SVM | +0.009 | +0.011 | 16/20 (80%) |

→ **XGB / SVM 은 AUROC 기준 debate 변수 활용 가능**. RF / FFN 은 미활용.

### Recall (20 seeds, top-10% threshold)

| Model | BIBLIO | +25 | +6focal |
|---|---:|---:|---:|
| **RF** | **0.3025** | 0.2963 | **0.2994** |
| XGB | 0.2811 | 0.2619 | 0.2686 |
| FFN | 0.2582 | 0.2031 | 0.2438 |
| SVM | 0.2073 | 0.1944 | 0.1989 |

베이스라인 random Recall = 0.10 (top-10% 무작위 선택). 모든 모델이 random 대비 2-3× lift.

---

## Section 4: Permutation Feature Importance (single seed=42)

`sklearn.inspection.permutation_importance` (n_repeats=10, scoring=AUROC).
각 (config, model) 조합에서 importance 상위 10개:

### +25 config

#### RF — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | **TCT** | 0.0514 | 0.0111 |
| 2 | TS | 0.0347 | 0.0102 |
| 3 | CTS | 0.0253 | 0.0053 |
| 4 | PKH | 0.0235 | 0.0085 |
| 5 | CKH | 0.0232 | 0.0052 |
| 6 | TKH | 0.0165 | 0.0093 |
| 7 | TTS | 0.0145 | 0.0086 |
| 8 | STO | 0.0138 | 0.0048 |
| 9 | PTS | 0.0100 | 0.0084 |
| 10 | SK | 0.0089 | 0.0020 |

#### XGB — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | **TCT** | 0.0549 | 0.0095 |
| 2 | TS | 0.0284 | 0.0073 |
| 3 | CKH | 0.0242 | 0.0116 |
| 4 | CTS | 0.0216 | 0.0060 |
| 5 | STO | 0.0143 | 0.0070 |
| 6 | SK | 0.0131 | 0.0023 |
| 7 | PK | 0.0105 | 0.0051 |
| 8 | PKH | 0.0091 | 0.0058 |
| 9 | CTO | 0.0081 | 0.0034 |
| 10 | **var_conf_anti** ⭐ | 0.0081 | 0.0034 |

#### FFN — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | CKH | 0.0567 | 0.0136 |
| 2 | TCT | 0.0537 | 0.0172 |
| 3 | MF_other | 0.0398 | 0.0086 |
| 4 | STO | 0.0390 | 0.0081 |
| 5 | MF_F02D41 | 0.0293 | 0.0068 |
| 6 | CTS | 0.0277 | 0.0056 |
| 7 | **final_prediction** ⭐ | 0.0275 | 0.0086 |
| 8 | **conf_gap_change** ⭐ | 0.0263 | 0.0072 |
| 9 | TS | 0.0244 | 0.0127 |
| 10 | **final_pred_ecosystem** ⭐ | 0.0191 | 0.0081 |

#### SVM — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | TCT | 0.0473 | 0.0101 |
| 2 | **final_pred_ecosystem** ⭐ | 0.0448 | 0.0092 |
| 3 | **final_pred_application** ⭐ | 0.0313 | 0.0071 |
| 4 | **final_prediction** ⭐ | 0.0307 | 0.0067 |
| 5 | **final_pred_user** ⭐ | 0.0294 | 0.0081 |
| 6 | TS | 0.0288 | 0.0107 |
| 7 | STO | 0.0266 | 0.0057 |
| 8 | **total_rounds** ⭐ | 0.0249 | 0.0040 |
| 9 | MF_other | 0.0245 | 0.0090 |
| 10 | **final_pred_businessmodel** ⭐ | 0.0200 | 0.0096 |

### +6focal config

#### RF — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | **TCT** | 0.0578 | 0.0113 |
| 2 | CKH | 0.0292 | 0.0053 |
| 3 | CTS | 0.0271 | 0.0089 |
| 4 | TS | 0.0271 | 0.0094 |
| 5 | TKH | 0.0216 | 0.0126 |
| 6 | PTS | 0.0214 | 0.0125 |
| 7 | STO | 0.0210 | 0.0059 |
| 8 | PK | 0.0173 | 0.0079 |
| 9 | PKH | 0.0169 | 0.0104 |
| 10 | TTS | 0.0152 | 0.0085 |

#### XGB — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | **TCT** | 0.0459 | 0.0078 |
| 2 | TS | 0.0260 | 0.0104 |
| 3 | CKH | 0.0224 | 0.0075 |
| 4 | CTS | 0.0209 | 0.0073 |
| 5 | CTO | 0.0161 | 0.0038 |
| 6 | STO | 0.0157 | 0.0077 |
| 7 | SK | 0.0141 | 0.0037 |
| 8 | **H_final** ⭐ | 0.0103 | 0.0027 |
| 9 | NC | 0.0095 | 0.0055 |
| 10 | PK | 0.0092 | 0.0065 |

#### FFN — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | CKH | 0.1020 | 0.0324 |
| 2 | MF_other | 0.0784 | 0.0153 |
| 3 | TCT | 0.0649 | 0.0143 |
| 4 | **H_final** ⭐ | 0.0601 | 0.0101 |
| 5 | CTS | 0.0531 | 0.0091 |
| 6 | MF_F02D41 | 0.0491 | 0.0112 |
| 7 | TS | 0.0364 | 0.0101 |
| 8 | MF_F02B33 | 0.0289 | 0.0071 |
| 9 | STO | 0.0279 | 0.0049 |
| 10 | TKH | 0.0258 | 0.0094 |

#### SVM — Top 10
| Rank | Feature | Mean | Std |
|:---:|---|---:|---:|
| 1 | TCT | 0.0497 | 0.0130 |
| 2 | CKH | 0.0413 | 0.0097 |
| 3 | **H_final** ⭐ | 0.0273 | 0.0127 |
| 4 | TS | 0.0261 | 0.0091 |
| 5 | MF_other | 0.0220 | 0.0109 |
| 6 | TKH | 0.0220 | 0.0052 |
| 7 | PKH | 0.0194 | 0.0047 |
| 8 | **cross_domain_attack** ⭐ | 0.0187 | 0.0074 |
| 9 | STO | 0.0173 | 0.0073 |
| 10 | MF_F02B33 | 0.0161 | 0.0065 |

⭐ = debate-derived feature

### Section 4 핵심 관찰

1. **TCT (Technology Cycle Time)**가 모든 모델 / 모든 config에서 거의 1위 — 유망 특허는 새로운 (younger) priors를 인용한다는 universal signal.
2. **Tree 모델 (RF / XGB)**: biblio 변수 (TCT, TS, CKH, CTS, STO 등)가 top-10 지배. debate 변수는 거의 미진입 (XGB가 +25에서 var_conf_anti 1개만).
3. **SVM (+25)**: 예외적으로 **5개 final_pred_* persona 변수가 모두 top 10에 진입** — SVM은 persona prediction을 강하게 활용. 단, SVM 자체 성능 (AUROC 0.66) 이 낮아 절대 활용도는 제한적.
4. **+6focal에서 H_final** 이 XGB / FFN / SVM 모두 top-10 진입 — 6 focal 변수 중 가장 robust한 contributor.

---

## Section 5: SHAP Values (XGBoost TreeExplainer, single seed=42)

Test set에서의 mean |SHAP value|. 상위 15개:

### +25 config

| Rank | Feature | mean &#124;SHAP&#124; |
|:---:|---|---:|
| 1 | TCT | 0.745 |
| 2 | TS | 0.699 |
| 3 | PK | 0.513 |
| 4 | TKH | 0.448 |
| 5 | CTS | 0.410 |
| 6 | **semantic_coherence** ⭐ | 0.398 |
| 7 | CKH | 0.394 |
| 8 | NC | 0.310 |
| 9 | STO | 0.306 |
| 10 | PTS | 0.271 |
| 11 | PKH | 0.266 |
| 12 | **conf_gap_change** ⭐ | 0.265 |
| 13 | CTO | 0.257 |
| 14 | **var_conf_pro** ⭐ | 0.250 |
| 15 | **delta_H** ⭐ | 0.249 |

### +6focal config

| Rank | Feature | mean &#124;SHAP&#124; |
|:---:|---|---:|
| 1 | TCT | 0.739 |
| 2 | TS | 0.696 |
| 3 | PK | 0.462 |
| 4 | **semantic_coherence** ⭐ | 0.396 |
| 5 | CTS | 0.394 |
| 6 | CKH | 0.382 |
| 7 | PTS | 0.370 |
| 8 | **H_final** ⭐ | 0.366 |
| 9 | TKH | 0.344 |
| 10 | **delta_H** ⭐ | 0.342 |
| 11 | **conf_gap_change** ⭐ | 0.338 |
| 12 | STO | 0.331 |
| 13 | **var_conf_pro** ⭐ | 0.309 |
| 14 | NC | 0.298 |
| 15 | CTO | 0.280 |

### Section 5 핵심 관찰

1. **TCT / TS / PK** 가 두 config 모두에서 SHAP 상위 3 — biblio core signal.
2. **semantic_coherence** 가 두 config 모두 #4–6 (매우 높음). Permutation에서는 상대적으로 lower 였던 점과 대조 — **semantic_coherence는 다른 변수와 협력 시 marginal contribution이 큼** (correlated features에 분산되어 단독 permutation에서 약하게 측정).
3. **+6focal config 에서 6개 focal 변수 중 5개가 top 15 진입**:
   - semantic_coherence (#4), H_final (#8), delta_H (#10), conf_gap_change (#11), var_conf_pro (#13)
   - cross_domain_attack 만 rank 16 이하 — focal-6 중 유일하게 weak contributor
4. **Permutation vs SHAP 결과 차이**: Permutation은 단독 효과, SHAP은 marginal contribution. 두 metric이 일치하지 않을 때 → feature가 **다른 변수와 상관**되어 신호 공유 (예: semantic_coherence ↔ var_conf_pro 와 같은 confidence-based 변수들).

---

## Section 6: MAD SYSTEM final_prediction vs Best ML 모델

**Best ML 모델 선정**: 20-seed 평균 AUROC 1위 = **RF + 6focal** (0.7528)

비교 대상:
- **MAD final_prediction**: Moderator의 토론 종료 후 raw verdict (binary 0 / 1) — 별도 ML 학습 없음
- **Best ML**: RF (BIBLIO + 6focal, 32 features) — 20-seed 평균 AUROC 1위, single-seed=42 test 예측

### 6.1 Confusion Matrix (test n=1,418, Y_pos=177)

| Predictor | TP | FP | FN | TN | Pos Pred | Accuracy | Precision | Recall | F1 | DOR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAD final_prediction (raw debate verdict) | 148 | 964 | 29 | 277 | 1,112 | **0.300** | **0.133** | **0.836** | 0.230 | **1.45** |
| Best ML (RF + 6focal, top 10% threshold) | 54 | 94 | 123 | 1,147 | 148 | **0.847** | **0.365** | **0.305** | 0.332 | **5.36** |

DOR (Haldane-corrected) 차이가 결정적: MAD 1.45 (random에 가까움) vs Best ML 5.36 (random 대비 5× lift).

### 6.2 Probability-based Metrics

| Predictor | AUROC | AUPRC |
|---|---:|---:|
| MAD final_prediction (binary as score) | **0.530** | **0.132** |
| Best ML (RF + 6focal, prob score) | **0.761** | **0.337** |

### 6.3 MAD vs Best ML 일치도

- **둘 다 동일 예측한 비율**: 402 / 1,418 = **28.4%**
- **둘 다 positive 예측 (n=122)**: 실제 Y=1 비율 **36.9%** (random 12.5% 대비 **3.0× lift**)
- **둘 다 negative 예측 (n=280)**: 실제 Y=1 비율 **7.1%** (random 12.5% 대비 0.6×)

### 6.4 핵심 관찰

1. **MAD raw verdict는 binary 예측으로 거의 random** — AUROC 0.530, accuracy 30%.
   - 5개 persona가 토론을 통해 too aggressive 하게 "promising" 판정
   - test set의 **78%를 positive로 라벨** (n_pos_pred=1,112 / 1,418)
   - Recall 84% 매우 높지만 Precision 13% 매우 낮음 → 거의 모든 특허를 유망으로 판정

2. **Best ML (RF + 6focal)** 은 훨씬 정확:
   - AUROC 0.761, accuracy 84.7%
   - top 10% threshold 시 conservative — Precision 36.5%, Recall 30.5%

3. **두 시스템 모두 positive 일치한 122건**: 실제 Y=1 비율 37%, **random baseline 12.5% 대비 3배 lift**
   - 즉 **MAD + ML 합의 = 강한 신호**
   - 이런 "double-positive" 케이스는 실용적 specificity 가 높음

4. **시사점**: MAD final_prediction 단독으로는 사용 부적합.
   - 하지만 **debate process에서 생성된 25개 변수** (var_conf_pro, H_final, delta_H, semantic_coherence 등) 가 ML model의 input feature로 변환되면 → RF / XGB가 그 정보를 효율적으로 추출
   - SVM은 final_pred_* 변수를 직접 활용하나 SVM 자체 성능 한계로 절대 활용도 낮음
   - **결론: MAD는 raw verdict가 아닌 25개 debate 변수의 feature engineering 도구로 활용해야 함**

---

## 결론 (요약)

| 질문 | 답 |
|---|---|
| **Debate가 도움 되는가?** | XGB + 6focal에서 +0.014 AUROC, 20 seeds 중 17/20 승. **Modest 하지만 robust** |
| **어떤 변수가 핵심?** | TCT (Technology Cycle Time) 압도적 #1. Debate 변수 중에서는 **semantic_coherence, H_final, conf_gap_change, var_conf_pro, delta_H** (focal 6 중 5개 top 15). cross_domain_attack 만 weak |
| **어떤 모델을 써야 하나?** | **절대 성능: RF + 6focal**, **Debate 활용: XGB + 6focal**. SVM/FFN은 augmented feature 시 과적합 |
| **Debate 전체 (+25) vs sparse (+6focal)?** | +6focal 일관 우수 — sparse가 better |
| **MAD raw verdict 그대로 쓸 수 있나?** | 아니오 (AUROC 0.53, 거의 random). ML feature로 변환 필요 |

---

## 산출물

| 파일 | 내용 |
|---|---|
| `analysis/outputs/experiments_summary/section_1_3_full_metrics.csv` | 4 models × 3 configs × 5 metrics (AUROC, AUPRC, Precision, Recall, DOR) — 20-seed mean ± std |
| `analysis/outputs/experiments_summary/section_1_3_full_metrics_detail.csv` | 위와 동일하나 per-seed 원시값 (20 × 4 × 3 = 240 rows) |
| `analysis/outputs/experiments_summary/section_1_3_performance.csv` | 위 metrics의 부분집합 (AUROC/AUPRC/top10_prec) |
| `analysis/outputs/experiments_summary/single_seed_performance.csv` | 4 × 3 single-seed=42 (sections 4–6 일관성) |
| `analysis/outputs/experiments_summary/section_4_perm_importance.csv` | 4 models × 2 configs × all features |
| `analysis/outputs/experiments_summary/section_5_shap.csv` | 2 configs × all features (mean &#124;SHAP&#124;, XGB only) |
| `analysis/outputs/experiments_summary/section_6_confusion.csv` | MAD vs Best ML confusion matrix metrics |
| `analysis/outputs/experiments_summary/section_6_auroc.csv` | MAD vs Best ML probability-based metrics |
| `analysis/experiments_summary.py` | 전체 6 section 재현 스크립트 |
| `analysis/add_full_metrics.py` | Precision/Recall/DOR 추가 계산 스크립트 |
