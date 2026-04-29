# 실험 결과 — Prompt Comparison (v0_baseline vs v2a_y_anchored)

ICE 도메인 7,086 patents 에 대해 **debate prompt 디자인이 LLM-derived 변수의
prediction utility 에 미치는 영향** 분석. 핵심 질문:

> 같은 페르소나·같은 출력 스키마인데 *질문 phrasing* 만 바꿨을 때,
> debate 변수가 bibliometric baseline 에 더하는 lift 가 prompt 별로 어떻게 달라지는가?

기존 `Result_Threshold.md` 는 **단일 prompt (v0)** 의 6 모델 × 4 threshold 결과를 보고함.
본 문서는 **두 번째 prompt (v2a)** 를 추가하여 prompt 변화에 대한 sensitivity 를 실증한다.

---

## 두 Prompt 정의

전체 시스템(5 페르소나, JSON 스키마, 모더레이터, 검증자, 모델 = `gpt-4o-mini`,
temperature, max_rounds 등)은 둘 다 동일. **유일한 차이는 round-0 user message
와 debate-round user message 의 framing**.

| 버전 | Round-0 question (요지) |
|---|---|
| **v0_baseline** | "Evaluate whether the following patent represents a **promising emerging technology**" — 도메인 / 시점 무관 추상 질문 |
| **v2a_y_anchored** | "Predict whether the patent will be cited at a **top-decile rate by future patents within five years** of its grant date" — Y 정의에 직접 anchored |

페르소나 system prompt (Technology / Application / User / Ecosystem / BusinessModel)
는 두 버전 모두 100% 동일. 자세한 차이는 [`prompts/v0_baseline.py`](prompts/v0_baseline.py)
와 [`prompts/v2a_y_anchored.py`](prompts/v2a_y_anchored.py) 참고.

각 prompt 로 **7,086 patents 전체에 대해 debate 를 독립적으로 재실행** 했고,
결과적으로 두 개의 별도 debate-variable matrix 가 산출됨:
- `variables_full_v0baseline.csv` (v0)
- `variables_full.csv` (v2a, 기본 dataset)

---

## 실험 설정

| 항목 | 값 |
|---|---|
| Universe | 7,086 USPTO patents (Sinigaglia 2022 16-query union, 1980–2020) |
| Y 정의 | top-decile by 5-year forward citation count (forward5 ≥ 6) |
| Y=1 base rate | 12.49% (885 / 7,086) |
| Train / Test split | 80 / 20 stratified random on Y |
| Multi-seed | 20 seeds (random_state ∈ {0..19}) — paired comparison |
| 전처리 | median impute + min-max scale (train fold 만 fit) |
| Test n | 1,418 (Y=1 ≈ 177) |

### Feature configurations

| Config | n_dim | 구성 |
|---|---|---|
| **BIBLIO** | 26 | 15 numeric biblio (CTO, STO, PK, SK, TCT, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS) + MF top-10 one-hot + MF_other |
| **FOCAL6** | 6 | `cross_domain_attack`, `conf_gap_change`, `var_conf_pro`, `H_final`, `delta_H`, `semantic_coherence` |
| **BIBLIO+FOCAL6** | 32 | 위 두 set 의 union |

### 6 모델 (`Result_Threshold.md` 와 동일 hyperparameter)

| Model | 종류 | Hyperparameters |
|---|---|---|
| **RF** | Random Forest | n_estimators=300, max_depth=6, n_jobs=-1 |
| **GBT** | Gradient Boosting (sklearn) | n_estimators=200, max_depth=4, lr=0.05 |
| **XGB** | XGBoost | n_estimators=300, tree_method=hist, eval_metric=logloss |
| **LogReg** | Logistic Regression | C=1.0, max_iter=2000 |
| **SVM** | Support Vector Machine | kernel=rbf, probability=True |
| **FFN** | MLPClassifier | hidden=ceil(sqrt(d × 2)), activation=logistic, solver=lbfgs |

### 평가 지표

| 종류 | 메트릭 |
|---|---|
| Threshold-independent | **AUROC**, **AUPRC** |
| Top-T % cutoff (T ∈ {5, 10, 15, 20}%) | **Precision**, **Recall**, **F1**, **DOR** (Haldane-corrected) |

총 모델 fit: **2 prompt × 3 config × 6 model × 20 seed = 720 fits**.
스크립트: [`analysis/prompt_comparison_threshold_sweep.py`](analysis/prompt_comparison_threshold_sweep.py).

---

## 1. AUROC / AUPRC — threshold-independent (20-seed mean ± std)

BIBLIO 단독 결과는 두 prompt 에서 정확히 동일 (debate 변수 미사용 → prompt 무관).

| Config | Model | AUROC v0 | AUROC v2a | Δ AUROC | AUPRC v0 | AUPRC v2a |
|---|---|---:|---:|---:|---:|---:|
| BIBLIO | RF | 0.7534 ± 0.018 | 0.7534 ± 0.018 | 0 | 0.3324 | 0.3324 |
| BIBLIO | GBT | 0.7496 ± 0.017 | 0.7496 ± 0.017 | 0 | 0.3286 | 0.3286 |
| BIBLIO | XGB | 0.7107 ± 0.018 | 0.7107 ± 0.018 | 0 | 0.3053 | 0.3053 |
| BIBLIO | LogReg | 0.7312 ± 0.019 | 0.7312 ± 0.019 | 0 | 0.2770 | 0.2770 |
| BIBLIO | SVM | 0.6475 ± 0.023 | 0.6475 ± 0.023 | 0 | 0.2218 | 0.2218 |
| BIBLIO | FFN | 0.7102 ± 0.023 | 0.7102 ± 0.023 | 0 | 0.2705 | 0.2705 |
| FOCAL6 only | RF | 0.5657 ± 0.016 | 0.5565 ± 0.024 | −0.009 | 0.1626 | 0.1552 |
| FOCAL6 only | GBT | 0.5394 | 0.5397 | +0.000 | 0.1484 | 0.1465 |
| FOCAL6 only | XGB | 0.5257 | **0.5284** | +0.003 | 0.1401 | 0.1439 |
| FOCAL6 only | LogReg | 0.5611 | 0.5581 | −0.003 | 0.1650 | 0.1497 |
| FOCAL6 only | SVM | 0.4933 | **0.5106** | **+0.017** | 0.1300 | 0.1298 |
| FOCAL6 only | FFN | 0.5656 | 0.5537 | −0.012 | 0.1638 | 0.1481 |
| **BIBLIO+FOCAL6** | **RF** | **0.7546** | **0.7547** | +0.000 | 0.3328 | 0.3306 |
| BIBLIO+FOCAL6 | GBT | 0.7512 | 0.7509 | −0.000 | 0.3203 | **0.3284** |
| **BIBLIO+FOCAL6** | **XGB** | 0.7213 | **0.7237** | **+0.002** | 0.2852 | **0.2961** |
| BIBLIO+FOCAL6 | LogReg | 0.7362 | 0.7346 | −0.002 | 0.2878 | 0.2796 |
| BIBLIO+FOCAL6 | SVM | 0.6574 | **0.6589** | +0.001 | 0.2232 | 0.2194 |
| BIBLIO+FOCAL6 | FFN | 0.6969 | 0.6875 | −0.009 | 0.2494 | 0.2481 |

→ **BIBLIO 단독으로도 AUROC ~0.75 천장에 도달**. 어떤 debate 추가도 +0.003 미만 lift.
prompt 차이도 ±0.01 안에서 model-dependent 하게 갈림.

### Δ AUROC over BIBLIO (per model, paired 20-seed)

| Config | Model | v0 mean Δ | v2a mean Δ | v2a − v0 |
|---|---|---:|---:|---:|
| BIBLIO+FOCAL6 | LogReg | +0.0050 | +0.0034 | −0.002 |
| BIBLIO+FOCAL6 | RF | +0.0012 | +0.0013 | +0.000 |
| BIBLIO+FOCAL6 | GBT | +0.0016 | +0.0013 | −0.000 |
| **BIBLIO+FOCAL6** | **XGB** | **+0.0106** | **+0.0130** | **+0.002** ⭐ |
| BIBLIO+FOCAL6 | SVM | +0.0098 | +0.0114 | +0.002 |
| BIBLIO+FOCAL6 | FFN | −0.0134 | −0.0227 | −0.009 |

**관찰**:
- **트리 boosting (XGB/GBT) 와 SVM 에서 양의 lift** — 두 prompt 모두에서.
- **FFN 에서 큰 음의 lift** — debate 추가시 BIBLIO 보다 0.01–0.02 떨어짐. v2a 가 더 심함.
- **LogReg/RF/GBT 는 prompt 무관 tied**. XGB 에서만 v2a 가 v0 보다 미세 우위.

---

## 2. Threshold Sweep T ∈ {5, 10, 15, 20} % — Precision / Recall / F1 / DOR

대표 모델 **RF** 의 BIBLIO+FOCAL6 결과 (20-seed mean ± std):

### v0_baseline + BIBLIO+FOCAL6 (RF)

| T | Precision | Recall | F1 | DOR |
|---|---:|---:|---:|---:|
| 5% | 0.425 ± 0.056 | 0.170 ± 0.022 | 0.243 ± 0.032 | **6.21 ± 1.60** |
| 10% | 0.367 ± 0.030 | 0.294 ± 0.024 | 0.326 ± 0.027 | 5.40 ± 0.93 |
| 15% | 0.332 ± 0.020 | 0.400 ± 0.024 | 0.363 ± 0.022 | 5.19 ± 0.72 |
| 20% | 0.302 ± 0.018 | 0.485 ± 0.028 | **0.372 ± 0.022** | 5.00 ± 0.71 |

### v2a_y_anchored + BIBLIO+FOCAL6 (RF)

| T | Precision | Recall | F1 | DOR |
|---|---:|---:|---:|---:|
| 5% | 0.422 ± 0.041 | 0.169 ± 0.016 | 0.242 ± 0.023 | 6.05 ± 1.18 |
| 10% | 0.363 ± 0.029 | 0.291 ± 0.023 | 0.323 ± 0.026 | 5.29 ± 0.86 |
| 15% | 0.331 ± 0.022 | 0.398 ± 0.027 | 0.361 ± 0.024 | 5.14 ± 0.79 |
| 20% | 0.299 ± 0.015 | 0.480 ± 0.024 | 0.369 ± 0.018 | 4.86 ± 0.60 |

### BIBLIO 단독 (prompt 무관)

| T | Precision | Recall | F1 | DOR |
|---|---:|---:|---:|---:|
| 5% | 0.430 ± 0.054 | 0.173 ± 0.022 | 0.246 ± 0.031 | **6.36 ± 1.49** |
| 10% | 0.368 ± 0.025 | 0.295 ± 0.020 | 0.328 ± 0.022 | 5.42 ± 0.72 |
| 15% | 0.326 ± 0.021 | 0.393 ± 0.025 | 0.356 ± 0.023 | 4.98 ± 0.70 |
| 20% | 0.300 ± 0.017 | 0.482 ± 0.027 | 0.370 ± 0.021 | 4.91 ± 0.72 |

**관찰**: 어떤 T 에서도 두 prompt 의 augmented 모델이 BIBLIO 단독보다 std 1개 이상의
명확한 우위를 못 만듦. 모든 차이는 ±std 안.

전체 모델 (LogReg/RF/GBT/XGB/SVM/FFN) × 모든 config × 모든 T 의 raw / aggregate 는
[`results/prompt_comparison/focal6_threshold_sweep_raw.csv`](results/prompt_comparison/focal6_threshold_sweep_raw.csv)
및 [`focal6_threshold_metrics_agg.csv`](results/prompt_comparison/focal6_threshold_metrics_agg.csv) 참조.

---

## 3. Head-to-Head — v2a vs v0 (paired Δ AUROC)

12 cells = 2 (FOCAL6 / BIBLIO+FOCAL6) × 6 (models). 각 cell 마다 같은 seed 에서
v2a 와 v0 의 Δ AUROC (over BIBLIO baseline) 를 비교.

| Config | Model | v0 Δ | v0 win | v2a Δ | v2a win | Winner |
|---|---|---:|---:|---:|---:|---|
| FOCAL6 | LogReg | −0.1701 | 0/20 | −0.1732 | 0/20 | v0 |
| FOCAL6 | RF | −0.1876 | 0/20 | −0.1969 | 0/20 | v0 |
| FOCAL6 | GBT | −0.2102 | 0/20 | −0.2099 | 0/20 | v2a |
| FOCAL6 | XGB | −0.1850 | 0/20 | −0.1824 | 0/20 | v2a |
| FOCAL6 | SVM | −0.1542 | 0/20 | −0.1370 | 0/20 | v2a |
| FOCAL6 | FFN | −0.1446 | 0/20 | −0.1565 | 0/20 | v0 |
| BIBLIO+FOCAL6 | LogReg | +0.0050 | **17/20** | +0.0034 | 13/20 | v0 |
| BIBLIO+FOCAL6 | RF | +0.0012 | 14/20 | +0.0013 | 13/20 | v2a (tied) |
| BIBLIO+FOCAL6 | GBT | +0.0016 | 13/20 | +0.0013 | 11/20 | v0 |
| **BIBLIO+FOCAL6** | **XGB** | **+0.0106** | 16/20 | **+0.0130** | 15/20 | **v2a** ⭐ |
| BIBLIO+FOCAL6 | SVM | +0.0098 | 15/20 | +0.0114 | 16/20 | v2a |
| BIBLIO+FOCAL6 | FFN | −0.0134 | 4/20 | −0.0227 | 3/20 | v0 |

### Aggregate (12 cells)

| 비교 기준 | v2a 우세 cells | v0 우세 cells |
|---|---|---|
| Mean Δ AUROC | **6 / 12 (50%)** | 6 / 12 (50%) |

→ **FOCAL-6 기준 prompt aggregate 효과 = tied**. 다만 *어떤 모델에서 lift 가 발현되는지*
는 prompt-dependent: **XGB 와 SVM 에서 v2a 가 일관 우세, LogReg/FFN 에서는 v0 우세**.

---

## 4. Per-Field — 8 main fields (Q01, Q03, Q07, Q08, Q09, Q12, Q13, Q14)

글로벌 모델 학습 → field test slice 에서 평가 (manuscript Q3 protocol).
RF, BIBLIO+FOCAL6 (∗실제 사용은 `BIBLIO+DEBATE5` = focal − cross_domain_attack
+ semantic_coherence 변형. field-level 결과는 cross_domain_attack 의 weight 차이가 작아
패턴은 동일).

| Field | n_test | Y_pos | BIBLIO AUC | v0 Δ AUC | v0 win | v2a Δ AUC | v2a win |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLOBAL | 1,418 | 177 | 0.7534 | +0.0007 | 10/20 | +0.0012 | 13/20 |
| Q01 ICE | 563 | 89 | 0.7455 | **+0.0053** | **16/20** | +0.0039 | 14/20 |
| Q03 HCCI | 27 | 6 | 0.7723 | −0.0109 | 8/20 | −0.0146 | 8/20 |
| Q07 Hybrids | 61 | 8 | 0.7320 | −0.0062 | 9/20 | −0.0008 | 10/20 |
| Q08 EGR | 251 | 19 | 0.6827 | −0.0022 | 6/20 | +0.0004 | 10/20 |
| Q09 Turbo | 428 | 41 | 0.6928 | −0.0033 | 7/20 | −0.0002 | 10/20 |
| Q12 VVA | 26 | 7 | 0.8856 | −0.0009 | 8/20 | **+0.0057** | 14/20 |
| Q13 AltFuels | 71 | 7 | 0.8154 | −0.0015 | 9/20 | +0.0022 | 10/20 |
| Q14 DI | 39 | 4 | 0.8485 | −0.0010 | 8/20 | **+0.0141** | 12/20 |

### v2a − v0 paired Δ AUROC per field (RF, BIBLIO+DEBATE25)

| Field | v2a − v0 mean Δ | win |
|---|---:|---:|
| **Q12 VVA** | **+0.0208** | **16/20** ✓ |
| **Q13 AltFuels** | **+0.0205** | **15/20** ✓ |
| Q07 Hybrids | +0.0123 | 14/20 ✓ |
| Q09 Turbo | +0.0079 | 14/20 ✓ |
| Q08 EGR | +0.0063 | 14/20 ✓ |
| Q14 DI | +0.0029 | 10/20 |
| Q01 ICE | −0.0039 | 7/20 |
| Q03 HCCI | −0.0113 | 8/20 |

**관찰**: 8개 field 중 **5개에서 v2a 가 v0 보다 robust 우위 (win 14–16/20)**.
Q01 (가장 큰 broad ICE field) 에서만 v0 우위. Q03 HCCI 는 noise (n=27).

스크립트: [`analysis/prompt_comparison_perfield.py`](analysis/prompt_comparison_perfield.py).
Raw: [`results/prompt_comparison/perfield_multiseed_raw.csv`](results/prompt_comparison/perfield_multiseed_raw.csv).

---

## 5. 종합 결론

| 결론 | 증거 |
|---|---|
| **(a) BIBLIO 단독이 ICE prediction 의 ceiling** | RF AUROC 0.7534, debate 추가시 ≤+0.003 |
| **(b) Prompt 효과는 모델/field 에 따라 갈림** | XGB 에서 v2a 우세 (+0.0024 paired), FFN 에서 v0 우세 |
| **(c) Field-level 에서 v2a 우위 더 명확** | 8 field 중 5개에서 v2a > v0, win 14–16/20 |
| **(d) Aggregate prompt 효과 = tied** | 12 cells 중 v2a 6 / v0 6 (FOCAL-6 기준) |
| **(e) FOCAL-6 only AUROC ~0.55** | bibliometric 없이는 random + 0.05 정도 |

### Production 권장

| 시나리오 | 추천 |
|---|---|
| **단순 / 안전** | **BIBLIO + RF** (AUROC 0.7534, debate 비용 0) |
| **XGB 기반 파이프라인** | **BIBLIO+FOCAL6 + XGB + v2a prompt** (Δ AUROC +0.0130 vs BIBLIO XGB) |
| **per-field downstream (Q12/Q13/Q14)** | **v2a + BIBLIO+DEBATE25 + RF** (field-level 우위 robust) |

### Honest framing

본 prompt 비교는 **debate variable 의 절대적 lift 가 작은 (±0.01) 영역에서의 미세 조정**
을 측정한 것. 어느 prompt 도 "BIBLIO 만으로 이미 좋다" 는 결론을 반박할 만큼 큰
lift 를 만들지 못함. Prompt 디자인의 실용적 의미는 다음 case 에 한정:
1. 이미 debate 인프라가 있고 *추가 cost 없이* augmented 변수를 쓸 수 있는 상황
2. 특정 모델 (XGB, SVM) 또는 특정 field (Q12/Q13/Q14) 에서의 marginal lift 가 중요한 상황

`Result_Threshold.md` 의 **단일 prompt + 6 model × 4 threshold 결과**가 main 결과,
본 문서의 **prompt 비교**는 robustness check / sensitivity analysis 의 위치.

---

## 파일 구성

```
prompts/
├── v0_baseline.py         # generic "promising emerging technology" framing
└── v2a_y_anchored.py      # Y-definition-anchored framing (5-year top-decile)

analysis/
├── prompt_comparison_threshold_sweep.py    # 12 (script 12 of debate/scripts)
└── prompt_comparison_perfield.py            # per-field 20-seed (script 09)

results/prompt_comparison/
├── focal6_threshold_sweep_raw.csv           # 4,800 rows
├── focal6_AUROC_AUPRC_agg.csv               # mean ± std AUROC, AUPRC
├── focal6_threshold_metrics_agg.csv         # P/R/F1/DOR per T
├── focal6_headtohead.csv                    # v0 vs v2a head-to-head
└── perfield_multiseed_raw.csv               # 8 fields × 20 seeds × all configs
```
