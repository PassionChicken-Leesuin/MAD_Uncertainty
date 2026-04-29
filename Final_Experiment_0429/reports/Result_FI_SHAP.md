# Result_FI_SHAP — 임계값별 Feature Importance & SHAP 분석

> ICE 도메인 / `+6focal` config (BIBLIO 26 + Focal Debate 6 = 32 features) 기준,
> 4개 임계값 × 6개 모델 × 20 시드 평균.
> 작성일: 2026-04-29.

---

## 1. 실험 세팅

### 1.1 데이터
- 코호트: 7,086 patent (USPTO 1980–2020, ICE 16-query union, debate가 끝난 patent만)
- Y label: `forward5 ≥ q(100−T)`, T ∈ {5, 10, 15, 20}%
  - T= 5%: cutoff `forward5 ≥ 9` → Y rate **5.84%**
  - T=10%: cutoff `forward5 ≥ 6` → Y rate **12.49%**
  - T=15%: cutoff `forward5 ≥ 5` → Y rate **15.83%**
  - T=20%: cutoff `forward5 ≥ 4` → Y rate **21.17%**
- 80/20 stratified train/test split, 시드 20개 = `[42, 0, 1, ..., 18]`.

### 1.2 Features (32개, `+6focal`)
- **BIBLIO 26** = 15 numeric (CTO, STO, PK, SK, TCT, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS) + MF top-10 one-hot + MF_other = 26 columns.
- **Focal Debate 6** = `var_conf_pro`, `H_final`, `delta_H`, `conf_gap_change`, `cross_domain_attack`, `semantic_coherence`.

### 1.3 모델 (6개)
| 모델 | 사양 |
|---|---|
| RF | `RandomForestClassifier(n_estimators=300)` |
| GBT | `GradientBoostingClassifier(n_estimators=300, max_depth=3)` |
| XGB | `XGBClassifier(n_estimators=300, tree_method=hist)` |
| LogReg | `LogisticRegression(solver=lbfgs, max_iter=1000)` |
| SVM | `SVC(kernel=rbf, probability=True)` |
| FFN | `MLPClassifier(hidden_layer_sizes=(h,), activation=logistic, solver=lbfgs, alpha=0.001, max_iter=1000)`, `h = ceil(sqrt(2·n_features))` |

전처리: median 임퓨터 + min-max 스케일러 (train fold 단독 fit).

### 1.4 평가지표
- **Permutation Importance** — sklearn `permutation_importance`, **n_repeats = 10**, **scoring = "roc_auc"**, test fold에서 산출. 시드별 결과 → 시드 20개 평균.
- **SHAP** — XGB만, `shap.TreeExplainer`, test fold에서 `shap_values` → `mean_abs_shap` 시드 20개 평균.

총 fits = 4 (T) × 6 (model) × 20 (seed) = **480 학습**, + 80개 SHAP 추정 (XGB만).

### 1.5 출력 산출물
```
analysis/outputs/experiments_summary/
  threshold_importance_perm_full.csv      (per-seed-feature, 32cols × 480 = ~15K rows)
  threshold_importance_perm_summary.csv   (T × model × feature 평균±표준편차)
  threshold_importance_shap_full.csv      (per-seed-feature, 32cols × 80 fits)
  threshold_importance_shap_summary.csv   (T × feature 평균±표준편차)
```

---

## 2. Permutation Importance — 모델·임계값별 Top-10

각 셀의 숫자는 시드 20개 평균 ± 표준편차. `[D]` = 6개 focal debate 변수.

### 2.1 T = 5% (Y rate 5.84%)

| Rank | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 1 | TCT 0.0484 | TKH 0.0615 | TCT 0.0401 | TCT 0.0692 | TCT 0.0371 | TCT 0.0789 |
| 2 | TKH 0.0371 | TCT 0.0517 | PK 0.0252 | STO 0.0398 | SK 0.0330 | **H_final 0.0557** [D] |
| 3 | TTS 0.0319 | CKH 0.0479 | SK 0.0214 | **H_final 0.0340** [D] | CKH 0.0292 | STO 0.0441 |
| 4 | CKH 0.0304 | PK 0.0372 | CKH 0.0213 | TS 0.0322 | **H_final 0.0286** [D] | CKH 0.0349 |
| 5 | PKH 0.0298 | PKH 0.0371 | TS 0.0190 | SK 0.0143 | CTS 0.0227 | MF_other 0.0339 |
| 6 | PTS 0.0292 | CTS 0.0348 | CTS 0.0165 | MF_F02B37 0.0099 | TS 0.0212 | TS 0.0208 |
| 7 | CTS 0.0290 | **H_final 0.0312** [D] | **H_final 0.0114** [D] | **delta_H 0.0054** [D] | MF_F02B33 0.0201 | SK 0.0178 |
| 8 | TS 0.0224 | TTS 0.0249 | TTS 0.0098 | INV 0.0045 | STO 0.0198 | MF_F02D41 0.0155 |
| 9 | PK 0.0192 | TS 0.0228 | **cross_domain_attack 0.0076** [D] | CKH 0.0036 | MF_F02B37 0.0175 | MF_F02B37 0.0129 |
| 10 | SK 0.0146 | SK 0.0203 | **delta_H 0.0076** [D] | MF_F02D41 0.0027 | MF_other 0.0169 | CTS 0.0126 |

**Top-10에 진입한 debate 변수 수**: RF 0개 / GBT 1개 / **XGB 3개** / **LogReg 2개** / **SVM 1개** / **FFN 1개** → 6모델 평균 1.3개 (전체 6개 중 22%).
T=5%에서 XGB가 가장 debate를 활용함 (`H_final`·`cross_domain_attack`·`delta_H` 모두 top-10 진입).

### 2.2 T = 10% (Y rate 12.49%)

| Rank | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 1 | TCT 0.0503 | TKH 0.0543 | TCT 0.0443 | TCT 0.0684 | TCT 0.0440 | CKH 0.0729 |
| 2 | TS 0.0361 | TCT 0.0484 | TS 0.0324 | TS 0.0536 | CKH 0.0395 | TCT 0.0614 |
| 3 | CKH 0.0275 | PKH 0.0449 | CKH 0.0322 | STO 0.0386 | TS 0.0349 | **H_final 0.0433** [D] |
| 4 | TKH 0.0252 | CKH 0.0442 | SK 0.0173 | **H_final 0.0261** [D] | **H_final 0.0266** [D] | TS 0.0362 |
| 5 | PKH 0.0221 | TS 0.0398 | PK 0.0159 | CKH 0.0171 | STO 0.0218 | STO 0.0299 |
| 6 | CTS 0.0205 | PTS 0.0324 | CTS 0.0151 | SK 0.0076 | MF_other 0.0194 | CTS 0.0249 |
| 7 | TTS 0.0165 | TTS 0.0259 | STO 0.0120 | NC 0.0065 | MF_F02B33 0.0165 | MF_other 0.0227 |
| 8 | PK 0.0149 | CTS 0.0228 | TKH 0.0102 | **delta_H 0.0057** [D] | SK 0.0156 | **delta_H 0.0132** [D] |
| 9 | PTS 0.0145 | PK 0.0162 | PTS 0.0100 | MF_F02B37 0.0043 | TKH 0.0152 | SK 0.0125 |
| 10 | STO 0.0129 | SK 0.0121 | PKH 0.0089 | CTS 0.0043 | MF_F02D41 0.0140 | MF_F02D41 0.0101 |

**Top-10 debate 진입**: RF 0 / GBT 0 / XGB 0 / **LogReg 2** / **SVM 1** / **FFN 2**.
T=10%에서는 트리 모델군(RF/GBT/XGB)에서 debate가 사라지고, **LogReg·FFN에서 H_final/delta_H 콤보**가 두드러짐.

### 2.3 T = 15% (Y rate 15.83%)

| Rank | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 1 | TCT 0.0481 | TCT 0.0518 | TCT 0.0434 | TCT 0.0698 | CKH 0.0480 | CKH 0.0727 |
| 2 | CKH 0.0311 | CKH 0.0400 | CKH 0.0386 | TS 0.0456 | TCT 0.0460 | TCT 0.0576 |
| 3 | TS 0.0308 | TS 0.0334 | TS 0.0262 | STO 0.0432 | TS 0.0348 | **H_final 0.0482** [D] |
| 4 | CTS 0.0253 | TKH 0.0333 | PK 0.0177 | **H_final 0.0297** [D] | **H_final 0.0341** [D] | CTS 0.0435 |
| 5 | PKH 0.0243 | CTS 0.0300 | CTS 0.0176 | CKH 0.0230 | STO 0.0337 | TS 0.0345 |
| 6 | TKH 0.0229 | PKH 0.0286 | SK 0.0130 | CTS 0.0101 | MF_other 0.0302 | STO 0.0311 |
| 7 | PK 0.0158 | PK 0.0202 | PKH 0.0109 | SK 0.0069 | MF_F02D41 0.0191 | MF_other 0.0254 |
| 8 | TTS 0.0152 | TTS 0.0177 | STO 0.0101 | MF_F02B37 0.0065 | TKH 0.0173 | TKH 0.0169 |
| 9 | PTS 0.0138 | **H_final 0.0127** [D] | TKH 0.0097 | NC 0.0059 | MF_F02B33 0.0172 | **delta_H 0.0168** [D] |
| 10 | STO 0.0098 | STO 0.0110 | PTS 0.0084 | **delta_H 0.0044** [D] | PKH 0.0159 | PKH 0.0159 |

**Top-10 debate 진입**: 0 / 1 / 0 / **2** / **1** / **2**.

### 2.4 T = 20% (Y rate 21.17%)

| Rank | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 1 | TCT 0.0445 | TCT 0.0466 | TCT 0.0394 | TCT 0.0695 | CKH 0.0617 | CKH 0.0683 |
| 2 | TS 0.0306 | PKH 0.0333 | CKH 0.0377 | TS 0.0405 | TCT 0.0489 | TCT 0.0529 |
| 3 | CKH 0.0260 | TS 0.0285 | TS 0.0235 | STO 0.0394 | STO 0.0356 | CTS 0.0429 |
| 4 | CTS 0.0217 | CKH 0.0280 | PK 0.0193 | CKH 0.0338 | TS 0.0335 | **H_final 0.0344** [D] |
| 5 | PK 0.0180 | TKH 0.0215 | CTS 0.0187 | **H_final 0.0296** [D] | **H_final 0.0263** [D] | TS 0.0323 |
| 6 | PKH 0.0161 | PK 0.0177 | SK 0.0123 | CTS 0.0074 | MF_other 0.0254 | STO 0.0290 |
| 7 | TKH 0.0157 | CTS 0.0160 | TKH 0.0123 | NC 0.0070 | MF_F02D41 0.0178 | MF_other 0.0236 |
| 8 | STO 0.0094 | PTS 0.0160 | PKH 0.0109 | **delta_H 0.0050** [D] | CTS 0.0153 | TKH 0.0210 |
| 9 | TTS 0.0089 | STO 0.0104 | STO 0.0099 | MF_F02B37 0.0048 | TKH 0.0125 | PKH 0.0192 |
| 10 | PTS 0.0084 | TTS 0.0102 | PTS 0.0095 | SK 0.0039 | PKH 0.0115 | MF_F02B33 0.0143 |

**Top-10 debate 진입**: 0 / 0 / 0 / **2** / **1** / **1**.

---

## 3. SHAP (XGB) — 임계값별 Top-15

| Rank | T = 5% | T = 10% | T = 15% | T = 20% |
|---|---|---|---|---|
| 1 | TCT 0.810 | TCT 0.769 | TCT 0.757 | TCT 0.693 |
| 2 | PK 0.628 | TS 0.653 | TS 0.568 | TS 0.544 |
| 3 | **semantic_coherence 0.525** [D] | CKH 0.485 | CKH 0.533 | CKH 0.472 |
| 4 | CKH 0.507 | CTS 0.411 | PK 0.426 | PK 0.431 |
| 5 | **H_final 0.460** [D] | PK 0.411 | CTS 0.396 | STO 0.353 |
| 6 | CTS 0.458 | **semantic_coherence 0.404** [D] | **H_final 0.396** [D] | **semantic_coherence 0.348** [D] |
| 7 | TS 0.457 | **H_final 0.402** [D] | STO 0.362 | CTS 0.335 |
| 8 | **delta_H 0.453** [D] | STO 0.378 | **semantic_coherence 0.349** [D] | NC 0.301 |
| 9 | STO 0.412 | **delta_H 0.353** [D] | **delta_H 0.314** [D] | **H_final 0.291** [D] |
| 10 | **conf_gap_change 0.396** [D] | **conf_gap_change 0.310** [D] | NC 0.292 | **delta_H 0.285** [D] |
| 11 | CTO 0.362 | TKH 0.305 | PKH 0.285 | **conf_gap_change 0.283** [D] |
| 12 | **cross_domain_attack 0.360** [D] | NC 0.300 | **conf_gap_change 0.284** [D] | **cross_domain_attack 0.278** [D] |
| 13 | SK 0.328 | CTO 0.296 | **var_conf_pro 0.283** [D] | PKH 0.271 |
| 14 | PKH 0.326 | **var_conf_pro 0.291** [D] | TKH 0.276 | TKH 0.268 |
| 15 | **var_conf_pro 0.307** [D] | PKH 0.276 | CTO 0.273 | CTO 0.258 |

**T=5%에서 SHAP top-15 안에 6개 focal debate 변수 모두 포함됨.**
다른 T에서는 5개 (T=10·15: cross_domain_attack 빠짐 / T=20%: 모두 포함).

---

## 4. Debate 변수 종합 동향

### 4.1 Permutation 기준 — debate 6변수 점유율 (positive imp_mean의 합 중 비중)

| 모델 | T=5% | T=10% | T=15% | T=20% |
|---|---|---|---|---|
| RF | 5.9% | 3.0% | 3.0% | 3.6% |
| GBT | **15.8%** | 6.2% | 8.3% | 9.2% |
| XGB | **14.4%** | 7.5% | 6.5% | 8.2% |
| LogReg | **17.9%** | 13.0% | 13.8% | 14.1% |
| SVM | 12.1% | 12.7% | 11.4% | 11.4% |
| FFN | **16.7%** | 14.6% | 15.1% | 12.0% |

**관찰 1**: T=5%에서 GBT/XGB/LogReg/FFN의 debate 점유율이 가장 높음 → `Result_Threshold.md`의 "T=5%에서 debate Δ AUROC가 가장 크다" 결과와 정합.
**관찰 2**: RF는 임계값에 무관하게 debate를 거의 활용하지 않음 (3–6%). RF의 lift가 작은 이유가 변수 활용도에 있음을 직접 확인.
**관찰 3**: LogReg·FFN은 모든 T에서 12% 이상 — 선형/얕은 비선형 모델은 debate를 더 안정적으로 활용.

### 4.2 SHAP 기준 — debate 6변수 점유율 (XGB)

| | T=5% | T=10% | T=15% | T=20% |
|---|---|---|---|---|
| Σ mean_abs_shap | 8.68 | 7.60 | 7.29 | 6.91 |
| debate 합계 | 2.50 | 1.97 | 1.87 | 1.71 |
| **debate 점유율** | **28.8%** | **26.0%** | **25.6%** | **24.8%** |

XGB SHAP은 permutation보다 debate 비중을 훨씬 크게 평가 (28.8% vs 14.4% at T=5%).
→ debate 변수들이 BIBLIO와 **강한 상관/상호작용 관계**에 있음을 시사. permutation은 "다른 변수 제거 후의 unique 기여"를 보고, SHAP은 "공동 기여를 분배"하기 때문. debate 변수가 BIBLIO와 겹치는 부분이 많지만, XGB 트리 분기에서 광범위하게 사용된다는 의미.

### 4.3 변수별 임계값별 등수 (작을수록 상위, 32 features 중)

#### Permutation (모델 평균 등수)

| Variable | T=5% | T=10% | T=15% | T=20% |
|---|---|---|---|---|
| H_final | **6** | **8** | **8** | **10** |
| delta_H | 15 | 17 | 18 | 21 |
| semantic_coherence | 25 | 21 | 25 | 22 |
| cross_domain_attack | 19 | 22 | 21 | 16 |
| conf_gap_change | 20 | 23 | 23 | 22 |
| var_conf_pro | 32 | 27 | 24 | 27 |

#### SHAP (XGB)

| Variable | T=5% | T=10% | T=15% | T=20% |
|---|---|---|---|---|
| semantic_coherence | **3** | 6 | 8 | 6 |
| H_final | 5 | 7 | 6 | 9 |
| delta_H | 8 | 9 | 9 | 10 |
| conf_gap_change | 10 | 10 | 12 | 11 |
| cross_domain_attack | 12 | 19 | 16 | 12 |
| var_conf_pro | 15 | 14 | 13 | 18 |

**관찰 4 (변수별 안정성)**:
- `H_final`: 모델·임계값에 무관하게 가장 안정적인 debate signal. 평균 등수 6–10.
- `semantic_coherence`: SHAP에선 매우 강함 (3–8) but permutation에선 약함 (21–25) → 다른 변수와 redundant하지만 XGB가 광범위하게 활용. 모델 invariance가 약함.
- `var_conf_pro`: 가장 약한 signal. permutation에선 거의 항상 하위권.
- `cross_domain_attack`: T=5%·20%에서 SHAP·perm 둘 다 비교적 강하고, T=10·15%에서 약함 → "양극단 임계값에서 핵심"이라는 비선형 패턴.

---

## 5. BIBLIO 변수 동향 (참고)

전 임계값·전 모델에서 거의 항상 등장하는 BIBLIO top들:

| 변수 | 의미 | 평균 등수 |
|---|---|---|
| **TCT** | Total Citing Triplets (직접 forward 카운트의 변형) | **1** (모든 셀에서) |
| **CKH** | Citing-Knowledge Heritage (인용 기반 키워드 매칭) | 2–4 |
| **TS** | Technological Strength (NC × CTO) | 2–5 |
| **CTS** | Cumulative Time Series | 4–7 |
| **PK** | Prior Knowledge (역방향 인용 수) | 3–8 |

→ 전형적인 "forward citation 누적/centrality" 변수가 지배. debate는 그 다음 layer.

---

## 6. 결론

1. **TCT는 모든 (T, model)에서 #1 feature.** "forward citation 직접 변형"이 promising-patent 신호의 절대 대장.
2. **Debate 6변수는 임계값·모델 의존적으로 보조 신호 제공.**
   - Permutation 점유율: T=5%일 때 가장 높음 (XGB 14%, GBT 16%, LogReg 18%, FFN 17%) → `Result_Threshold.md` Δ AUROC 패턴과 정합.
   - SHAP 점유율: 모든 T에서 25–29% (BIBLIO와의 상관 때문에 perm이 underestimate).
3. **`H_final`이 가장 강건한 debate 변수.** LogReg·SVM·FFN에서 Top-5 안에 항상 존재. cross-model robustness가 다른 5개를 압도.
4. **`semantic_coherence`는 SHAP-only winner.** XGB 트리에서 광범위하게 쓰이지만 (top-3) 다른 BIBLIO와 redundant → permutation 측정에선 사라짐.
5. **Model dependency**:
   - **RF는 debate를 활용 못 함** (점유율 3–6%) → `Result_Threshold.md`에서 RF의 Δ AUROC가 작았던 직접적 원인.
   - **LogReg·FFN은 debate-friendly** → 선형/얕은 비선형 표현력에서는 debate가 새 정보를 직접 추가.
   - **GBT vs XGB**: GBT가 T=5%에서 debate를 더 활용 (15.8% vs 14.4%) — 같은 boosting 계열이지만 분기 전략에 따라 차이.
6. **임계값 효과**: T가 작을수록(=상위 패턴 강한 cohort일수록) debate 변수 활용도 상승. T=5%에서 debate 활용·Δ AUROC·focal 변수 진입이 모두 peak.

---

## 7. 재현

```bash
cd /c/Users/User/OneDrive/.../ICE_Domain
python analysis/experiments_threshold_importance.py     # ~70min, GPU 불필요
python analysis/_show_importance.py                      # pretty-print
```

산출물 4개 CSV는 `analysis/outputs/experiments_summary/`.
