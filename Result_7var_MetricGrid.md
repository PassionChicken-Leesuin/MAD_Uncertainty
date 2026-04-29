# 7-변수 부분집합 전수탐색 — Precision / Recall / DOR @ k% Metric Grid

**연구 질문**: 사전에 선정된 7개 debate 변수 중, biblio-only 베이스라인
대비 유망기술(Y) 라벨링 예측 성능을 가장 끌어올리는 부분집합은
무엇인가? AUROC 단일 지표가 아니라, **추천 시스템 관점의 top-k%
정밀도/재현도/진단오즈비**로 평가했을 때 진정한 winner를 식별한다.

---

## 1. 실험 세팅

### 1.1 데이터 및 분할

| 항목 | 값 |
|---|---|
| 코호트 | ICE 도메인 USPTO 특허 **7,086건** (debate-completed 전체) |
| 출처 | `debate/runs/v2a_y_anchored/results/variables_full.csv` |
| 결과변수 (Y) | 5년 forward citation 수 기준 글로벌 top 10% (`forward5 ≥ q90`) |
| Y 양성률 | 12.49% (885/7,086) |
| Train/Test 분할 | **80/20 stratified random**, 5개 시드: **42, 0, 1, 2, 3** |
| Test n | 1,418건/seed (Y_pos ≈ 177) |
| Feature scaling | 시드별 train fold에서 fit한 median imputer + min-max scaler |
| MF top-10 one-hot | 시드별 train에서 빈도 top-10 제조사 산출 (no leakage) |

### 1.2 평가 모델 (6개)

| 모델 | 라이브러리 / 설정 |
|---|---|
| **RF** | sklearn `RandomForestClassifier(n_estimators=300)` |
| **LogReg** | sklearn `LogisticRegression(max_iter=2000)` (L2, C=1.0) |
| **GBT** | sklearn `HistGradientBoostingClassifier(max_iter=300)` |
| **SVM** | sklearn `SVC(kernel="rbf", probability=True)` |
| **FFN** | sklearn `MLPClassifier(hidden=(h,), activation="logistic", solver="lbfgs", alpha=0.001, max_iter=1000)` with `h = ⌈√(n_features × 2)⌉` |
| **XGB** | xgboost `XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss")` |

각 모델 `random_state` = 시드. CLAUDE.md R-4와 동일.

### 1.3 후보 변수 (7개)

사용자 그룹화 (불확실성의 4가지 면):

| 그룹 | 라벨 | 변수 | 의미 |
|---|---|---|---|
| **A** | 상호작용 불확실성 | `cross_domain_attack` | 도메인 간 페르소나의 공격 빈도 |
| **B** | 토론을 통한 변화 | `conf_gap_change` | Pro/Anti confidence gap의 수렴 |
| B | (동일) | `delta_H` | 라운드 0 → 최종 entropy 변화 |
| **C** | 개별 에이전트 내적 | `H_final` | 최종 라운드 평균 binary entropy |
| C | (동일) | `var_conf_pro` | Pro 측 confidence 분산 |
| **D** | 합의 응집도 | `semantic_coherence` | 의미적 일관성 |
| (X) | (미분류) | `prediction_volatility` | round 간 prediction 변동성 |

**탐색 공간**: 2⁷ − 1 = **127개 비공집합** + BIBLIO 베이스라인 (총 128).

### 1.4 평가 metric (12 slot)

평가 단위: **Precision / Recall / DOR @ top-k%**, `k ∈ {5, 10, 15, 20}`.
사용자 요청에 따라 `k=1%` (test 14건, 노이즈)와 `k=2%, 25%`는 제외.

| k% | test 절대 건수 | 의미 |
|---:|---:|---|
| 5% | 71 | top-shortlist (소수 정밀 추천) |
| 10% | 142 | Y rate(12.5%) 직하 — 자연 임계값 |
| 15% | 213 | Y rate 약간 초과 |
| 20% | 284 | 넓은 후보군 |

**3 metric × 4 k = 12 slot.** 각 slot에서 128 subset의 ranking 산출.

#### Metric 정의

각 모델의 predict_proba(class=1)를 내림차순 정렬, 상위 `k_abs = round(n_test × k%)` 건을 양성으로 예측. 혼동행렬 `(TP, FP, FN, TN)` 도출 후:

$$\mathrm{Precision} = \frac{TP}{TP + FP}, \quad \mathrm{Recall} = \frac{TP}{TP + FN}$$

**DOR (Diagnostic Odds Ratio, Haldane-corrected)**:

$$\mathrm{DOR} = \frac{(TP + 0.5)(TN + 0.5)}{(FP + 0.5)(FN + 0.5)}$$

Haldane 보정(+0.5)은 셀이 0이 되어 분모가 폭발하는 것을 방지.
DOR > 1이면 분류기가 무작위보다 정보적이며, 클수록 좋음.

### 1.5 집계 방식

각 (subset, model, seed, k_pct) 셀에서 metric 값 1개 산출 →
**6 모델 × 5 시드 = 30 평균값** 으로 grand mean 계산.
이 grand mean 기준으로 12 slot 각각에서 128 subset의 ranking 결정.

### 1.6 컴퓨테이션 자원

- 총 fit 수: 128 subset × 6 model × 5 seed = **3,840 model fits**
- Wall time: **61분** (Intel/Python 3.14, sklearn 1.8, xgboost 3.2)
- 실행: [analysis/exhaustive_7vars_metricgrid.py](analysis/exhaustive_7vars_metricgrid.py)

---

## 2. 핵심 결과

### 2.1 12 slot consensus — top-1 등장 빈도

| subset | size | top-1 / 12 | top-3 / 12 | top-5 / 12 |
|---|---:|---:|---:|---:|
| 🏆 **`cross_domain_attack + semantic_coherence + prediction_volatility`** | **3** | **6** | 6 | 6 |
| `cross_domain_attack + semantic_coherence` | 2 | 3 | 4 | 6 |
| `delta_H + semantic_coherence` | 2 | 3 | 3 | 3 |
| `cross_domain_attack + H_final + semantic_coherence` | 3 | 0 | 3 | 3 |
| `cross_domain_attack + conf_gap_change + prediction_volatility` | 3 | 0 | 3 | 3 |
| `cross_domain_attack + H_final + delta_H + semantic_coherence` | 4 | 0 | 3 | 3 |
| `var_conf_pro + H_final + semantic_coherence` | 3 | 0 | 3 | 3 |
| `var_conf_pro + delta_H + semantic_coherence` | 3 | 0 | 0 | 5 |

→ **Winner는 `cross_domain_attack + semantic_coherence + prediction_volatility` (size 3)**, 12개 metric slot 중 **6개 slot에서 #1**.
→ `semantic_coherence` 변수는 **모든 12 slot의 top-1 winner에 100% 포함됨**.

### 2.2 Slot별 winner 상세

| k% | metric | best subset | BIBLIO 값 | winner 값 | Δ |
|---:|---|---|---:|---:|---:|
| **5** | Precision | CA + SC | 0.39343 | 0.40235 | **+0.00892** |
| 5 | Recall | CA + SC | 0.15782 | 0.16139 | +0.00357 |
| 5 | DOR | CA + SC | 5.49501 | 5.73300 | **+0.23799** |
| **10** | Precision | **CA + SC + PV** | 0.33052 | 0.34272 | **+0.01220** |
| 10 | Recall | CA + SC + PV | 0.26516 | 0.27495 | **+0.00979** |
| 10 | DOR | CA + SC + PV | 4.49632 | 4.83608 | **+0.33976** |
| **15** | Precision | CA + SC + PV | 0.29421 | 0.30469 | **+0.01048** |
| 15 | Recall | CA + SC + PV | 0.35405 | 0.36667 | **+0.01262** |
| 15 | DOR | CA + SC + PV | 4.10900 | 4.41150 | **+0.30250** |
| **20** | Precision | delta_H + SC | 0.27430 | 0.28040 | +0.00610 |
| 20 | Recall | delta_H + SC | 0.44011 | 0.44991 | **+0.00980** |
| 20 | DOR | delta_H + SC | 4.06028 | 4.26813 | **+0.20785** |

(CA = cross_domain_attack, SC = semantic_coherence, PV = prediction_volatility)

### 2.3 k% 구간별 winner 패턴

- **k=5% (top-71, 정밀 shortlist)**: `CA + SC` (size 2)
- **k=10–15% (top-142~213, 핵심 추천 영역)**: **`CA + SC + PV` (size 3)** ← 진정한 winner
- **k=20% (top-284, 넓은 후보)**: `delta_H + SC` (size 2)

### 2.4 AUROC 기준 winner와의 비교

같은 데이터로 AUROC를 ranking criterion으로 했을 때의 결과
([analysis/outputs/forward_selection/exh7_subset_scores.csv](analysis/outputs/forward_selection/exh7_subset_scores.csv)):

| ranking criterion | size 3 winner | Δ vs BIBLIO |
|---|---|---:|
| **AUROC** (분포 전반의 분리) | `H_final + delta_H + semantic_coherence` | +0.0100 (AUROC) |
| **Precision/Recall/DOR @ k=10-15%** (top-shortlist 정밀) | `cross_domain_attack + semantic_coherence + prediction_volatility` | +0.012 ~ +0.014 |

**둘은 다르다**. `H_final + delta_H + SC`는 *전체 ranking quality* 우위,
`CA + SC + PV`는 *상위 k 추천 정밀도* 우위. 두 winner의 공통 변수는
**`semantic_coherence`** 만.

유망기술 추천 task는 **상위 k건 선택의 정밀도**가 핵심이므로, 이
연구의 진정한 winner는 **`CA + SC + PV`**.

---

## 3. 의미 해석

### 3.1 Winner 변수의 의미

| 변수 | 정의 | 사용자 그룹 |
|---|---|---|
| `cross_domain_attack` | 다른 페르소나(다른 도메인) 간 공격 액션 빈도 / 라운드 수 | A: 상호작용 불확실성 |
| `semantic_coherence` | 토론 발화의 의미적 응집도 (embedding-based) | D: 합의 응집도 |
| `prediction_volatility` | round를 거치며 prediction이 얼마나 흔들렸는가 | (미분류, 동학 계열) |

→ **세 변수 모두 "토론이 얼마나 unsettled한가"를 다른 단위로 측정**:
- *액션* 단위: 얼마나 격렬하게 다투었나 (CA)
- *의미* 단위: 최종 발언이 얼마나 응집됐나 (SC)
- *예측* 단위: 라운드 간 결론이 얼마나 흔들렸나 (PV)

### 3.2 그룹 다양성 관점

사용자 그룹 분류 기준으로 보면:
- Winner = A + D + (PV) → **"상호작용 + 합의 + 예측 변동성"**
- AUROC winner = B + C + D → "동학 + 개별내적 + 합의"
- 두 winner 모두 **`semantic_coherence` (D) 포함** ← 가장 robust

12 slot 분석에서 **`cross_domain_attack` (A) + `semantic_coherence` (D)는 9/12 slot의 top-1**에 등장. 즉 **A 그룹 (상호작용)** 변수는 AUROC에서는 redundant했지만, **top-k 정밀도에서는 핵심**.

### 3.3 왜 metric 선택이 결과를 바꾸는가

- **AUROC**: ROC 곡선 아래 면적 — 모든 임계값 평균. 분포 양 끝의 분리력 모두 반영.
- **Precision @ top-k%**: 임계값을 top-k에 고정 — *ranking head의 정밀도만* 반영.
- **DOR @ top-k%**: TP/FN과 TN/FP의 비율 — 헤드+꼬리 동시 반영.

`H_final, delta_H` (entropy 계열)는 분포 전체의 부드러운 분리에 강하지만, 상위 추천 영역에서는 `cross_domain_attack`의 강한 시그널보다 약함. 반면 `cross_domain_attack` + `prediction_volatility`는 ranking head를 sharpening 하여 top-k 정밀도를 끌어올림.

→ **task에 맞는 metric을 골라야 한다**. 유망기술 추천 = top-k 정밀도 = `CA + SC + PV` 우월.

---

## 4. 한계

1. **선택 편향**: 동일 test set으로 선택과 평가를 모두 수행. 5-시드
   평균으로 single-split luck은 줄였으나, fully held-out 검증은 아님 (CLAUDE.md R-11).
2. **Winner 우열차가 작다**: `CA + SC + PV` Δ Precision@10% = +0.012, Δ Recall@10% = +0.010. 부트스트랩 paired CI는 별도 계산하지 않음 — 통계적 유의성 검증은 follow-up 필요.
3. **127 subset 한정**: 7개 후보 중에서만 탐색. 25개 전체 debate 변수 풀에서의 globally-optimal subset과는 다를 수 있음.
4. **단일 cohort (ICE)**: AV 또는 다른 도메인에서는 winner가 다를 가능성. 본 결과는 ICE 도메인 + 5년 forward citation Y 정의에 conditional.

---

## 5. 산출물

| 파일 | 내용 |
|---|---|
| [analysis/exhaustive_7vars_metricgrid.py](analysis/exhaustive_7vars_metricgrid.py) | 실험 스크립트 |
| [analysis/outputs/forward_selection/exh7g_metric_grid.csv](analysis/outputs/forward_selection/exh7g_metric_grid.csv) | (subset × seed × model × k%) raw long-form (3,840 × 4 = 15,360 rows) |
| [analysis/outputs/forward_selection/exh7g_subset_per_slot.csv](analysis/outputs/forward_selection/exh7g_subset_per_slot.csv) | (subset × k%) 30-평균 grand mean ± std |
| [analysis/outputs/forward_selection/exh7g_winners.csv](analysis/outputs/forward_selection/exh7g_winners.csv) | 12 slot 각각의 top-5 ranking |
| [analysis/outputs/forward_selection/exh7g_consensus.csv](analysis/outputs/forward_selection/exh7g_consensus.csv) | 128 subset × top-1/3/5 등장 빈도 |
| [analysis/outputs/forward_selection/exh7g_summary.md](analysis/outputs/forward_selection/exh7g_summary.md) | auto-generated 마크다운 요약 |

---

## 6. 결론 (한줄)

> **유망기술 추천 task에서 biblio-only 베이스라인 대비 가장 큰 추가
> 효과를 내는 debate 변수 부분집합은 `cross_domain_attack +
> semantic_coherence + prediction_volatility` (size 3)이며, 12개
> metric slot 중 6개에서 winner. k=10~15% 추천 영역에서 Precision +1.0~1.2%p,
> Recall +1.0~1.3%p, DOR +0.30 향상.**
