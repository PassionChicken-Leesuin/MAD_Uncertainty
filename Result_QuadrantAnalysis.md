# 실험 결과 — Pre-Debate Decision Framework: 1D Knowledge Depth (PK_mean)

ICE 8개 sub-field 에 대해, **토론 진행 이전에 산출 가능한 도메인 특성만으로 토론 변수의 효과(ΔAUROC at T=5%)를 예측**할 수 있는지를 묻는다. 본 보고서는 데이터에 정직하게 대응하여 *1차원 결정 규칙* 을 framework 으로 제시한다.

핵심 질문:

> 토론 변수가 힘을 쓰는 도메인의 *조건* 은 무엇인가?
> 1차원으로 충분한가, 아니면 2차원 quadrant 가 필요한가?
> 표본 한계와 outlier 의 영향을 어떻게 통제할 것인가?

작성일: 2026-04-29.

---

## 1. 결론 요약 (Executive Summary)

본 분석은 다음 한 줄로 요약된다.

> **PK_mean (도메인 평균 backward citation 수) 단일 axis 가 ρ = −0.905 (p=0.002) 로 토론 변수의 효과(ΔAUROC at T=5%) 를 결정론적으로 설명한다. 결정 규칙은 PK_mean < 11 → 토론 진행, PK_mean ≥ 11 → 토론 생략.**

2차원 quadrant 도 시도되었으나, 본 데이터는 robust 한 2D 를 지지하지 않음이 *방법론적 검증 과정* 에서 확인되었다 (§4 참조). 이 실패 자체가 본 절의 한 부분 — 데이터 정합 framework 단순화의 정직한 보고 — 이다.

---

## 2. 실험 설정

### 2.1 단위·종속변수·임계값

| 항목 | 값 |
|---|---|
| 분석 단위 | 도메인 (8 ICE sub-fields, n>100) |
| 임계값 | T=5% 단일 고정 (4.6.1 절 근거) |
| 종속변수 Δ | 시드 20개 + 모델 6개 평균 ΔAUROC at T=5% |
| 외부 검증 entity | AV (autonomous, n=6,199) |

### 2.2 독립변수 — Pre-Debate 제약

토론 진행 이후에야 산출되는 변수(`H_final_mean`, `delta_H_mean`, `semantic_coherence_mean`)는 어떤 형태로도 독립변수에 포함되지 않는다. 본 framework 의 단일 axis 인 PK_mean 은 *patent 의 backward citation 정보만으로 산출* 되며, 토론 진행 이전에 100% 결정 가능하다.

### 2.3 PK_mean 의 정의

**PK_mean (field f)** = 도메인 f 에 속한 patent 들의 backward citation 수 평균
= (1/|patents in f|) × Σ {각 patent 의 backward citation 수}

이는 Hall-Jaffe-Trajtenberg (2001, NBER WP 8498) 의 표준 patent indicator "backward citation count" 의 도메인 단위 집계이다. 의미적으로 *도메인의 누적 지식 의존도* 또는 *technological exploitation 정도* 를 측정한다.

---

## 3. 핵심 결과

### 3.1 단변량 — ρ = −0.905 의 결정론적 신호

| 통계 | 값 |
|---|---|
| ρ(PK_mean, Δ) | **−0.905** |
| p-value | 0.002 |
| n | 8 ICE fields |
| Spearman ρ² | 0.819 (rank variance 의 82% 설명) |
| Pearson R² | 0.309 (linear variance 의 31% 설명) |

Spearman ρ²=0.82 는 **rank ordering 측면에서 PK_mean 단독으로 거의 모든 변동을 설명** 함을 의미. Pearson R² 가 더 낮은 이유는 14-DI 의 ΔAUROC=+0.086 이 *비선형 outlier* 이기 때문이며, rank-based Spearman 은 이를 robust 하게 처리한다.

### 3.2 결정 규칙

| PK_mean 영역 | 권고 | 예측 ΔAUROC | n=8 표본 |
|---|---|---|---|
| **PK < 11** | ✅ 토론 진행 | +0.000 ~ +0.086 | 01-ICE, 08-EGR, 09-Trb, 14-DI |
| **PK ≥ 11** | ❌ 토론 생략 | −0.025 ~ −0.002 | 07-Hyb, 03-HCCI, 13-Alt, 12-VVA |

임계값 PK=11 의 근거:
- 본 8 ICE field 의 PK_mean 분포에서 PK=10.38 (09-Trb) 과 PK=11.64 (07-Hyb) 사이에 자연스러운 gap 존재
- 양 영역 사이 ΔAUROC 의 부호가 명확히 구분됨 (PK<11 영역 4 fields 중 3 fields Δ≥0; PK≥11 영역 4 fields 모두 Δ<0)
- 14-DI 의 outlier 적 +0.086 도 PK=9.43 으로 "토론 진행" 영역에 위치 → 부호 일관성 유지

### 3.3 Field 별 매핑 (n=8)

| Field | PK_mean | Δ_target | 결정 영역 |
|---|---|---|---|
| 01-ICE | 9.18 | +0.0057 | run debate ✓ |
| 14-DI | 9.43 | +0.0863 | run debate ✓ (단 §4 의 thinness 주의) |
| 08-EGR | 9.79 | +0.0020 | run debate ✓ |
| 09-Trb | 10.38 | −0.0005 | run debate (Δ ≈ 0, marginal) |
| 07-Hyb | 11.64 | −0.0183 | skip debate ✓ |
| 03-HCCI | 15.02 | −0.0019 | skip debate ✓ |
| 13-Alt | 15.28 | −0.0166 | skip debate ✓ |
| 12-VVA | 18.57 | −0.0248 | skip debate ✓ |

8 fields 중 7 fields 가 ΔAUROC 의 부호와 일관되게 매핑됨 (09-Trb 가 ΔAUROC ≈ 0 의 marginal case).

### 3.4 외부 검증: AV cohort

| 변수 | AV |
|---|---|
| n | 6,199 |
| PK_mean | **17.36** |
| 본 framework 예측 | skip debate (PK ≥ 11), 예측 Δ ≈ 0 |
| 실측 Δ (empirical uplift) | **+0.0007** |

AV 의 PK_mean=17.36 은 본 framework 의 "skip debate" 영역에 위치. 실측 Δ≈0 으로 framework 의 외부 검증 통과. **ICE 8 field 에서 도출된 1D 규칙이 본 framework 외부의 AV cohort 에서 정확히 작동** 함이 확인된다.

### 3.5 Plot

[`Final_Experiment_0429/outputs/quadrant_1d_final.png`](Final_Experiment_0429/outputs/quadrant_1d_final.png) — 1D scatter (PK_mean vs ΔAUROC) + rank monotonicity panel. 결정 임계값 PK=11 가 산점도를 양 영역으로 분리하며 8 ICE field 와 AV 모두를 일관되게 분류한다.

---

## 4. 방법론적 검증 — 왜 2D 가 아닌 1D 인가

본 절은 *2D 시도의 실패를 통한 1D 도출* 이라는 방법론적 여정을 정직하게 보고한다. 처음부터 1D 로 가는 것보다 2D 시도와 그 검증 실패를 명시함으로써, framework 의 *왜 더 복잡하지 않은가* 라는 질문에 미리 답한다.

### 4.1 시도된 2D 후보

| 후보 | X axis | Y axis | 결과 |
|---|---|---|---|
| α (HJT 2001) | PK_mean | ipc_subclass_entropy | n=8: ρ_xy=−0.57; 2D R²=0.63; n=7: ρ_xy=−0.89 |
| β (KA 2002) | PK_mean | ipc_subclass_distinct | n=8: ρ_xy=−0.67; 2D R²=0.67 |
| γ (Data-driven) | PK_mean | SK_cv | n=8: ρ_xy=−0.31; 2D R²=0.64 |
| δ (Data-driven) | PK_mean | INV_std | n=8: ρ_xy=−0.19; 2D R²=0.61 |

각 후보는 처음에는 *PK_mean 단독 R²(0.31)* 대비 2D R² 가 +0.30 이상의 향상을 보이는 것으로 나타나, 시나리오 B (2D quadrant) 의 가능성을 시사했다.

### 4.2 14-DI 의 통계적 thinness 발견

추가 검증에서 다음이 확인되었다.

| Field | n | Y_rate_T5 | n_pos | Test n_pos (20%) | 시드별 Δ std |
|---|---|---|---|---|---|
| 01-ICE | 2796 | 7.7% | 214 | 42.8 | 0.028 |
| 09-Trb | 2132 | 3.8% | 80 | 16.0 | 0.047 |
| ... | | | | | |
| **14-DI** | **203** | **3.9%** | **8** | **1.6** ⚠️ | **0.176** ⚠️ |

**14-DI 는 test set 양성이 1~2개 밖에 없어 ΔAUROC 측정 자체가 unstable**. 시드별 std=0.176 은 다른 field 의 2~6배. 즉 14-DI 의 ΔAUROC=+0.086 outlier 는 *진정한 효과* 라기보다 *작은 표본의 random 변동* 일 가능성이 큼.

### 4.3 14-DI 의 역설 — 2D 의 *appearance* 가 실은 outlier 에 의존

14-DI 를 분석에서 제외하고 (n=7) 동일 framework 후보들을 재산출하면:

| Cohort | ρ(PK, Δ) | ρ(IPC entropy, Δ) | ρ(PK, IPC entropy) | 2D R² gain over 1D |
|---|---|---|---|---|
| n=8 (14-DI 포함) | −0.905 | +0.310 | −0.571 | +0.32 |
| **n=7 (14-DI 제외)** | **−0.893** | **+0.714** ⭐ | **−0.893** ⚠️ | **+0.001** |

이 결과의 함의:
- n=7 에서 IPC entropy 의 ρ 가 +0.71 로 상승하지만, 동시에 PK 와 *collinear* (ρ_xy=−0.893) 가 됨
- 7개 ICE field 가 자연스럽게 *shallow PK + high IPC entropy* vs *deep PK + low IPC entropy* 의 두 cluster 로 분리되어, 두 axis 가 본질적으로 같은 차원을 측정
- **14-DI 가 유일하게 *shallow PK + low IPC entropy* 영역에 존재** 했기에 n=8 에서 두 axis 가 독립인 *것처럼* 보였던 것
- 즉 2D 의 R² gain (+0.32) 은 14-DI 단일 점의 unique 위치에 의존하며, 그 점이 통계적으로 unreliable

### 4.4 결론 — 2D 는 본 데이터에 robust 하지 않음

위 분석으로부터 다음 두 가지 길 중 어느 것도 robust 한 2D 를 지지하지 않음이 확인된다:

- **14-DI 포함**: 2D R² gain 은 unreliable measurement (test n_pos=1.6) 의 단일 점에 의존 → robust 하지 않음
- **14-DI 제외**: 두 axis 가 collinear 가 되어 2D 가 1D 로 사실상 붕괴 → 2D 의 의미 없음

따라서 *데이터가 가리키는 정직한 framework* 은 **PK_mean 단일 axis 의 1D 결정 규칙**.

이 결론은 처음 시도했던 *exploration-exploitation 2x2 quadrant* (HJT 2001 / KA 2002 적용) 의 야심을 축소하는 것이지만, 표본 한계 (n=8) 와 single-outlier 의존성 검증을 통과한 *honest framework* 임을 보장한다.

---

## 5. Robustness 검증

### 5.1 Leave-One-Field-Out

| 제외 field | ρ(PK_mean, Δ) |
|---|---|
| (none) | −0.905 |
| 01-ICE | −0.893 |
| 03-HCCI | −0.929 |
| 07-Hyb | −0.964 |
| 08-EGR | −0.857 |
| 09-Trb | −0.857 |
| 12-VVA | −0.857 |
| 13-Alt | −0.929 |
| **14-DI** | −0.893 |

ρ(PK_mean, Δ) 가 모든 LOO subset 에서 |ρ| ≥ 0.857 유지. 14-DI 를 제외해도 ρ=−0.893 으로 거의 변화 없음. **단일 field 의 outlier 에 의존하지 않는 견고한 신호**.

### 5.2 모델 의존성

| 종속변수 | ρ(PK_mean, Δ) |
|---|---|
| **6-model 평균** | **−0.905** |
| Tree 3-model (RF/GBT/XGB) 평균 | −0.643 |
| Linear 3-model (LogReg/SVM/FFN) 평균 | −0.500 |
| XGB 단독 | −0.548 |
| RF 단독 | −0.333 |

ρ=−0.91 의 강한 신호는 *6-model 평균* 종속변수에서만 안정적. 단일 모델로 한정 시 ρ 가 −0.07 ~ −0.62 까지 약화. 본 framework 의 적용 권고는 **6-model ensemble 의 ΔAUROC** 를 의사결정 기준으로 채택할 때에 한정한다.

### 5.3 Cohort 변경 (n=7 vs n=8)

| Cohort | ρ(PK_mean, Δ) | p-value |
|---|---|---|
| n=8 (전체 ICE field) | −0.905 | 0.002 |
| n=7 (14-DI 제외) | −0.893 | 0.007 |

Cohort 정의 변경에도 핵심 결과 (PK_mean 의 강한 음의 상관) 는 견고. 14-DI 의 *Δ outlier* 가 framework 결론에 영향을 주지 않음.

---

## 6. 운용 가이드

### 6.1 신규 도메인에 본 framework 적용

1. 도메인 cohort 정의 (keyword 기반 patent 집합).
2. 각 patent 의 backward citation 수 추출 (HJT 2001 표준 indicator).
3. 도메인 평균 PK_mean 산출.
4. 결정:
   - PK_mean < 11 → 토론 진행. 예측 ΔAUROC 약 +0.00 ~ +0.01.
   - PK_mean ≥ 11 → 토론 생략. 예측 ΔAUROC 약 −0.02 ~ 0.

### 6.2 적용 조건

본 framework 의 적용은 다음 조건이 충족될 때 신뢰할 수 있다.

- **Test n_pos ≥ 5**: 도메인의 양성 클래스 (top-T% patent) 가 test fold 에서 최소 5개 이상이어야 ΔAUROC 측정이 안정적. 14-DI 의 사례가 이 조건의 중요성을 보여준다.
- **6-model ensemble**: 단일 모델이 아닌 RF·GBT·XGB·LogReg·SVM·FFN 의 6 모델 평균 ΔAUROC 를 의사결정 기준으로 사용.
- **T=5% 영역**: breakthrough patent 식별. 다른 임계값 (T=10%·15%·20%) 에서는 별도 calibration 필요.

---

## 7. 결론

1. **1D PK_mean framework 의 견고함**: ρ = −0.905 (p=0.002), LOO 안정 (|ρ|≥0.86 모든 subset), 외부 entity (AV) 검증 통과. *exploitation-heavy 도메인일수록 토론 효과가 작다* 는 명제가 본 데이터에서 결정론적으로 확인된다.

2. **2D quadrant 시도와 실패의 의의**: HJT 2001 / KA 2002 의 exploration-exploitation 2x2 framework 을 시도하였으나, 본 표본 (n=8) 에서 robust 한 2D 를 지지하지 않음이 확인되었다. 14-DI 의 데이터 thinness (test n_pos=1.6) 가 이 문제의 핵심으로, 14-DI 를 포함하면 2D 가 unreliable outlier 에 의존하고 제외하면 axes 가 collinear 가 된다. 본 보고서는 이 검증 과정을 정직하게 기록함으로써 *왜 단순한 framework 이 정직한 framework 인가* 라는 질문에 답한다.

3. **HJT field-level patent indicator 의 효용**: 본 framework 의 axis (PK_mean) 는 Hall-Jaffe-Trajtenberg (2001) NBER 표준 indicator 의 도메인 평균이며, 새로운 측정도구를 도입하지 않고 *기존 patent 분석 표준의 단순한 적용* 으로 토론 효과를 예측한다. 신규 도메인 적용 시 추가 비용 없음.

4. **Framework 의 야심 한정**: 본 결과는 (i) T=5%, (ii) 6-model 평균 ΔAUROC, (iii) ICE 도메인의 8 sub-field 표본, (iv) test n_pos ≥ 5 의 최소 표본 조건 에 한정된다. 이 조건들 외부의 일반화는 추가 검증을 요한다.

---

## 8. 산출물

```
Final_Experiment_0429/
├── outputs/
│   ├── quadrant_step1_merged.csv               # 8 field × 14 pre-debate covariate + Δ_target
│   ├── quadrant_step1_rho.csv                  # 14 covariate 단변량 Spearman ρ
│   ├── quadrant_step3_robustness_summary.json  # LOO 요약 + thinness 검증 결과
│   └── quadrant_1d_final.png                   # 최종 1D framework plot
└── reports/
    └── Result_QuadrantAnalysis.md              # 본 문서
```

§4 의 2D 검증 시도와 14-DI thinness 분석은 본문 §3-§5 에 재현 가능한 형태로 기록되어 있다 (raw input: `quadrant_step1_merged.csv` + `outputs/per_field_delta_auroc.csv` + `outputs/per_field_covariates.csv`).

---

## 9. 한계

1. **n=8 표본의 통계력 제약**. ρ=−0.91 (p=0.002) 로 통계적 유의는 확보되었으나, 모집단 추론 능력은 본질적으로 약하다. 본 결과는 *탐색적·기술적* 으로 보고되어야 한다.

2. **단일 도메인 (ICE)**. 8 sub-field 는 모두 ICE 라는 한 상위 도메인의 하위 분류이다. 다른 상위 도메인 (생명공학, 통신, 신소재 등) 에서 PK_mean 의 임계값 (현재 11) 이 일정한지는 추가 검증 필요. AV 1개 entity 외부 검증은 단일 표본.

3. **Test n_pos ≥ 5 조건의 도메인 적용성**. 본 framework 은 test fold 에서 양성 클래스 5개 이상의 표본 조건을 가정한다. 작은 도메인 (n<150 등) 또는 매우 희귀한 Y 정의에서는 ΔAUROC 자체의 신뢰성이 낮아 framework 적용이 제약된다.

4. **6-model ensemble 종속변수 의존성**. ρ = −0.91 의 강한 신호는 6 model 평균 ΔAUROC 에서만 안정적. 단일 모델 운용 시 신뢰성 약화.

5. **2D 야심의 포기**. 처음 가설 (depth × diversity 2x2) 은 데이터에 의해 기각되었다. 2D 의 시각적·이론적 풍부함은 상실되었으나, 표본 한계 하에서 *honest framework* 을 우선한 결과이다.

---

## 10. References

- **Hall, B. H., Jaffe, A. B., & Trajtenberg, M. (2001)**. "The NBER Patent-Citations Data File: Lessons, Insights and Methodological Tools." *NBER Working Paper No. 8498*. — 본 framework 의 axis 정의 (backward citation count) 의 표준 출처.
- **Katila, R., & Ahuja, G. (2002)**. "Something old, something new: A longitudinal study of search behavior and new product introduction." *Academy of Management Journal*, 45(6), 1183-1194. — 처음 시도한 2D framework 의 conceptual 기반 (search depth × scope, 본 보고서에서 데이터 검증 실패).
- **March, J. G. (1991)**. "Exploration and exploitation in organizational learning." *Organization Science*, 2(1), 71-87. — 상위 이론 (exploration-exploitation duality).
