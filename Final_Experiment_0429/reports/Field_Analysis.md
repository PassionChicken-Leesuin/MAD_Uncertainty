# 실험 결과 — Per-Field 분석 (Debate 변수의 도메인별 효과)

ICE 도메인 16개 ICE field 중 **n>100인 8개 field**에 대해, debate 변수
(BIBLIO+6focal) 추가가 bibliometric-only baseline 대비 분류 성능에 어떤 효과를
주는지, 그리고 그 효과가 field 특성에 따라 어떻게 달라지는지 분석.

핵심 질문:

> Debate 변수의 추가 효과는 모든 field에서 동일한가?
> 효과가 큰 field와 작은 field를 구분 짓는 field-level 특성은 무엇인가?
> 어떤 임계값(T)에서 어떤 field가 가장 큰 lift를 받는가?

---

## 1. 실험 설정

### 1.1 코호트 및 Field 정의

| 항목 | 값 |
|---|---|
| Universe | 7,086 USPTO patents (Sinigaglia 2022 16-query union, 1980–2020), `mean_conf_pro` 결측 제거 후 |
| 분석 대상 | n>100인 8개 ICE field |
| Field 멤버십 정의 | **Explode-style** — 한 patent이 `08,09`처럼 다중 소속이면 양쪽 field 실험에 모두 등장 |
| Y 라벨 | **Global** top-T% (전체 7,086 코호트 기준 percentile, 모든 field에서 동일 cutoff 사용) |

#### 8개 Field

| code | name | n (explode) | Y_rate @ T=10% |
|---|---|---|---|
| 01 | Internal Combustion Engine | 2,796 | 16.06% |
| 03 | HCCI | 130 | 21.54% |
| 07 | Hybrids | 306 | 12.75% |
| 08 | EGR | 1,277 | 8.07% |
| 09 | Super-/Turbocharging | 2,132 | 9.43% |
| 12 | Variable Valve Actuation | 133 | 24.81% |
| 13 | Alternative Fuels | 352 | 9.66% |
| 14 | Direct Injection | 203 | 8.87% |

Field별 Y rate 편차가 크다 (3.94% ~ 32.33%) — global Y를 사용했으므로
**불균형 자체가 field 간 차이의 일부**다.

### 1.2 Sweep 차원

| 차원 | 값 | 개수 |
|---|---|---|
| Field | 01 / 03 / 07 / 08 / 09 / 12 / 13 / 14 | 8 |
| Threshold T | 5% / 10% / 15% / 20% | 4 |
| Seed | [42, 0, 1, 2, …, 18] | 20 |
| Configuration | BIBLIO (26 feat) / BIBLIO+6focal (32 feat) | 2 |
| Model | RF / GBT / XGB / LogReg / SVM / FFN | 6 |
| **총 fit 수** | | **7,680** |

### 1.3 Train/Test split

각 (field, T, seed)별로 **per-field 내부 stratified 80/20 split** on Y_T.
즉, 각 field의 실험은 자기 field 내부 patent으로만 train/test하며, seed별로
split이 달라진다 (총 20개 split per field).

전처리: median impute + min-max scaling, train fold 단독 fit.

### 1.4 변수 구성

**BIBLIO (26 features)**

| 그룹 | 변수 |
|---|---|
| Numeric (15) | CTO, STO, PK, SK, TCT, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS |
| Categorical | MF top-10 one-hot + MF_other (per-field train fold 기준 top-10 재계산) |

**+6focal Debate (32 features = BIBLIO + 6)**

prior `Result_DebateBestSubset.md` 분석에서 가장 일관된 lift를 준 6개:
`var_conf_pro, H_final, delta_H, conf_gap_change, cross_domain_attack, semantic_coherence`

### 1.5 모델 사양

| Model | 종류 | Hyperparameters |
|---|---|---|
| **RF** | Random Forest (bagging) | `n_estimators=300, n_jobs=-1` |
| **GBT** | Gradient Boosting (sklearn) | `n_estimators=300, max_depth=3` |
| **XGB** | XGBoost (boosting + regularization) | `n_estimators=300, tree_method=hist, eval_metric=logloss` |
| **LogReg** | Logistic Regression (linear) | `solver=lbfgs, max_iter=1000` |
| **SVM** | RBF Support Vector Machine | `kernel=rbf, probability=True` |
| **FFN** | Feedforward MLP | `hidden=ceil(√(n_features × 2)), logistic, solver=lbfgs, alpha=0.001` |

---

## 2. 평가 메트릭

각 (field, T, seed, config, model) 셀에서 test fold에 대해 다음 7개 metric 계산:

| Metric | 정의 | Threshold-free? |
|---|---|---|
| **AUROC** | Receiver Operating Characteristic 곡선 아래 면적. 임의 양/음성 쌍에서 양성 score가 더 높을 확률. | ✓ |
| **AUPRC** | Precision-Recall 곡선 아래 면적 (= average precision). 불균형 데이터에서 AUROC보다 민감. | ✓ |
| **Precision** | top-T% threshold 적용 시 TP / (TP+FP) | × |
| **Recall** | top-T% threshold 적용 시 TP / (TP+FN) | × |
| **F1** | Precision과 Recall의 조화평균 | × |
| **Accuracy** | (TP+TN) / N | × |
| **DOR** | (Diagnostic Odds Ratio, Haldane corrected) `((TP+0.5)·(TN+0.5)) / ((FP+0.5)·(FN+0.5))` — 진단 odds의 비율, 0/0 division 방지 위해 0.5 보정 | × |

Top-T% threshold: `score >= quantile(score, 1−T/100)` — predict 양성 비율을
test fold의 Y rate가 아니라 T에 맞추는 방식 (전체 sweep 통일).

### 2.1 Δ 정의

각 (field, T, model)에서:
- $\Delta\text{AUROC} = \text{AUROC}_{+6\text{focal}} - \text{AUROC}_{\text{BIBLIO}}$ (paired by seed)
- $\Delta\text{DOR} = \text{DOR}_{+6\text{focal}} - \text{DOR}_{\text{BIBLIO}}$

20 seed에 걸쳐 paired difference 계산.

### 2.2 통계 검정

각 (field, T, model) 셀:
- **paired-t test** (one-sample t on per-seed differences)
- 셀당 검정 1회 → T별로 8 field × 6 model = 48 검정
- **BH-FDR** (Benjamini-Hochberg) 보정 per T (검정 family = 48 cells)
- 보고: `p_paired` (raw), `p_bh` (adjusted), `sig_bh_05`, `sig_bh_10`

### 2.3 Why-분석 (Spearman ρ)

Field-level (n=8) covariate 17개와 Δ 메트릭 사이의 Spearman 순위 상관:

| Group | Covariates |
|---|---|
| **규모/희귀도** | n, Y_rate_T{5,10,15,20} (4개) |
| **Baseline 강도** | BL_AUROC = mean(BIBLIO AUROC over 6 model × 4 T) |
| **지식 깊이** | PK_mean, SK_mean, TCT_mean, CKH_mean |
| **Assignee 집중도** | asg_HHI, top5_share |
| **IPC subclass 다양성** | Shannon entropy, distinct count, top-1 share, mean tags/patent |
| **IPC main_group 다양성** | Shannon entropy, distinct count |
| **Debate 신호** | H_final_mean, delta_H_mean, semantic_coherence_mean |
| **Quadrant** | zX (Concentration), zY (Depth) — `per_field_quadrant_placement.csv` |

n=8이므로 |ρ| > 0.71 ≈ p<0.05. 분석 성격은 **탐색적**.

Δ는 (field, T)별로 6개 model을 평균낸 single-value 사용. T별로 4세트의 ρ 결과
× 2 (AUROC, DOR target) = 8개 ρ 표.

---

## 3. 결과 — Δ AUROC

각 셀: 20 seed 평균. 표시: `*` = BH-FDR q<0.05, `·` = q<0.10, blank = ns.

### 3.1 T = 5% (가장 희귀, top 5.84%)

| Field | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 01-ICE | +0.0052 | +0.0114\* | +0.0372\* | -0.0004 | +0.0077 | -0.0270· |
| 03-HCCI | +0.0105 | +0.0196 | -0.0268 | -0.0181 | +0.0159 | -0.0123 |
| 07-Hyb | -0.0259 | +0.0219 | -0.0523· | -0.0131 | -0.0429 | +0.0028 |
| 08-EGR | +0.0019 | -0.0033 | +0.0379\* | -0.0172\* | -0.0040 | -0.0033 |
| 09-Trb | -0.0225\* | -0.0187 | -0.0208· | +0.0113 | +0.0317· | +0.0162 |
| 12-VVA | -0.0395\* | -0.0086 | -0.0200 | -0.0118· | -0.0218 | -0.0473\* |
| 13-Alt | +0.0137 | +0.0037 | +0.0207 | -0.0032 | -0.0636\* | -0.0709 |
| **14-DI** | **+0.0968\*** | **+0.1763·** | **+0.2962\*** | -0.0141 | +0.0179 | -0.0551 |

BH-FDR sig: **10/48 q<0.05, 16/48 q<0.10**.

### 3.2 T = 10%

| Field | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 01-ICE | +0.0112\* | +0.0010 | +0.0383\* | +0.0067· | -0.0011 | -0.0189 |
| 03-HCCI | -0.0250· | -0.0004 | +0.0117 | -0.0254· | +0.0333 | -0.0067 |
| 07-Hyb | +0.0212 | -0.0140 | -0.0146 | -0.0205· | -0.0303 | +0.0050 |
| 08-EGR | +0.0050 | +0.0019 | +0.0394\* | -0.0032 | +0.0330\* | +0.0022 |
| 09-Trb | -0.0152· | -0.0151 | -0.0132 | +0.0067· | -0.0127 | -0.0301· |
| 12-VVA | +0.0032 | -0.0239 | +0.0193 | +0.0050 | -0.0232 | +0.0225 |
| 13-Alt | +0.0098 | -0.0037 | +0.0176 | -0.0023 | -0.0597\* | -0.0141 |
| 14-DI | +0.0150 | +0.0270 | +0.0993\* | -0.0139 | -0.0476\* | -0.0044 |

BH-FDR sig: 7/48 q<0.05, 14/48 q<0.10.

### 3.3 T = 15%

| Field | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 01-ICE | +0.0074\* | +0.0003 | +0.0182\* | +0.0062\* | -0.0056 | -0.0202\* |
| 03-HCCI | -0.0133 | -0.0173 | +0.0102 | -0.0320· | +0.0154 | -0.0481 |
| 07-Hyb | -0.0113 | +0.0242 | +0.0181 | -0.0188\* | -0.0237 | +0.0498 |
| 08-EGR | +0.0181· | +0.0365\* | +0.0193 | -0.0053 | +0.0679\* | -0.0092 |
| 09-Trb | -0.0099· | -0.0139· | -0.0081 | +0.0039 | +0.0055 | -0.0181 |
| 12-VVA | +0.0227 | +0.0086 | +0.0260· | +0.0237\* | +0.0385\* | +0.0595\* |
| 13-Alt | -0.0094 | +0.0119 | +0.0114 | +0.0118 | -0.0028 | +0.0190 |
| 14-DI | -0.0174 | -0.0211 | +0.0189 | -0.0106 | -0.0228 | +0.0336 |

BH-FDR sig: 10/48 q<0.05, 15/48 q<0.10.

### 3.4 T = 20% (가장 덜 희귀)

| Field | RF | GBT | XGB | LogReg | SVM | FFN |
|---|---|---|---|---|---|---|
| 01-ICE | +0.0060\* | +0.0068 | +0.0200\* | +0.0069\* | +0.0079 | -0.0132 |
| 03-HCCI | +0.0382· | +0.0307 | +0.0592\* | -0.0003 | -0.0176 | +0.0042 |
| 07-Hyb | -0.0087 | +0.0088 | +0.0138 | -0.0208\* | -0.0451\* | +0.0222 |
| 08-EGR | +0.0132\* | +0.0124 | +0.0537\* | -0.0080 | +0.0483\* | +0.0098 |
| 09-Trb | -0.0230\* | -0.0236\* | -0.0238\* | +0.0016 | +0.0007 | -0.0405\* |
| 12-VVA | +0.0094 | +0.0108 | +0.0293\* | -0.0114 | +0.0284\* | +0.0694\* |
| 13-Alt | +0.0236\* | +0.0255· | +0.0440\* | +0.0142· | +0.0162 | +0.0362 |
| 14-DI | -0.0230 | -0.0529\* | -0.0062 | +0.0031 | +0.0107 | +0.0538 |

BH-FDR sig: **19/48 q<0.05, 22/48 q<0.10** — 모든 T 중 가장 많음.

### 3.5 Field-averaged Δ AUROC (4 T × 6 model 평균)

| Field | Δ AUROC | std | 비고 |
|---|---|---|---|
| **14-DI** | **+0.0233** | 0.0784 | XGB T=5%에서 +0.296 (extreme outlier, n=203) |
| **08-EGR** | **+0.0145** | 0.0227 | 모든 T에서 일관되게 +, XGB와 SVM 안정적 lift |
| 12-VVA | +0.0070 | 0.0290 | T=15/20에서 강함 (FFN +0.069) |
| 01-ICE | +0.0047 | 0.0154 | 가장 큰 field, 안정적이지만 작음 |
| 13-Alt | +0.0021 | 0.0292 | T=20에서만 일관 |
| 03-HCCI | +0.0002 | 0.0254 | mixed, n=130 노이즈 |
| 07-Hyb | -0.0064 | 0.0254 | 미세하게 해로움 |
| **09-Trb** | **-0.0096** | 0.0164 | 가장 부정적, RF/GBT/XGB 모두 - |

---

## 4. 결과 — Δ DOR

DOR (Haldane) 분산이 매우 크다 (특히 T=5,10에서 단일 셀이 ±10 이상 흔들림).
대부분 셀에서 BH-FDR p<0.05 못 넘김. 의미있는 시그널은 **T=15, 20%** 에서 나옴.

### 4.1 BH-FDR sig 셀 수 (out of 48)

| T | q<0.05 | q<0.10 |
|---|---|---|
| 5% | 0 | 0 |
| 10% | 0 | 0 |
| 15% | 4 | 6 |
| 20% | 11 | 13 |

T=20%에서 의미있는 패턴 — Δ DOR 데이터 표는 `outputs/per_field_delta_dor.csv` 참조.

핵심 패턴: **08-EGR**과 **13-Alt**이 T=20%에서 여러 모델에 걸쳐 일관된 + DOR
lift, **09-Trb**는 일관된 − DOR (RF/GBT/XGB 모두 negative).

---

## 5. Why? — Field-level Covariate vs Δ 상관

n=8이므로 표는 탐색적. * = p<0.05.

### 5.1 T = 5% (희귀)

**ΔAUROC 상위 |ρ|**

| Covariate | ρ | p | 해석 |
|---|---|---|---|
| **PK_mean** | **−0.905\*** | 0.002 | 선행지식(PK) 적은 field에서 debate 강함 |
| **SK_mean** | **−0.857\*** | 0.006 | 과학지식(NPL) 의존 적은 field에서 debate 강함 |
| Y_rate_T10 | −0.595 | 0.119 | 희귀 field일수록 debate ↑ (방향 약함) |

**ΔDOR 상위 |ρ|**

| Covariate | ρ | p |
|---|---|---|
| **CKH_mean** | **+0.762\*** | 0.028 |
| **ipc_tags_per_patent_mean** | **+0.738\*** | 0.037 |
| **Y_rate_T5** | **−0.714\*** | 0.046 |

**해석 (T=5%)**: bibliometric 신호가 빈약한 field — 즉 선행지식·과학지식 의존도가
낮고, Y가 희귀한 field — 에서 debate가 가장 큰 lift를 준다. AUROC 차원에서는
PK/SK가 dominant covariate. DOR 차원에서는 누적 인용지식(CKH)과 IPC 다중분류
정도가 양의 효과를 만든다.

### 5.2 T = 15%

**ΔAUROC 상위**

| Covariate | ρ | p |
|---|---|---|
| **zY (Depth)** | **+0.714\*** | 0.046 |
| **ipc_maingroup_entropy** | **−0.714\*** | 0.046 |
| BL_AUROC | +0.619 | 0.102 |

**ΔDOR 상위**

| Covariate | ρ | p |
|---|---|---|
| **delta_H_mean** | **−0.833\*** | 0.010 |
| **TCT_mean** | **+0.714\*** | 0.046 |

**해석 (T=15%)**: Quadrant 상의 Depth(zY)가 높은 field — 즉 누적 지식 깊이가
큰 도메인 — 에서 AUROC 효과 큼. IPC main_group 분산이 클수록(다양성 ↑) debate
효과는 약화. Δ DOR 측면에서는 belief change(delta_H)가 작은 field에서 오히려
큰 DOR uplift — debate가 강한 의견 변화를 만들지 않아도 final state의 정보가
유효하게 작동한다는 뜻.

### 5.3 T = 20%

**ΔAUROC, ΔDOR 모두 상위**

| Covariate | ρ | p |
|---|---|---|
| **semantic_coherence_mean** | **+0.762\*** | 0.028 (양쪽 동일) |

**해석 (T=20%)**: 완화된 임계값에서는 **debate 자체의 신호 품질**(semantic
coherence — 양 진영 발화가 의미공간에서 일관성 있게 응집되는 정도)이 lift 결정.
의미 일관된 debate가 발생한 field일수록 model이 그 신호를 잘 학습.

### 5.4 종합 — T별 dominant covariate

| T | dominant covariate | 방향 | story |
|---|---|---|---|
| 5% | **PK_mean, SK_mean** | − | bibliometric 약한 field에서 debate 보완 |
| 10% | (없음, 모두 \|ρ\|<0.71) | — | T=10에서는 field-level 설명력 약함 |
| 15% | **zY (depth), delta_H** | +, − | depth↑ + belief change↓ 조건에서 lift |
| 20% | **semantic_coherence** | + | debate 자체 신호 품질이 lift 결정 |

---

## 6. 핵심 발견

1. **Field별 효과는 명백히 이질적**. Field-averaged Δ AUROC 범위는 −0.010
   (09-Trb) ~ +0.023 (14-DI). Pooled (모든 cohort 단일 실험) 결과로는 보이지
   않는 신호.

2. **T=20%에서 가장 robust한 효과**. 48 셀 중 19 (40%)가 BH-FDR q<0.05.
   덜 희귀한 promising 정의에서 debate의 marginal contribution이 가장
   안정적으로 측정된다.

3. **T=5%의 효과는 PK/SK가 낮은 field에 집중** (ρ ≈ −0.9). 이는 debate가
   bibliometric의 빈 자리를 채우는 보완적 정보원이라는 가설을 뒷받침.

4. **Field 14 (Direct Injection)의 XGB extreme lift** (T=5% Δ AUROC =
   +0.296, p<0.05). n=203이지만 Y_pos=8개로 분산 매우 큼 (paired std=0.078).
   Outlier로 해석할지 본질적 신호로 해석할지 추가 검증 필요.

5. **Field 09 (Turbocharging)의 일관된 negative**. Δ AUROC −0.010, Δ DOR도
   T=20% 여러 모델에서 음. n=2,132로 가장 큰 field 중 하나임에도 debate가
   해롭다는 점은 별도 분석 가치 (debate 변수가 이 field의 특성과 mismatch?).

6. **모델별 패턴**: XGB가 거의 모든 (field, T)에서 가장 큰 |Δ|를 보임 — debate
   신호를 가장 잘 추출. SVM은 high variance, LogReg는 보수적, FFN은 작은 n
   field에서 unstable.

---

## 7. 산출물

```
Final_Experiment_0429/
├── code/
│   ├── experiments_per_field_sweep.py        ← 메인 sweep (7,680 fits)
│   ├── experiments_per_field_covariates.py   ← 17 field-level covariate + Spearman ρ
│   └── _show_per_field.py                    ← grid view 출력
├── data/
│   ├── variables_full_partial.csv            (7,086 patent × 16 biblio + 25 debate + Y/forward5/fields)
│   ├── ipc_lookup.parquet                    (patent_id × subclass_full × main_group_full, IPC 다양성 covariate 계산용)
│   └── per_field_quadrant_placement.csv      (zX, zY, asg_HHI, top5_share — Concentration × Depth quadrant 좌표)
├── outputs/
│   ├── per_field_threshold_full.csv          (7,680 rows: per seed × T × cfg × model × field)
│   ├── per_field_threshold_summary.csv       (seed mean ± std)
│   ├── per_field_delta_auroc.csv             (192 rows: 8 field × 4 T × 6 model)
│   ├── per_field_delta_dor.csv               (192 rows)
│   ├── per_field_covariates.csv              (8 rows × 24 covariates)
│   ├── per_field_correlation.csv             (Spearman ρ table, long format)
│   └── per_field_sweep.log                   (실행 로그)
└── reports/
    └── Field_Analysis.md                     ← 본 문서
```

**데이터 출처**:
- `variables_full_partial.csv`는 `debate/runs/v2a_y_anchored/results/variables_full_partial.csv`의 사본 (기존 스냅샷에 포함).
- `ipc_lookup.parquet`은 `data_collection/intermediate/ipc_lookup.parquet`의 사본 — Final_Experiment_0429 단독 재현을 위해 포함.
- `per_field_quadrant_placement.csv`는 자매 프로젝트(Autonomous_Vehicle)의 `scripts/per_field_quadrant_placement.csv` 사본 — quadrant covariate(zX, zY) 재현을 위해 포함.

### 재현

```bash
cd /c/Users/User/.../ICE_Domain
python Final_Experiment_0429/code/experiments_per_field_sweep.py        # ~30분-1h
python Final_Experiment_0429/code/experiments_per_field_covariates.py   # <1분
python Final_Experiment_0429/code/_show_per_field.py                    # 즉시
```

---

## 8. 한계 및 향후 작업

- **n=8 field-level 회귀의 통계력 제약**. Spearman ρ의 |ρ|>0.71 임계값은
  단순한 검증선이며, 단일 outlier가 ρ를 크게 흔들 수 있다 (특히 14-DI의
  +0.296 outlier). 부트스트랩 검정 추가 가치 있음.

- **Multi-label patent의 비독립성**. Explode-style은 한 patent을 여러 field
  실험에 등장시킨다 (387개 patent이 다중 소속). robustness check로
  sole-field-only 분석 추가 가능.

- **Debate 변수의 field-specific 해석**. semantic_coherence가 T=20%에서 강한
  predictor라는 발견은, 어떤 field에서 debate가 더 일관된 의미공간에서
  진행되는지 (그리고 왜)에 대한 후속 질문을 낳는다.

- **Field 09 (Turbocharging) 음의 효과 원인**. 가장 큰 field 중 하나임에도
  debate가 해롭다는 점은 독립 분석 가치.
