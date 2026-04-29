# Final_Experiment_0429

> 2026-04-29 기준, ICE 도메인 multi-agent debate 변수의 효용을 평가한
> 세 가지 실험의 코드·데이터·결과를 한 폴더에 정리한 스냅샷:
> 1. **임계값(Threshold) Sweep 분석** (`Result_Threshold.md`)
> 2. **Feature Importance / SHAP 분석** (`Result_FI_SHAP.md`)
> 3. 🆕 **Debate 변수 최적 부분집합 탐색** (`Result_DebateBestSubset.md`)

---

## 1. 폴더 구조

```
Final_Experiment_0429/
├── README.md                  ← 본 문서
├── data/
│   ├── variables_full_partial.csv         (실험 1, 2 입력)
│   └── variables_full.csv                 (실험 3 입력 — 동일 7,086 cohort)
├── code/
│   ├── experiments_threshold_sweep.py     (실험 1)
│   ├── experiments_threshold_importance.py (실험 2)
│   ├── exhaustive_6vars_20seed.py         (실험 3 메인)
│   ├── narrowing_justification.py         (실험 3 — 25→6 narrowing 검증)
│   ├── _show_threshold.py
│   └── _show_importance.py
├── outputs/
│   ├── (실험 1)
│   │   threshold_sweep_full.csv          (per-seed-T-config-model)
│   │   threshold_sweep_summary.csv       (시드 평균±std)
│   │   threshold_sweep_delta.csv         (Δ AUROC vs BIBLIO)
│   │   threshold_sweep_dor_delta.csv     (Δ DOR vs BIBLIO)
│   ├── (실험 2)
│   │   threshold_importance_perm_full.csv     (per-seed-T-model-feature)
│   │   threshold_importance_perm_summary.csv
│   │   threshold_importance_shap_full.csv     (XGB only)
│   │   threshold_importance_shap_summary.csv
│   └── (실험 3)
│       narrow6_evidence.csv               (25 → 6 narrowing 종합 ranking)
│       exh6_metric_grid.csv               (7,680 fits × 4 k% raw)
│       exh6_auroc_long.csv                (AUROC/AUPRC raw)
│       exh6_subset_per_slot.csv           (subset × k% 집계, 120 obs/cell)
│       exh6_subset_AUROC.csv              (subset × AUROC/AUPRC 집계)
│       exh6_winners.csv                   (12 slot × top-5 ranking)
│       exh6_consensus.csv                 (subset × top-1/3/5 등장 빈도)
│       exh6_summary.md                    (자동 요약)
└── reports/
    ├── Result_Threshold.md                (실험 1)
    ├── Result_FI_SHAP.md                  (실험 2)
    └── Result_DebateBestSubset.md         (🆕 실험 3)
```

---

## 2. 실험 개요

### 2.1 공통 세팅
- **코호트**: 7,086 USPTO grant patent (1980–2020), Sinigaglia 2022 16개 ICE query union, debate가 종료된 patent.
- **80/20 stratified random split**, **20개 시드** (`[42, 0, 1, ..., 18]`).
- **전처리**: median impute + min-max scale, train fold 단독 fit.
- **Y 라벨**: `forward5 ≥ q(100−T)`, T ∈ {5, 10, 15, 20}%.

| T | cutoff (forward5) | Y rate |
|---|---|---|
| 5%  | ≥ 9 | 5.84%  |
| 10% | ≥ 6 | 12.49% |
| 15% | ≥ 5 | 15.83% |
| 20% | ≥ 4 | 21.17% |

### 2.2 실험 1 — Threshold Sweep (`experiments_threshold_sweep.py`)
- 4 임계값 × 3 config (BIBLIO / +6focal / +25) × 6 모델 (RF, GBT, XGB, LogReg, SVM, FFN) × 20 시드.
- 평가지표: AUROC, AUPRC, Precision, Recall, F1, DOR.
- → **`Result_Threshold.md`**

### 2.3 실험 2 — Feature Importance + SHAP (`experiments_threshold_importance.py`)
- 4 임계값 × **+6focal config 단독** × 6 모델 × 20 시드.
- **Permutation Importance**: sklearn, n_repeats=10, scoring=AUROC, test fold.
- **SHAP**: XGB만, `shap.TreeExplainer`, test fold, 평균 절대 SHAP.
- → **`Result_FI_SHAP.md`**

### 2.4 실험 3 — Debate 변수 최적 부분집합 (`exhaustive_6vars_20seed.py`)
- **6 후보 변수**: cross_domain_attack, conf_gap_change, var_conf_pro, H_final, delta_H, semantic_coherence
  - 4-그룹 불확실성 분류체계(A 상호작용 / B 동학 / C 개별내적 / D 합의)에서 각 그룹 대표 1~2개로 narrowing.
  - Narrowing 정당화 (`narrowing_justification.py`): 결합 시너지 + 이론 기반.
- **탐색 공간**: 2^6 − 1 = 63 비공집합 + BIBLIO = **64 configs**.
- **평가**: T=10% (Y rate 12.49%) 고정. 6 모델 × 20 시드 × 64 configs = **7,680 model fits** (~2시간).
- **Metric grid** (12 slot): Precision / Recall / DOR (Haldane) @ k% with k ∈ {5, 10, 15, 20}.
  - 추가로 AUROC, AUPRC (k-independent).
- **Robust winner 식별**: 12 slot 각각의 top-5 ranking + consensus (top-1/3/5 등장 빈도).
- → **`Result_DebateBestSubset.md`**

### 2.4 변수 정의
- **BIBLIO 26** = 15 numeric (CTO, STO, PK, SK, TCT, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS) + MF top-10 one-hot + MF_other.
- **Focal Debate 6** = `var_conf_pro`, `H_final`, `delta_H`, `conf_gap_change`, `cross_domain_attack`, `semantic_coherence`.
- **+25** = 26 BIBLIO + 25 debate variables 전부.

### 2.5 모델 사양

| 모델 | 사양 |
|---|---|
| RF | `RandomForestClassifier(n_estimators=300)` |
| GBT | `GradientBoostingClassifier(n_estimators=300, max_depth=3)` |
| XGB | `XGBClassifier(n_estimators=300, tree_method=hist)` |
| LogReg | `LogisticRegression(solver=lbfgs, max_iter=1000)` |
| SVM | `SVC(kernel=rbf, probability=True)` |
| FFN | `MLPClassifier(hidden_layer_sizes=(h,), activation=logistic, solver=lbfgs, alpha=0.001, max_iter=1000)`, `h = ceil(sqrt(2·n_features))` |

---

## 3. 핵심 발견 (TL;DR)

### 3.1 Threshold Sweep — debate 효과는 T가 작을수록 큼
- **T=5%에서 debate Δ AUROC가 최대.** XGB +6focal: +0.0321 (19/20 wins vs BIBLIO).
- LogReg는 모든 T에서 안정적 lift (17/20 wins at T=5/15/20%).
- RF는 임계값 무관하게 debate lift 미미.
- 자세한 표·수치는 `reports/Result_Threshold.md`.

### 3.2 Feature Importance — debate 변수의 보조 신호
- **TCT (Total Citations)**가 모든 (T, model)에서 #1.
- **`H_final`**이 가장 강건한 debate signal — LogReg/SVM/FFN에서 항상 Top-5.
- **`semantic_coherence`**는 SHAP에선 강함 (top-3) but permutation에선 약함 → BIBLIO와 redundant이지만 XGB가 광범위 사용.
- T=5%에서 debate 점유율 peak: GBT 16%, XGB 14%, LogReg 18%, FFN 17%.
- 자세한 표·수치는 `reports/Result_FI_SHAP.md`.

### 3.3 Debate 변수 최적 부분집합 — sparse가 robust
- **🏆 최적 부분집합: `cross_domain_attack + semantic_coherence`** (size 2)
  - 12 metric slot 중 **6 slot에서 #1**, **12 slot 모두 top-5** (100%).
  - k=15-20% 추천 영역에서 일관 winner: Precision +0.5~0.7%p, Recall +0.7~0.8%p, DOR +0.15~0.18.
- **size 2가 size 3+ 보다 일관되게 우수** — over-fitting 회피.
- k=5%에서는 BIBLIO_ONLY가 1위 — top-shortlist 영역에선 debate 추가 효과 거의 없음.
- 5-seed → 20-seed 변경 시 효과 크기 ~6배 축소 (Δ +0.012 → +0.002 수준) → 5-seed 결과는 split-luck 부풀림. 20-seed가 honest.
- AUROC 기준 winner는 다름 (`CA + H_final + delta_H + SC`, size 4) — task에 맞는 metric 선택 중요.
- 자세한 표·수치는 `reports/Result_DebateBestSubset.md`.

---

## 4. 재현 방법

### 4.1 환경
```
python = 3.x (signed PSF build 권장 — Smart App Control 환경)
pandas, numpy, scikit-learn, xgboost, shap
```

### 4.2 실행
이 폴더의 코드는 원본 위치 (`analysis/`) 기준 절대경로로 작성되어 있어,
**원본 위치에서 그대로 실행**하면 같은 결과가 재현됩니다.

```bash
cd /c/Users/User/.../ICE_Domain
python analysis/experiments_threshold_sweep.py        # 실험 1: ~3-4시간
python analysis/experiments_threshold_importance.py   # 실험 2: ~30분
python analysis/narrowing_justification.py            # 실험 3a: ~1분
python analysis/exhaustive_6vars_20seed.py            # 실험 3b: ~2시간 (7,680 fits)
python analysis/_show_threshold.py                    # threshold sweep summary
python analysis/_show_importance.py                   # importance summary
```

이 폴더의 `code/`, `data/`, `outputs/`는 **2026-04-29 시점의 스냅샷**이며,
연구 노트에서 직접 참조 가능한 freeze된 산출물입니다.
재실행 시 결과 CSV는 `analysis/outputs/experiments_summary/`에 새로 쓰여집니다 — 덮어쓰기 주의.

### 4.3 입력 데이터 출처
`data/variables_full_partial.csv`는
`debate/runs/v2a_y_anchored/results/variables_full_partial.csv`의 사본.
컬럼:
- `patent_id, patent_date, fields, forward5, Y, Category` (라벨/메타)
- BIBLIO 16: `CTO, STO, PK, SK, TCT, MF, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS`
- DEBATE 25: `mean_conf_pro, mean_conf_anti, var_conf_pro, var_conf_anti, conf_gap_change, H_final, delta_H, cross_domain_support, cross_domain_attack, same_domain_support, same_domain_attack, acceptability_gap, fact_ratio_pro, fact_ratio_anti, final_prediction, prediction_volatility, final_pred_technology, final_pred_application, final_pred_user, final_pred_ecosystem, final_pred_businessmodel, total_rounds, term_unanimous, term_extended_debate, semantic_coherence`

`mean_conf_pro`가 NaN인 patent는 debate 미완료로 제외 (cohort 7,086에 포함).

---

## 5. 관련 의사결정 기록

`../CLAUDE.md` §7의 R-1 ~ R-15 의사결정과 일치:
- Y = global top-T% (per-year/per-field 계층화 없음). T가 정해지면 q(100−T)에 ties → Y rate 약간 초과 가능 (예: T=10% → 12.49%).
- 80/20 stratified random split, single partition, `random_state` = seed.
- 6 모델은 AV `_common.py::make_models` 패밀리에서 확장 (LogReg + GBT 추가).
- `patent_date` 미사용.
- 클래스 가중 미사용 (raw imbalanced training).
- 의도적 한계: 동일 test set으로 +6focal 선정·평가 → optimistic bias 존재.
  Result_FI_SHAP.md는 confirmatory가 아니라 exploratory 분석으로 해석.
