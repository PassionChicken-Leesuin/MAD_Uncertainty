# MAD_Uncertainty — Multi-Agent Debate 불확실성 변수의 유망기술 예측 효용 분석

> ICE(Internal Combustion Engine) 도메인 USPTO 특허 7,086건을 대상으로,
> 5개 전문가 페르소나의 **multi-agent debate(MAD)** 에서 추출한
> 불확실성 변수가 bibliometric-only baseline 대비 유망기술 라벨링
> (`forward5 ≥ q90`, top-10%) 예측 성능을 어떻게 끌어올리는지 검증한
> 실험 코드·데이터·결과 스냅샷.

---

## 1. 핵심 연구 질문

1. **변수 효용**: 5인 debate 트레이스에서 추출한 25개 불확실성 변수가
   bibliometric 26개 baseline 대비 분류 성능을 개선하는가?
2. **최적 부분집합**: 25개 중 이론적으로 narrowing한 6개 focal 변수의
   2⁶−1 = 63개 부분집합 중 robust winner는 무엇인가?
3. **도메인 의존성**: ICE 16개 sub-field 중 어느 field에서 debate 변수가
   힘을 발휘하며, **debate 진행 이전**에 그 효과를 예측 가능한가?
4. **임계값 감수성**: Y 정의의 percentile cutoff (T = 5/10/15/20%)에
   따라 변수 효용 ranking이 어떻게 변하는가?

## 2. 실험 파이프라인

```
USPTO 특허 (제목+초록)
   └─ MAD 프로토콜 (5 페르소나 × 최대 6 라운드, gpt-4o-mini)
       │   Technology · Application · User · Ecosystem · BusinessModel
       │   + Validator (사실성 판정) + Moderator (rule-based 종료)
       └─ DebateState (rounds, validations, termination, final_pred)
           └─ 변수 추출 25개 (개별내적 H_final, 동학 delta_H,
                              상호작용 cross_domain_attack, 합의 semantic_coherence, …)
               └─ 6 분류기 (RF / GBT / XGB / LogReg / SVM / FFN) × 20 시드
                   └─ 평가: AUROC, AUPRC, Precision/Recall/DOR @ k%
```

알고리즘 상세 → [`ALGORITHM.md`](ALGORITHM.md), 변수 형식 정의 →
[`VARIABLES.md`](VARIABLES.md).

## 3. 코호트와 라벨

| 항목 | 값 |
|---|---|
| Universe | USPTO grant patent **7,086건** |
| 출처 | Sinigaglia (2022) Applied Energy 306, 118003 의 16개 ICE 검색식 union |
| 기간 | `patent_date` ∈ [1980-01-01, 2020-12-31] |
| Y 정의 | $Y(X) = \mathbb{1}[\phi_5(X) \geq q_{90}]$, $\phi_5$ = 5년 forward citation |
| Y rate (T=10%) | 12.49% (885/7,086) — ties로 인해 정확히 10%는 아님 |
| Train/Test | 80/20 stratified, 시드 20개 (`[42, 0, 1, ..., 18]`) |

## 4. 디렉터리 구조

```
.
├── ALGORITHM.md                  # MAD 프로토콜 명세 (페르소나/Validator/Moderator)
├── VARIABLES.md                  # 형식적 변수 정의 — 코드보다 우선
├── Result_Threshold.md           # 실험 1: T sweep × 6 모델 × 3 config
├── Result_FI_SHAP.md             # 실험 2: Permutation Importance + SHAP
├── Result_DebateBestSubset.md    # 실험 3: 6-변수 부분집합 전수탐색 (64 configs)
├── Result_QuadrantAnalysis.md    # 실험 4: pre-debate decision framework (PK_mean)
├── Field_Analysis.md             # 실험 5: ICE sub-field 8개 per-field 효과
│
├── prompts/
│   ├── v0_baseline.py            # baseline persona 프롬프트
│   └── v2a_y_anchored.py         # Y-anchored 프롬프트
│
├── analysis/                     # 분석 스크립트 모음
│   ├── exhaustive_6vars_20seed.py
│   ├── exhaustive_7vars_metricgrid.py
│   ├── experiments_threshold_sweep.py
│   ├── experiments_threshold_importance.py
│   ├── narrowing_justification.py
│   ├── prompt_comparison_*.py
│   └── outputs/                  # 분석 결과 CSV/요약
│
├── results/prompt_comparison/    # 프롬프트 v0 ↔ v2a 비교 결과
│
└── Final_Experiment_0429/        # 2026-04-29 기준 실험 1·2·3 패키지 스냅샷
    ├── README.md                 # 폴더별 상세 설명
    ├── code/                     # 재현 가능한 실험 스크립트
    ├── data/
    │   ├── variables_full.csv          # 7,086 × 전 변수 (실험 3)
    │   ├── variables_full_partial.csv  # 실험 1, 2 입력
    │   └── ipc_lookup.parquet
    ├── outputs/                  # threshold/importance/exhaustive 결과
    └── reports/                  # Result_*.md 사본
```

## 5. 주요 변수군

**Bibliometric 26** = 15 numeric (CTO, STO, PK, SK, TCT, TS, NC, COL,
INV, TKH, CKH, PKH, TTS, CTS, PTS) + MF top-10 one-hot + MF_other.

**Focal Debate 6** (25 → 6 narrowing) — 4-그룹 불확실성 분류체계의 각 그룹 대표:

| Group | 의미 | 변수 |
|---|---|---|
| A 상호작용 | 도메인 간 충돌 강도 | `cross_domain_attack` |
| B 동학 | 토론 진행에 따른 불확실성 변화 | `conf_gap_change`, `delta_H` |
| C 개별내적 | 페르소나 자체 불확실성 | `var_conf_pro`, `H_final` |
| D 합의응집 | 최종 합의 일관성 | `semantic_coherence` |

Narrowing 정당화 → `analysis/narrowing_justification.py`.

## 6. 모델 사양

| 모델 | 설정 |
|---|---|
| RF | `RandomForestClassifier(n_estimators=300)` |
| GBT | `HistGradientBoostingClassifier(max_iter=300)` |
| XGB | `xgboost.XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss")` |
| LogReg | `LogisticRegression(L2, C=1.0, max_iter=2000)` |
| SVM | `SVC(kernel="rbf", probability=True)` |
| FFN | `MLPClassifier(hidden=(⌈√(n_features×2)⌉,), logistic, lbfgs, α=0.001)` |

각 모델 `random_state` = 시드. 전처리: train fold 단독 fit한 median
imputer + min-max scaler (no leakage).

## 7. LLM 설정

| Setting | Value |
|---|---|
| `MODEL_NAME` | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` (semantic_coherence용) |
| `TEMPERATURE_AGENT` / `_VALIDATOR` | 0.7 / 0.3 |
| `MAX_DEBATE_ROUNDS` | 6 (round 0 + 1..5) |
| `MODERATOR_FORCE_ROUND` | 5 |
| `MAX_RETRIES` | 20 |
| `RANDOM_SEED` | 42 |

**도메인 누설 방지 규칙**: 페르소나 / Validator 프롬프트는 dataset
식별 단어(`autonomous`, `vehicle`, `EV`, `combustion`, `engine`,
`hybrid`, `turbo` 등)를 포함하지 않음 — 페르소나는 특허 텍스트만 평가.

## 8. 데이터 의존성 (저장소 외부)

다음은 `.gitignore` 처리되어 있으며 별도 경로에서 빌드해야 합니다.

- `data_collection/intermediate/`, `data_collection/cache__*/`,
  `data_collection/sets__paper_1980_2018/` — bulk USPTO TSV 파생물
- `debate/runs/*/checkpoints/`, `debate/runs/*/debates/`,
  `debate/runs/*/logs/` — debate 실행 체크포인트와 per-patent 로그
- `*.parquet`, `*.pkl` — 빌드 가능한 캐시

## 9. 결과 문서 진입점

| 파일 | 핵심 결론 (요약) |
|---|---|
| [Result_Threshold.md](Result_Threshold.md) | T(5/10/15/20%) × 3 config × 6 모델 × 20 시드 sweep |
| [Result_FI_SHAP.md](Result_FI_SHAP.md) | `+6focal` config의 임계값별 Permutation/SHAP importance |
| [Result_DebateBestSubset.md](Result_DebateBestSubset.md) | 64 configs × 6 모델 × 20 시드 = 7,680 fits, robust winner 식별 |
| [Result_QuadrantAnalysis.md](Result_QuadrantAnalysis.md) | 1D 결정 규칙: PK_mean < 11 → 토론 진행, ρ = −0.905 (p=0.002) |
| [Field_Analysis.md](Field_Analysis.md) | n>100 8개 ICE field별 ΔAUROC 패턴 |
