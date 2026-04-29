# 실험 결과 — Ranking 기반 평가

ICE 도메인 7,086 patents 에 대한 **ranking 성능 평가**.
Decision-rule (threshold) 에 따라 달라지는 confusion matrix 가 아닌,
**예측 확률의 ordering 품질** 자체를 측정.

binary classification (`Result.md`) / multi-class (`Result_L1-L4.md`)
와 동일한 cohort, 동일한 모델, 동일한 protocol — 평가 metric 만 ranking 기반.

---

## 실험 설정

| 항목 | 값 |
|---|---|
| Universe | 7,086 USPTO patents (Sinigaglia 2022 16-query union, 1980–2020) |
| Target | Y (binary, top 10% forward5) + forward5 (continuous, NDCG-graded용) |
| Train / Test split | 80 / 20 stratified random on Y |
| Protocol | 20 seeds (42, 0..18) mean ± std |
| 전처리 | median impute + min-max scale (train fold만 fit) |
| Test n | 1,418, base rate Y=1 = 12.49% |

3 가지 feature configuration (binary 실험과 동일):
- **BIBLIO** — 26 features (15 numeric biblio + MF top-10 one-hot + MF_other)
- **+25** — 51 features (BIBLIO + 25 debate 변수 전체)
- **+6focal** — 32 features (BIBLIO + 6 focal: var_conf_pro, H_final, delta_H, conf_gap_change, cross_domain_attack, semantic_coherence)

Ranking k 값: **10, 25, 50, 100, 142, 200, 300, 500** (test n=1,418).
k=142 가 top 10% (binary Y boundary 와 일치).

---

## Metric 정의

| Metric | 정의 | 의미 |
|---|---|---|
| **Precision@k** | $\|TP \cap \text{top}_k\| / k$ | top-k 중 실제 Y=1 비율 (decision maker가 top-k만 검토할 때 정확도) |
| **Recall@k** | $\|TP \cap \text{top}_k\| / \|TP\|$ | 전체 Y=1 중 top-k에 들어온 비율 (coverage) |
| **F1@k** | 2 P@k R@k / (P@k + R@k) | P@k와 R@k 의 조화평균 |
| **Lift@k** | P@k / base_rate | random 대비 몇 배 우수 (base_rate = 12.49%) |
| **NDCG@k binary** | $\frac{DCG@k}{\text{IDCG@k}}$, relevance ∈ {0, 1} | top-k 안에서 positive 가 얼마나 *앞쪽*에 있는지 (위치 가중) |
| **NDCG@k graded** | 같은 식, relevance = log1p(forward5) | 인용수 자체를 ground truth — high-cit patent를 top에 두는 능력 |
| **Spearman ρ** | spearmanr(predicted prob, forward5) | 전체 ranking 의 forward5 와의 일치도 |
| AUROC / AUPRC | (참조) | threshold-free 표준 지표 |

**Lift chart**: cumulative Recall (= TPR) at percentage points {1, 2, 5, 10, 14, 20, 30, 50, 100}%.
"top X% 검토 시 전체 positive 의 몇 % 잡히나" 를 직접 보여줌.

---

## AUROC / AUPRC / Spearman ρ — 참조 지표

| Config | Model | AUROC | AUPRC | Spearman ρ |
|---|---|---:|---:|---:|
| **BIBLIO** | **RF** | **0.7507 ± 0.018** | **0.3496 ± 0.027** | **0.3961 ± 0.019** |
| BIBLIO | XGB | 0.7130 ± 0.016 | 0.3056 ± 0.031 | 0.3380 ± 0.018 |
| BIBLIO | FFN | 0.7110 ± 0.023 | 0.2718 ± 0.023 | 0.3566 ± 0.018 |
| BIBLIO | SVM | 0.6482 ± 0.023 | 0.2228 ± 0.023 | 0.2516 ± 0.029 |
| **+6focal** | **RF** | **0.7528 ± 0.017** | 0.3434 ± 0.029 | **0.4009 ± 0.020** |
| +6focal | XGB | 0.7266 ± 0.019 | 0.2980 ± 0.020 | 0.3608 ± 0.019 |
| +6focal | FFN | 0.6951 ± 0.021 | 0.2577 ± 0.018 | 0.3252 ± 0.027 |
| +6focal | SVM | 0.6589 ± 0.015 | 0.2197 ± 0.018 | 0.2612 ± 0.021 |
| +25 | RF | 0.7426 ± 0.017 | 0.3290 ± 0.027 | 0.3948 ± 0.022 |
| +25 | XGB | 0.7168 ± 0.016 | 0.2888 ± 0.024 | 0.3561 ± 0.023 |
| +25 | FFN | 0.6584 ± 0.022 | 0.2126 ± 0.019 | 0.2664 ± 0.031 |
| +25 | SVM | 0.6569 ± 0.017 | 0.2073 ± 0.015 | 0.2570 ± 0.024 |

**관찰**: Spearman ρ 가 추가로 ranking 품질을 알려줌.
- RF 의 Spearman 0.40 = 전체 ordering 이 forward5 와 moderate 하게 일치
- SVM 의 Spearman 0.25 — 전체 ordering 약함

---

## Precision@k (mean over 20 seeds)

| Config | Model | P@10 | P@25 | P@50 | P@100 | P@142 | P@200 | P@300 | P@500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **0.785** | **0.634** | **0.534** | **0.421** | **0.376** | **0.338** | **0.297** | **0.239** |
| BIBLIO | XGB | 0.665 | 0.564 | 0.459 | 0.386 | 0.350 | 0.308 | 0.268 | 0.220 |
| BIBLIO | FFN | 0.500 | 0.392 | 0.380 | 0.335 | 0.322 | 0.296 | 0.267 | 0.221 |
| BIBLIO | SVM | 0.360 | 0.330 | 0.328 | 0.295 | 0.259 | 0.242 | 0.222 | 0.195 |
| +6focal | RF | 0.770 | 0.624 | 0.512 | 0.406 | 0.369 | 0.334 | 0.291 | **0.239** |
| +6focal | XGB | 0.590 | 0.526 | 0.424 | 0.368 | 0.335 | 0.307 | 0.269 | 0.227 |
| +6focal | FFN | 0.435 | 0.410 | 0.367 | 0.319 | 0.304 | 0.289 | 0.256 | 0.212 |
| +6focal | SVM | 0.335 | 0.314 | 0.304 | 0.269 | 0.248 | 0.235 | 0.223 | 0.200 |
| +25 | RF | 0.745 | 0.616 | 0.472 | 0.399 | 0.363 | 0.325 | 0.281 | 0.235 |
| +25 | XGB | 0.580 | 0.504 | 0.431 | 0.364 | 0.326 | 0.299 | 0.262 | 0.223 |
| +25 | FFN | 0.230 | 0.282 | 0.296 | 0.272 | 0.253 | 0.245 | 0.223 | 0.197 |
| +25 | SVM | 0.275 | 0.310 | 0.271 | 0.244 | 0.242 | 0.229 | 0.215 | 0.198 |

**관찰**: 
- **BIBLIO / RF 의 P@10 = 0.785** — top 10 patents 중 78.5% 가 실제 Y=1 (random 12.5% 대비 6.3× lift).
- **k 가 클수록 Precision 감소** (자연 — top-k 가 넓어지면 false positive 증가).
- **RF 가 모든 k 에서 압도적 1위**. XGB → FFN → SVM 순.
- **+6focal RF 가 BIBLIO RF 와 거의 동일** (P@10 0.770 vs 0.785, 차이 미미).

---

## Recall@k

| Config | Model | R@10 | R@25 | R@50 | R@100 | R@142 | R@200 | R@300 | R@500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | 0.044 | **0.090** | **0.151** | **0.238** | **0.301** | **0.382** | **0.504** | 0.675 |
| BIBLIO | XGB | 0.038 | 0.080 | 0.130 | 0.218 | 0.281 | 0.348 | 0.455 | 0.620 |
| BIBLIO | FFN | 0.028 | 0.055 | 0.107 | 0.189 | 0.258 | 0.335 | 0.452 | 0.624 |
| BIBLIO | SVM | 0.020 | 0.047 | 0.093 | 0.167 | 0.207 | 0.273 | 0.376 | 0.551 |
| +6focal | RF | 0.044 | 0.088 | 0.145 | 0.229 | 0.296 | 0.378 | 0.493 | **0.676** |
| +6focal | XGB | 0.033 | 0.074 | 0.120 | 0.208 | 0.269 | 0.347 | 0.456 | 0.642 |

**관찰**: 
- **k=142 (top 10%) 에서 R = 0.30 (RF)** — 전체 Y=1 의 30% 만 top decile에 들어옴.
- **k=500 (top 35%) 에서 R = 0.68 (RF)** — 전체 positive 의 약 2/3 이 top 35% 안에 있음.
- "유망 특허 발굴" 관점에서 top 10% 만 검토하면 **30% 만 잡힘** — coverage 증대를 위해선 top 35% 정도 검토 필요.

---

## Lift@k = P@k / base_rate (random = 1.0, **높을수록 좋음**)

| Config | Model | Lift@10 | Lift@25 | Lift@50 | Lift@100 | Lift@142 | Lift@200 | Lift@300 | Lift@500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **6.29** | **5.08** | **4.28** | **3.37** | **3.01** | **2.71** | **2.38** | 1.91 |
| BIBLIO | XGB | 5.33 | 4.52 | 3.68 | 3.09 | 2.81 | 2.46 | 2.15 | 1.76 |
| BIBLIO | FFN | 4.01 | 3.14 | 3.04 | 2.68 | 2.58 | 2.37 | 2.14 | 1.77 |
| BIBLIO | SVM | 2.88 | 2.64 | 2.63 | 2.36 | 2.07 | 1.94 | 1.78 | 1.56 |
| +6focal | RF | 6.17 | 5.00 | 4.10 | 3.25 | 2.96 | 2.68 | 2.33 | **1.92** |
| +6focal | XGB | 4.73 | 4.21 | 3.40 | 2.95 | 2.68 | 2.46 | 2.16 | 1.82 |
| +6focal | FFN | 3.48 | 3.28 | 2.94 | 2.56 | 2.43 | 2.31 | 2.05 | 1.70 |
| +25 | RF | 5.97 | 4.94 | 3.78 | 3.19 | 2.91 | 2.61 | 2.25 | 1.89 |
| +25 | XGB | 4.65 | 4.04 | 3.45 | 2.91 | 2.62 | 2.40 | 2.10 | 1.78 |

**관찰**: 
- **BIBLIO / RF Lift@10 = 6.29** — **top 10 검토 시 random 대비 6.3× 효율적**.
- 실용적 의미: 사람이 1,418개 모두 검토 못하고 10개만 본다면, RF 추천 top 10 중 78.5% 가 진짜 유망. random 추출 시 12.5%.
- k 가 커질수록 lift 감소 (top 100 → 3.4×, top 500 → 1.9×).

---

## NDCG@k (binary relevance, Y ∈ {0, 1})

| Config | Model | NDCG@10 | NDCG@25 | NDCG@50 | NDCG@100 | NDCG@142 | NDCG@200 | NDCG@300 | NDCG@500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **0.834** | **0.701** | **0.603** | **0.490** | **0.441** | **0.435** | **0.520** | **0.631** |
| BIBLIO | XGB | 0.693 | 0.608 | 0.515 | 0.438 | 0.399 | 0.389 | 0.464 | 0.571 |
| BIBLIO | FFN | 0.524 | 0.434 | 0.410 | 0.364 | 0.347 | 0.353 | 0.435 | 0.547 |
| BIBLIO | SVM | 0.399 | 0.360 | 0.347 | 0.315 | 0.283 | 0.290 | 0.362 | 0.475 |
| +6focal | RF | 0.823 | 0.693 | 0.584 | 0.475 | 0.432 | 0.429 | 0.510 | 0.628 |
| +25 | RF | 0.783 | 0.675 | 0.546 | 0.462 | 0.421 | 0.415 | 0.491 | 0.613 |

**관찰**: 
- **BIBLIO RF NDCG@10 = 0.834** — 매우 높음. top 10 ranking 의 위치 가중 품질 우수.
- NDCG 가 k 따라 U-자 형태 (k=10에서 high, k=200에서 low, k=500에서 다시 회복) — top-k 가 매우 좁으면 ideal에 가까움, 중간엔 noisy, 충분히 크면 IDCG도 함께 큼.

---

## NDCG@k (graded relevance, log1p(forward5))

| Config | Model | NDCG@10 | NDCG@25 | NDCG@50 | NDCG@100 | NDCG@142 | NDCG@200 | NDCG@300 | NDCG@500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **0.718** | **0.680** | **0.659** | **0.646** | 0.646 | 0.652 | 0.667 | 0.691 |
| BIBLIO | XGB | 0.631 | 0.610 | 0.596 | 0.597 | 0.603 | 0.607 | 0.621 | 0.654 |
| BIBLIO | FFN | 0.552 | 0.533 | 0.549 | 0.564 | 0.580 | 0.593 | 0.615 | 0.651 |
| BIBLIO | SVM | 0.484 | 0.495 | 0.508 | 0.521 | 0.520 | 0.534 | 0.554 | 0.599 |
| +6focal | RF | 0.709 | 0.673 | 0.652 | 0.645 | **0.648** | **0.653** | 0.662 | **0.691** |
| +25 | RF | 0.679 | 0.659 | 0.634 | 0.633 | 0.636 | 0.642 | 0.652 | 0.683 |

**관찰**: 
- **graded NDCG 가 binary 보다 안정적** (0.65-0.72 범위).
- **+6focal 이 BIBLIO 와 거의 동등 — 일부 k 에서 약간 우세** (NDCG@142, 200).
  → 인용수 자체 (continuous) 를 ground truth 로 보면 +6focal 의 ranking 이 BIBLIO 에 동등 이상.
- **+25 약간 손해** (NDCG@10: 0.679 vs BIBLIO 0.718).

---

## Lift Chart — Cumulative Recall at percentage points

상위 k% 검토 시 전체 positive 의 누적 비율. 도표 데이터 (mean over 20 seeds):

| Config | Model | 1% | 2% | 5% | 10% | 14% | 20% | 30% | 50% | 100% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BIBLIO** | **RF** | **0.058** | **0.098** | **0.191** | **0.301** | **0.380** | **0.485** | **0.619** | **0.811** | 1.000 |
| BIBLIO | XGB | 0.050 | 0.087 | 0.171 | 0.281 | 0.347 | 0.441 | 0.564 | 0.756 | 1.000 |
| BIBLIO | FFN | 0.036 | 0.062 | 0.143 | 0.258 | 0.334 | 0.435 | 0.568 | 0.760 | 1.000 |
| BIBLIO | SVM | 0.028 | 0.051 | 0.127 | 0.207 | 0.272 | 0.361 | 0.491 | 0.683 | 1.000 |
| +6focal | RF | 0.058 | 0.095 | 0.182 | 0.296 | 0.377 | 0.478 | 0.615 | **0.812** | 1.000 |
| +6focal | XGB | 0.044 | 0.080 | 0.159 | 0.269 | 0.346 | 0.439 | 0.581 | 0.779 | 1.000 |
| +25 | RF | 0.057 | 0.092 | 0.176 | 0.291 | 0.367 | 0.460 | 0.599 | 0.796 | 1.000 |

**관찰** (BIBLIO RF 기준):
- **상위 1% (n=14) 검토 시 전체 positive 의 5.8%** 검출 (random = 1%, 5.8× lift)
- **상위 10% (n=142) 검토 시 30.1%** 검출 (3.0× lift)
- **상위 30% (n=425) 검토 시 61.9%** 검출 (2.1× lift)
- **상위 50% (n=709) 검토 시 81.1%** 검출 (1.6× lift)

→ **top 30% 만 봐도 promising 의 60% 잡힘**. 효율적 trade-off zone.

### Lift chart ASCII 시각화 (BIBLIO RF)

```
Cumulative Recall (%)
100 |- - - - - - - - - - - - - - - - - - - - - - - - * 
 90 |                                              .*
 80 |                                          .*  
 70 |                                      .*      
 60 |                                  *           
 50 |                              *               
 40 |                          *                   
 30 |                    *                         
 20 |              *                              
 10 |        *                                    
  0 |__*___________________________________________
      1  2   5  10  14  20    30        50         100
                Percentile (top X% reviewed)
```

* = BIBLIO/RF cumulative recall 점.
random baseline = 직선 (top X% 검토 시 정확히 X% recall).

---

## 핵심 관찰

### 1. **RF 가 ranking 에서도 압도적 1위**

| Metric | BIBLIO RF | BIBLIO XGB | 차이 |
|---|---:|---:|---:|
| AUROC | 0.751 | 0.713 | +0.038 |
| Spearman ρ | 0.396 | 0.338 | +0.058 |
| P@10 | 0.785 | 0.665 | +0.120 |
| Lift@10 | 6.29 | 5.33 | +0.96 |
| NDCG@10 binary | 0.834 | 0.693 | +0.141 |
| NDCG@10 graded | 0.718 | 0.631 | +0.087 |

→ Decision-rule 기반 분석에서도 RF 1위였고, ranking 기반에서는 더욱 명확하게 1위.

### 2. **+6focal 는 BIBLIO 와 거의 동등 (ranking 에서)**

| Metric | BIBLIO RF | +6focal RF | +25 RF |
|---|---:|---:|---:|
| AUROC | 0.7507 | **0.7528** | 0.7426 |
| Spearman | 0.3961 | **0.4009** | 0.3948 |
| P@10 | **0.785** | 0.770 | 0.745 |
| Lift@10 | **6.29** | 6.17 | 5.97 |
| NDCG@10 binary | **0.834** | 0.823 | 0.783 |
| NDCG@10 graded | **0.718** | 0.709 | 0.679 |
| NDCG@142 graded | 0.646 | **0.648** | 0.636 |

→ **+6focal 이 AUROC, Spearman 에서 약간 우세**, P/Lift/NDCG 에서 약간 손해.
→ +25 는 ranking 에서도 일관 손해 (모든 metric 에서 BIBLIO < +6focal < +25 순서로 손해 진행).

### 3. **상위 10 patents 의 강력한 lift**

BIBLIO RF top 10 = 78.5% 진짜 유망 (Lift 6.29×). NDCG@10 = 0.834.

→ **실용적 deployment**: 인간 평가자가 상위 10-25개만 검토하는 시나리오에서 매우 효율적. 검토 부담 1/100 로 줄이면서 lift 5-6× 확보.

### 4. **k 별 trade-off**

| k | P@k | R@k | Lift@k | 의미 |
|---:|---:|---:|---:|---|
| 10 | 0.785 | 4.4% | 6.29× | 매우 conservative — top10 안정적 |
| 50 | 0.534 | 15.1% | 4.28× | 효율적 zone |
| 142 (top 10%) | 0.376 | 30.1% | 3.01× | 균형점 |
| 500 (top 35%) | 0.239 | 67.5% | 1.91× | broad coverage |

→ Decision maker가 "review budget" 따라 k 선택:
- **strict (n≤25)**: top 25 review with 63% precision — *quality over quantity*
- **balanced (n=142)**: top 10% with 38% precision and 30% recall
- **broad (n=500)**: top 35% with 24% precision but 68% recall — *coverage over precision*

### 5. **Decision-rule 무관 ranking 신호 검증**

이전 binary classification 에서 본 패턴:
- AUROC 기준 +6focal 약간 우세 (XGB +0.014)
- DOR (top-10% threshold 기반) 기준 BIBLIO 우세

→ Ranking 기반에서는 **threshold 영향 없음**. 결과:
- AUROC, Spearman, NDCG_graded@142+: +6focal RF 우세
- P@k, Lift@k, NDCG_binary@10-100: BIBLIO RF 우세
- → **두 config 가 본질적으로 거의 동등** — debate 추가가 ranking 자체에는 거의 효과 없음.

### 6. **FFN/SVM 은 ranking 에서도 약함**

특히 +25 config 에서 FFN P@10 = 0.230 (RF 의 1/3 수준), SVM Lift@10 = 2.20 (RF 의 1/3).
→ 이전 결론 재확인: 51-feature 입력에 FFN/SVM 은 과적합/처리한계.

---

## 결론

| 질문 | 답 |
|---|---|
| Ranking 성능은 어떤 모델이 좋은가? | **RF 압도적** (모든 ranking metric 1위) |
| Debate 변수 추가 효과는? | **거의 없음** — +6focal RF ≈ BIBLIO RF, +25 RF 약간 손해 |
| Decision-rule 무관 평가에서도 같은 결론? | YES — binary classification 결론과 일관 |
| 실용적 deployment 권장? | **BIBLIO RF top 25 review** — Precision 63%, Lift 5×, 효율적 |
| Debate 가 도움 안 되는 이유? | Ranking 의 강한 신호는 biblio 변수 (특히 TCT, PK, SK) 가 이미 캡처. Debate 의 marginal contribution 은 threshold 기반 metric 에서만 detectable |

### 종합 추천

**Production deployment**: **BIBLIO + Random Forest** model with top-k ranking output.
- top 10: 정확도 78.5%
- top 25: 정확도 63.4%
- top 142: 정확도 37.6% but recall 30%

→ Debate run 비용 (~25 LLM calls/patent) 대비 ranking 향상 거의 없음. **Decision rule 변경할 때만 debate 가치 발현** (이전 분석에서 본 +0.014 AUROC).

---

## 산출물

| 파일 | 내용 |
|---|---|
| `analysis/outputs/experiments_summary/ranking_full_metrics.csv` | per-seed × per (config, model) all ranking metrics |
| `analysis/outputs/experiments_summary/ranking_summary.csv` | mean ± std aggregated, 모든 metric × k |
| `analysis/outputs/experiments_summary/ranking_lift_chart.csv` | cumulative Recall at 9 percentage points |
| `analysis/experiments_ranking.py` | 재현 스크립트 |
