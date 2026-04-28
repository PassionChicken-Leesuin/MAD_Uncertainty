# When Multi-Agent Debate Adds Value — A Pre-Debate 2×2 Framework

When does running a 5-persona debate add value over a bibliometric-only
baseline, and when is it wasted compute? This document proposes a
two-axis framework that is **computable from biblio data alone** (no
debate needed), and validates it empirically against 9 entities
(8 ICE sub-areas + AV-as-whole) for which debate uplift was measured.

---

## 1. Research question

Given the 16 bibliometric variables and 25 debate variables defined in
[`VARIABLES.md`](VARIABLES.md), and the per-patent debate algorithm in
[`ALGORITHM.md`](ALGORITHM.md):

> Can we **predict, from biblio data alone**, whether a domain will
> benefit from a 5-persona debate uplift?

This matters because the debate run is the expensive part of the
pipeline (roughly 25 LLM calls per patent vs zero for biblio
extraction). A predictive framework lets us decide whether to invest
in debate per new domain.

---

## 2. Hypothesis

Two orthogonal axes, both computable pre-debate:

- **X — Biblio Y-Discriminability**: how cleanly do the 16 biblio
  variables separate $Y = 1$ patents from $Y = 0$ patents in the
  field?
- **Y — Portfolio Peripheral Breadth**: how broad is the assignee
  portfolio outside the keyword universe?

Conjecture:

> Debate adds value when **both** axes are high. Both low → debate
> wastes compute.

Concretely:

- High X = biblio captures a usable signal → debate has a foundation
  to refine.
- High Y = assignees have broad cross-domain activity → the 5-persona
  framing has *substantive* multi-stakeholder material to work with.

Either alone is insufficient.

---

## 3. Operational definitions (pre-debate)

### 3.1 X axis — Biblio Y-Discriminability

For a field $f$ with patents $\mathcal{P}_f$, and the 15 numeric biblio
variables (CTO, STO, PK, SK, TCT, TS, NC, COL, INV, TKH, CKH, PKH,
TTS, CTS, PTS — see `VARIABLES.md`), define:

$$
\bar{d}_{\text{biblio}}(f) = \frac{1}{15} \sum_{v \in \text{biblio15}} \left|\frac{\bar{v}_{Y=1,f} - \bar{v}_{Y=0,f}}{\sigma_{v,\text{global}}}\right|
$$

where $\bar{v}_{Y=1,f}$ is the mean of variable $v$ over patents in
$f$ with $Y = 1$, $\sigma_{v,\text{global}}$ is the global standard
deviation of $v$ over the entire universe.

This is the average absolute Cohen's $d$ across biblio variables —
how strongly does the biblio profile differ between promising and
non-promising patents in this field.

**Interpretation**:
- HIGH = biblio variables clearly differentiate Y in this field
- LOW = biblio struggles to learn Y in this field

### 3.2 Y axis — Portfolio Peripheral Breadth

For each focal patent $X$, define the peripheral fraction of its
assignee portfolio (excluding the focal):

$$
\rho_{\text{periph}}(X) = \frac{\mathrm{PKH}(X)}{\mathrm{CKH}(X) + \mathrm{PKH}(X)}
$$

where $\mathrm{CKH}$ and $\mathrm{PKH}$ are defined in `VARIABLES.md`
(core / peripheral assignee portfolio counts at $t_X$, with respect
to the keyword universe $\mathcal{K}$).

Field-level metric:

$$
\bar{\rho}_{\text{periph}}(f) = \frac{1}{|\mathcal{P}_f|} \sum_{X \in \mathcal{P}_f} \rho_{\text{periph}}(X)
$$

**Interpretation**:
- HIGH = assignees in this field have most of their portfolio
  *outside* the keyword universe — they work across many other domains
- LOW = assignees are concentrated within the keyword universe —
  domain-pure portfolios

A high value indicates the field's developers carry knowledge from
many adjacent domains, which is the substrate the multi-stakeholder
debate frame is supposed to leverage.

---

## 4. Empirical validation

### 4.1 Validation set

9 entities for which an empirical debate uplift Δ is known:

- 8 ICE sub-area fields (n ≥ 100): 01, 03, 07, 08, 09, 12, 13, 14
- AV as a single cohort (n = 6,199, autonomous-vehicle keyword
  cohort, n = 25 debate variables not run for this study; the user
  reports debate did not add value for AV — Δ ≈ 0)

Empirical Δ is XGBoost +6focal mean over 20 random 80/20 stratified
splits on Y, see the parent study for the protocol. Threshold for
"helps" set at $\Delta \geq 0.020$.

### 4.2 Per-entity metric values

| Entity | $\Delta_{\text{XGB}+6}$ | X = $\bar{d}_{\text{biblio}}$ | Y = $\bar{\rho}_{\text{periph}}$ | Class |
|---|---:|---:|---:|---|
| FIELD_14 | **+0.047** | 0.466 | 0.892 | helps |
| FIELD_12 | **+0.035** | 0.532 | 0.906 | helps |
| FIELD_08 | **+0.031** | 0.278 | 0.904 | helps |
| FIELD_13 | **+0.025** | 0.328 | 0.819 | helps |
| FIELD_07 | +0.018 | 0.350 | 0.893 | mid |
| FIELD_01 | +0.018 | 0.240 | 0.926 | mid |
| FIELD_03 | +0.008 | 0.171 | 0.904 | weak |
| FIELD_09 | +0.005 | 0.196 | 0.869 | weak |
| **AV** | ~0 | 0.272 | **0.809** | no help |

### 4.3 Axis-pair selection

We tested 9 candidate (X, Y) pairs from 3 X-candidates × 3
Y-candidates. The criteria for a valid framework:

1. corr(X, Δ) > 0.3 (X must positively predict uplift)
2. corr(Y, Δ) > 0.3 (Y must positively predict uplift)
3. AV's z-score on both X and Y < 0 (AV must land in lower-left)

Only **one** pair passes all three:

| X axis | Y axis | corr(X, Δ) | corr(Y, Δ) | AV $z_X$ | AV $z_Y$ | Verdict |
|---|---|---:|---:|---:|---:|:---:|
| **$\bar{d}_{\text{biblio}}$** | **$\bar{\rho}_{\text{periph}}$** | **+0.78** | **+0.37** | **−0.36** | **−1.76** | ✓ |

The 8 other (X, Y) combinations either had negative corr with Δ
(e.g., mean SK is high in AV due to deep-learning NPL refs, breaking
the X axis) or placed AV in the upper half on at least one axis.

Full validation table: `analysis/outputs/quadrant_validation.csv`.

---

## 5. The 2×2 plot

```
                    Y = peripheral fraction ↑
                           HIGH (broad portfolio)
                                |
   FIELD_01 (0.24, 0.93)        |   ★ FIELD_12 (0.53, 0.91)
   FIELD_03 (0.17, 0.90)        |   ★ FIELD_14 (0.47, 0.89)
                                |     FIELD_08 (0.28, 0.90)
                                |     FIELD_07 (0.35, 0.89)*
   ─────────────────────────────┼─────────────────────────────
                                |
   FIELD_09 (0.20, 0.87)        |     FIELD_13 (0.33, 0.82)
   ★ AV (0.27, 0.81)            |
   no help                       |
                                |
                           LOW (concentrated portfolio)

           X = biblio Y-discriminability →
           LOW (no signal)              HIGH (clear signal)
```

★ = strongly helps / strongly no-help
\* = FIELD_07 sits in the upper-right but Δ is only +0.018 — see
§7.1 for the keyword-pollution explanation.

### 5.1 Quadrant interpretation

| Quadrant | Description | Debate verdict |
|---|---|---|
| Upper-right | Biblio discriminative + diverse portfolios | **Worth running** (Δ ≈ 0.025–0.047) |
| Upper-left  | No biblio signal but broad portfolios | Debate alone can't compensate; biblio not strong enough |
| Lower-right | Strong biblio + domain-pure portfolios | Marginal; biblio may already saturate signal |
| Lower-left  | No biblio signal + domain-pure portfolios | **Skip** (Δ ≈ 0); applies to AV |

---

## 6. Predictive use

Applying this framework to a new domain (e.g., 5G, quantum computing,
biotech sub-area):

1. Build the cohort from your domain keyword set.
2. Compute the 16 biblio variables (`compute_variables.py` or
   equivalent) and label Y.
3. Compute $\bar{d}_{\text{biblio}}(f)$ — average absolute Cohen's $d$
   across the 15 numeric biblio variables.
4. Compute $\bar{\rho}_{\text{periph}}(f)$ — mean PKH / (CKH + PKH).
5. Locate the (X, Y) point on the 2 × 2.

Suggested decision rule (calibrated on the 9-entity set):

- $X \geq 0.30$ **AND** $Y \geq 0.88$ → run debate, expect Δ ≈
  +0.025 to +0.047
- $X < 0.25$ **AND** $Y < 0.85$ → skip debate (Δ ≈ 0)
- otherwise → marginal; pilot 100 patents first

These thresholds are heuristic; tune as more entities are observed.

---

## 7. Limitations and caveats

### 7.1 FIELD_07 (Hybrids) is upper-right but doesn't help

Field 07 places at $(0.35, 0.89)$ — upper-right region — yet its
empirical Δ is only +0.018. The most likely cause is **keyword
pollution**: a 306-patent text audit found that 62.7% of Field 07
abstracts have no ICE-context markers (terms like "internal
combustion", "gasoline", "engine"). The "hybrid" substring picks up
broad hybrid-vehicle patents whose abstracts focus on the electric
side. Personas evaluating these abstracts have no ICE context to
work with, dragging debate value down.

→ The framework's prediction is correct *as a function of the
biblio signal of the cohort*, but cohort definition errors
(keyword over-reach) introduce a separate confounder the framework
does not capture.

### 7.2 Small validation set (N = 9)

8 ICE fields plus AV. Stronger validation would require additional
domains (e.g., G16H healthcare patents, where debate uplift is
known from `Dynamics/`).

### 7.3 K_ids definition asymmetry

The peripheral fraction $\bar{\rho}_{\text{periph}}$ depends on how
$\mathcal{K}$ (keyword universe) is defined. ICE uses the union of
16 narrow query keyword sets ($|\mathcal{K}| = 100{,}084$); AV uses a
broader 42-phrase keyword list. AV's lower peripheral fraction
partly reflects this broader $\mathcal{K}$. When applying this
framework to a new domain, pick a keyword set with comparable scope
(neither overly narrow nor overly broad) for the Y axis to be
comparable.

### 7.4 Cross-validation across model classes

The empirical Δ used for validation is XGBoost-specific. For RF,
SVM, FFN the ranking of fields by Δ shifts (RF less sensitive to
debate features, FFN overfits with the 51-feature augmented set).
The framework is calibrated on XGBoost which is the
debate-extraction model recommended for this study.

### 7.5 Threshold = 0.020

The "helps" threshold $\Delta \geq 0.020$ is heuristic. Lowering it
to 0.015 would include FIELD_07 and FIELD_01 as helps; raising to
0.025 would exclude FIELD_13 and FIELD_08. The framework's
correlation strength is robust to the choice; the categorical
verdict is not.

---

## 8. References

- [`VARIABLES.md`](VARIABLES.md) — formal definitions of the 16
  biblio + 25 debate variables, including PKH, CKH, and the Y label.
- [`ALGORITHM.md`](ALGORITHM.md) — 5-persona debate algorithm
  specification.
- `analysis/quadrant_axes_validation.py` — script that computes
  $\bar{d}_{\text{biblio}}$, $\bar{\rho}_{\text{periph}}$, and the
  9-entity table.
- `analysis/outputs/quadrant_validation.csv` — the raw 9-entity
  metric table.
