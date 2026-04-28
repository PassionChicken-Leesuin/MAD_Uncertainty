# Variables — Formal Definitions and Notation

Single source of truth for all variable formulas used in the ICE Domain
multi-agent debate uplift study.

This document is paired with:
- `data_collection/compute_variables.py` (bibliometric implementation)
- `debate/` (debate orchestration and variable extraction)
- The reference AV implementation at `../Autonomous_Vehicle/variables.md`

If the code disagrees with this document, **fix the code, not the doc**.

---

## Universal notation

| Symbol | Meaning |
|---|---|
| $X$ | A focal patent in the analysis universe |
| $t_X$ | Grant date of $X$, from `g_patent.patent_date` |
| $A(X)$ | Set of assignee ids of $X$, from `g_assignee_disambiguated.tsv` |
| $I(X)$ | Set of inventor ids of $X$, from `g_inventor_disambiguated.tsv` |
| $BC(X)$ | Backward citations of $X$ = $\{p : (X, p) \in \texttt{g\_us\_patent\_citation}\}$ — i.e., rows where `patent_id == X`, `citation_patent_id` gives the cited target |
| $FC_k(X)$ | $k$-year forward citations of $X$ = $\{z : (z, X) \in \texttt{g\_us\_patent\_citation},\ t_z \in (t_X,\ t_X + k\!\cdot\!\text{yr}]\}$ |
| $K_{\text{ids}}$ | Keyword universe — bulk patents matching the project's keyword definition (no date or IPC filter) |
| $P_a(t)$ | Patent portfolio of assignee $a$ truncated at time $t$ = $\{p : a \in \text{assignees}(p),\ t_p \leq t,\ p \neq X\}$ |

For the ICE project specifically:
- **Universe**: 7,086 distinct USPTO granted patents from the union of
  Sinigaglia (2022) Applied Energy 306, 118003's 16 search strategies,
  with `patent_date` $\in$ [1980-01-01, 2020-12-31].
- **$k = 5$** (5-year forward citation window).
- **$K_{\text{ids}}$** = union over the 16 ICE query keyword sets matched
  against bulk title + abstract (no date / IPC filter; `\b`
  word-boundary regex). Size: 100,084.
- **Bulk currency**: `g_patent.patent_date` $\leq$ 2025-12-30, so
  $t_X + 5\text{yr} \leq 2025\text{-}12\text{-}31$ is fully observed.

---

## Output: $Y$ (binary outcome)

$Y$ is the top-decile indicator of forward citation count.

$$
\text{forward5}(X) = |FC_5(X)| \qquad
q_{90} = \text{percentile}_{90}\{\text{forward5}(X) : X \in \text{Universe}\}
$$

$$
Y(X) = \begin{cases} 1 & \text{if } \text{forward5}(X) \geq q_{90} \\ 0 & \text{otherwise} \end{cases}
$$

**Boundary rule**: $\geq$ (ties at $q_{90}$ resolve to $Y=1$, matches AV convention).

For the ICE cohort: $q_{90} = 6.0$, positives $= 885 / 7{,}086 = 12.49\%$
(slightly above 10% due to ties).

---

## Bibliometric variables (16)

Lifted from `../Autonomous_Vehicle/variables.md` with `forward3 → forward5`
window adjustment. All other formulas identical.

### Group 1: Novelty (3 variables)

#### 1. CTO — Class-level Technological Originality

Herfindahl over IPC subclass of backward-cited patents
(Trajtenberg/Jaffe formulation: one primary class per cited patent).

**IPC subclass** = `section + ipc_class + subclass`, e.g.
`F` + `02` + `D` $\to$ `F02D`.

For each $c \in BC(X)$:
- $\text{primary\_sub}(c)$ = subclass of $c$'s row in `g_ipc_at_issue`
  with smallest `ipc_sequence`. No `classification_value` filter
  (inventional + non-inventional pooled).
- Drop $c$ if $\text{primary\_sub}(c)$ is empty.

Let $n$ = # surviving $c$, and for each subclass $j$:

$$
B_j = \frac{|\{c \in BC(X) : \text{primary\_sub}(c) = j\}|}{n}
$$

$$
\boxed{\text{CTO}(X) = 1 - \sum_j B_j^2}
$$

If $n = 0$: $\text{CTO} = \text{NaN}$.
By construction $\sum_j B_j = 1$, so $\text{CTO} \in [0,\ 1 - 1/n]$.

#### 2. STO — Main-group-level Technological Originality

Same as CTO at IPC **main_group** level.

**IPC main_group** = `section + ipc_class + subclass + main_group`,
e.g. `F02D` + `041` $\to$ `F02D041`.

$$
\boxed{\text{STO}(X) = 1 - \sum_j B_j^2}
$$

with $B_j$ = share of $c \in BC(X)$ whose $\text{primary\_mg}(c) = j$.
NaN if no usable cited main_group.

#### 3. PK — Prior Knowledge

$$
\boxed{\text{PK}(X) = |BC(X)|}
$$

Count of rows in `g_us_patent_citation` where `patent_id == X`.

### Group 2: Science Intensity (1 variable)

#### 4. SK — Scientific Knowledge

$$
\boxed{\text{SK}(X) = |\{r \in \texttt{g\_other\_reference} : r.\text{patent\_id} = X\}|}
$$

Number of non-patent literature (NPL) references — proxy for science
linkage.

### Group 3: Growth Speed (1 variable)

#### 5. TCT — Technology Cycle Time

$$
\boxed{\text{TCT}(X) = \text{median}\{(t_X - t_c) : c \in BC(X),\ t_c \text{ known}\}\ \text{in days}}
$$

NaN if no $c$ has known $t_c$.

### Group 4: Scope and Coverage (3 variables)

#### 6. MF — Main Field (categorical)

$$
\boxed{\text{MF}(X) = \text{main\_group of } X\text{'s IPC row with smallest ipc\_sequence}}
$$

In modeling: encoded as top-10 one-hot indicators + `MF_other`
fallback (so 11 columns in feature space).

#### 7. TS — Technological Scope

$$
\boxed{\text{TS}(X) = |\{\text{distinct main\_groups across all IPC rows of } X\}|}
$$

#### 8. NC — Number of Claims

$$
\boxed{\text{NC}(X) = \texttt{g\_patent.num\_claims}(X)}
$$

### Group 5: Development Effort and Capability (8 variables)

All eight apply a **time cutoff at $t_X$** to the assignee portfolio
to prevent leakage. Define:

$$
P_a(t_X) = \{p : a \in \text{assignees}(p),\ t_p \leq t_X,\ p \neq X\}
$$

$$
P(X) = \bigcup_{a \in A(X)} P_a(t_X) \qquad \text{(de-duplicated)}
$$

#### 9. COL — Collaboration

$$
\boxed{\text{COL}(X) = \begin{cases} 1 & \text{if } |A(X)| \geq 2 \\ 0 & \text{otherwise} \end{cases}}
$$

#### 10. INV — Inventors

$$
\boxed{\text{INV}(X) = |I(X)|}
$$

#### 11. TKH — Total Know-How

$$
\boxed{\text{TKH}(X) = |P(X)|}
$$

Total assignee portfolio size up to $t_X$, de-duplicated across $X$'s
multiple assignees.

#### 12. CKH — Core Area Know-How

$$
\boxed{\text{CKH}(X) = |P(X) \cap K_{\text{ids}}|}
$$

Portion of the assignee portfolio that lies in the keyword universe
(i.e., is "ICE-area").

#### 13. PKH — Peripheral Area Know-How

$$
\boxed{\text{PKH}(X) = \text{TKH}(X) - \text{CKH}(X)}
$$

#### 14. TTS — Total Technological Strength

For $p \in P(X)$, let

$$
\text{fc\_le}(p, t_X) = |\{z : (z, p) \in \texttt{g\_us\_patent\_citation},\ t_z \leq t_X\}|
$$

$$
\boxed{\text{TTS}(X) = \sum_{p \in P(X)} \text{fc\_le}(p, t_X)}
$$

Sum of forward citations on the assignee portfolio, with
**citing-side cutoff** at $t_X$.

**Leakage safety**: $Y$ counts citations with $t_z > t_X$ (within
$(t_X, t_X + 5\text{yr}]$), TTS counts citations with $t_z \leq t_X$.
The two windows are disjoint, so no overlap with $Y$.

#### 15. CTS — Core Area Technological Strength

$$
\boxed{\text{CTS}(X) = \sum_{p \in P(X) \cap K_{\text{ids}}} \text{fc\_le}(p, t_X)}
$$

#### 16. PTS — Peripheral Area Technological Strength

$$
\boxed{\text{PTS}(X) = \text{TTS}(X) - \text{CTS}(X)}
$$

### Algebraic invariants (verified at extraction time)

- $\text{TKH} = \text{CKH} + \text{PKH}$
- $\text{TTS} = \text{CTS} + \text{PTS}$
- $\text{CKH} \leq \text{TKH}$, $\text{CTS} \leq \text{TTS}$
- $\text{CTO}, \text{STO} \in [0, 1]$
- $\text{TS}, \text{NC} \geq 1$, $\text{COL} \in \{0, 1\}$, $\text{INV} \geq 0$

### Missing-data convention

CTO, STO, TCT are NaN when $X$ has $BC(X) = \emptyset$ or all cited
patents lack IPC / grant_date. In ICE: 7.17% (508 / 7,086 patents)
hit this case. Downstream median imputation is applied per train fold.

---

## Debate variables (25)

Source: `../Autonomous_Vehicle/debate/variables.py`. Adapted to a
**5-persona panel** (Technology, Application, User, Ecosystem,
BusinessModel) with 4-case moderator termination.

### Debate-specific notation

| Symbol | Meaning |
|---|---|
| `state` | DebateState for $X$ |
| $R$ | $= |\text{state.rounds}|$, total rounds played, $R \in \{2, 3, 4, 5, 6\}$ (Round 0 + 1..5 debate rounds) |
| $r_0$ | Round 0 outputs (initial predictions) |
| $r_f$ | Final round outputs (= state.rounds[-1]) |
| $\mathcal{D}_r$ | $r$-th round outputs = list of 5 persona dicts |
| $\text{pred}_a(r)$ | Persona $a$'s prediction in round $r$, $\in \{0, 1\}$ (1 = Promising) |
| $c_a(r)$ | Persona $a$'s `confidence_for` in round $r$, $\in [0, 1]$ |
| $\text{action}_{a \to b}(r)$ | $a$'s action targeting $b$ in round $r$, $\in \{\text{Support}, \text{Attack}\}$ |
| $\text{fact}_a(r)$ | Validator's fact-based judgment of $a$'s reason in round $r$, $\in \{0, 1\}$ |
| $\text{debate\_rounds}$ | $= R - 1$ (round 0 is initial; rounds 1..$R-1$ are "debate rounds") |
| $H(p)$ | Binary entropy = $-p \log_2 p - (1-p) \log_2 (1-p)$, with $H(0) = H(1) = 0$ |

Personas (n=5): $\Pi = \{\text{Technology}, \text{Application},
\text{User}, \text{Ecosystem}, \text{BusinessModel}\}$

Define helper sets:
- $\text{All}_{\text{pro}} = \{c_a(r) : \text{pred}_a(r) = 1, r \in [0, R)\}$
- $\text{All}_{\text{anti}} = \{c_a(r) : \text{pred}_a(r) = 0, r \in [0, R)\}$
- $\text{gap}(r) = | \text{mean}\{c_a(r) : \text{pred}_a(r) = 1\} - \text{mean}\{c_a(r) : \text{pred}_a(r) = 0\} |$

### Group 1: Confidence (7 variables)

#### 1. mean_conf_pro

$$
\boxed{\text{mean\_conf\_pro} = \text{mean}(\text{All}_{\text{pro}})}
$$

(0.0 if $\text{All}_{\text{pro}} = \emptyset$.)

#### 2. mean_conf_anti

$$
\boxed{\text{mean\_conf\_anti} = \text{mean}(\text{All}_{\text{anti}})}
$$

#### 3. var_conf_pro

$$
\boxed{\text{var\_conf\_pro} = \text{Var}(\text{All}_{\text{pro}}) \text{ with ddof}=0}
$$

(0.0 if $|\text{All}_{\text{pro}}| \leq 1$.)

#### 4. var_conf_anti

$$
\boxed{\text{var\_conf\_anti} = \text{Var}(\text{All}_{\text{anti}}) \text{ with ddof}=0}
$$

#### 5. conf_gap_change

$$
\boxed{\text{conf\_gap\_change} = \text{gap}(r_0) - \text{gap}(r_f)}
$$

How much the pro/anti confidence gap shrunk from the initial round to
the final round. Positive value = personas converged.

#### 6. H_final

$$
\boxed{\text{H\_final} = \frac{1}{|\Pi|} \sum_{a \in \Pi} H(c_a(r_f))}
$$

Average per-persona binary entropy in the final round.

#### 7. delta_H

$$
\boxed{\text{delta\_H} = H(r_0) - H(r_f) \text{ where } H(r) = \frac{1}{|\Pi|} \sum_{a \in \Pi} H(c_a(r))}
$$

How much average uncertainty decreased over the debate. Positive =
debate produced more confident predictions overall.

### Group 2: Argument Graph (5 variables)

For round $r \in [1, R)$, each persona issues actions targeting other
personas (or self). Define:

For $a, b \in \Pi$, $r \in [1, R)$:
- $\text{is\_same}(a, b) = (a = b)$
- $\text{is\_cross}(a, b) = (a \neq b)$

Aggregate counts over all $(a, b, r)$ where $a$ issued action targeting $b$ in round $r$:

$$
\begin{aligned}
\text{cross\_s} &= |\{(a, b, r) : \text{is\_cross}(a,b) \land \text{action}_{a \to b}(r) = \text{Support}\}| \\
\text{cross\_a} &= |\{(a, b, r) : \text{is\_cross}(a,b) \land \text{action}_{a \to b}(r) = \text{Attack}\}| \\
\text{same\_s}  &= |\{(a, b, r) : \text{is\_same}(a,b) \land \text{action}_{a \to b}(r) = \text{Support}\}| \\
\text{same\_a}  &= |\{(a, b, r) : \text{is\_same}(a,b) \land \text{action}_{a \to b}(r) = \text{Attack}\}|
\end{aligned}
$$

Also define receiving counts (supports received in round $r$ from any
source):
$$
\begin{aligned}
\text{pro\_recv}  &= |\{(a, b, r) : \text{action}_{a \to b}(r) = \text{Support} \land \text{pred}_b(r-1) = 1\}| \\
\text{anti\_recv} &= |\{(a, b, r) : \text{action}_{a \to b}(r) = \text{Support} \land \text{pred}_b(r-1) = 0\}|
\end{aligned}
$$

If $\text{debate\_rounds} = 0$: all = 0.0. Else:

#### 8. cross_domain_support

$$
\boxed{\text{cross\_domain\_support} = \text{cross\_s} / \text{debate\_rounds}}
$$

#### 9. cross_domain_attack

$$
\boxed{\text{cross\_domain\_attack} = \text{cross\_a} / \text{debate\_rounds}}
$$

#### 10. same_domain_support

$$
\boxed{\text{same\_domain\_support} = \text{same\_s} / \text{debate\_rounds}}
$$

#### 11. same_domain_attack

$$
\boxed{\text{same\_domain\_attack} = \text{same\_a} / \text{debate\_rounds}}
$$

#### 12. acceptability_gap

$$
\boxed{\text{acceptability\_gap} = (\text{pro\_recv} - \text{anti\_recv}) / \text{debate\_rounds}}
$$

Positive value = pro side received more support across the debate.

### Group 3: Validator (2 variables)

For each round $r$ and persona $a$, the validator emits
$\text{fact}_a(r) \in \{0, 1\}$ ("is the persona's reason fact-based?").
Aggregate:

$$
\begin{aligned}
\text{fact\_pro}  &= \sum_{r \in [0, R)} \sum_{a : \text{pred}_a(r) = 1} \text{fact}_a(r) \\
\text{fact\_anti} &= \sum_{r \in [0, R)} \sum_{a : \text{pred}_a(r) = 0} \text{fact}_a(r)
\end{aligned}
$$

#### 13. fact_ratio_pro

$$
\boxed{\text{fact\_ratio\_pro} = \text{fact\_pro} / R}
$$

#### 14. fact_ratio_anti

$$
\boxed{\text{fact\_ratio\_anti} = \text{fact\_anti} / R}
$$

(Both 0.0 if $R = 0$.)

### Group 4: Prediction (7 variables)

#### 15. final_prediction

$$
\boxed{\text{final\_prediction} = \texttt{state.final\_prediction} \in \{0, 1\}}
$$

Moderator-decided final prediction: majority vote (5:0 or 4:1) when
consensus reached, or forced-majority (3:2) at round 5.

#### 16-20. final_pred_{technology, application, user, ecosystem, businessmodel}

$$
\boxed{\text{final\_pred}_a = \text{pred}_a(r_f) \in \{0, 1\} \quad \forall a \in \Pi}
$$

Each of the 5 personas' final-round prediction (5 columns).

#### 21. prediction_volatility

For each round $r$, ratio$(r) = (\text{# personas predicting 1 in } r) / 5$.

If $\text{debate\_rounds} = 0$: 0. Else:

$$
\boxed{\text{prediction\_volatility} = \frac{\sum_{i=1}^{R-1} |\text{ratio}(i) - \text{ratio}(i-1)|}{\text{debate\_rounds}}}
$$

Captures how much the pro-supporter ratio fluctuated across rounds.

### Group 5: Moderator (3 variables)

The moderator applies the n=5 termination rule (see config). Termination
reason ∈ {unanimous, consensus_mid, consensus_late, forced_majority},
with the round-1 unanimous case singled out:

#### 22. total_rounds

$$
\boxed{\text{total\_rounds} = \texttt{state.termination\_round} \in \{1, 2, 3, 4, 5\}}
$$

#### 23. term_unanimous

$$
\boxed{\text{term\_unanimous} = \begin{cases} 1 & \text{if } \texttt{termination\_reason} = \text{unanimous} \\ 0 & \text{otherwise} \end{cases}}
$$

Round-1 immediate consensus indicator (5:0 or 4:1 in the very first
debate round).

#### 24. term_extended_debate

$$
\boxed{\text{term\_extended\_debate} = 1 - \text{term\_unanimous}}
$$

Indicator of any termination after round 1 (= consensus_mid OR
consensus_late OR forced_majority).

### Group 6: Text (1 variable)

#### 25. semantic_coherence

Embed the 5 reason strings $\{\text{reason}_a(r_f) : a \in \Pi\}$
using OpenAI `text-embedding-3-small`. Compute pairwise cosine
similarity for all $\binom{5}{2} = 10$ pairs. Average.

$$
\boxed{\text{semantic\_coherence} = \frac{1}{\binom{5}{2}} \sum_{i < j} \cos(\text{emb}_i,\ \text{emb}_j)}
$$

Higher value = personas' reasoning text more similar in latent
embedding space.

---

## Variable count summary

| Category | Count | Variables |
|---|---|---|
| Bibliometric (Phase 0) | **16** | CTO, STO, PK, SK, TCT, MF, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS |
| Debate confidence | 7 | mean_conf_pro/anti, var_conf_pro/anti, conf_gap_change, H_final, delta_H |
| Debate argument graph | 5 | cross_domain_support/attack, same_domain_support/attack, acceptability_gap |
| Debate validator | 2 | fact_ratio_pro, fact_ratio_anti |
| Debate prediction | 7 | final_prediction, final_pred_{tech,app,user,eco,biz}, prediction_volatility |
| Debate moderator | 3 | total_rounds, term_unanimous, term_extended_debate |
| Debate text | 1 | semantic_coherence |
| **Total debate** | **25** | |
| **Output** | 1 | $Y$ (top-decile of forward5) |
| **Grand total per patent** | **42** | |

In modeling, MF is one-hot-encoded as 11 binary columns
(top-10 + 'other'), so the **feature dimension** in the augmented
(BIBLIO + 25 debate) model is $15 + 11 + 25 = 51$.

---

## Implementation references

| Component | File |
|---|---|
| Bibliometric extraction | `data_collection/compute_variables.py` |
| Bibliometric verification (per-variable AV mapping) | `data_collection/VERIFICATION.md` |
| Debate orchestration | `debate/main.py`, `debate/debate.py`, `debate/agents.py` |
| Debate variable extraction | `debate/variables.py::VariableExtractor` |
| Persona prompts | `debate/config.py::AGENT_PERSONAS` |
| Validator prompt | `debate/config.py::VALIDATOR_SYSTEM_PROMPT` |
| Moderator termination logic | `debate/agents.py::Moderator` |
| ICE-specific decisions and overrides | `CLAUDE.md` |

## Key references in the literature

- Sinigaglia, T., Martins, M. E. S., & Siluk, J. C. M. (2022).
  Technological evolution of internal combustion engine vehicle: A
  patent data analysis. *Applied Energy*, 306, 118003.
- Trajtenberg, M., Henderson, R., & Jaffe, A. (1997). University versus
  corporate patents: A window on the basicness of invention. *Economics
  of Innovation and New Technology*, 5(1), 19–50.
  (CTO/STO formulation)
- Kapoor, R., et al. — 5 sources of uncertainty framework (basis for
  the 5 personas).
