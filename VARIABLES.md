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
| $BC(X)$ | Backward citations of $X$ — set of patents $X$ cites |
| $FC_k(X)$ | Forward citations of $X$ within $k$ years — patents $z$ that cite $X$ with $t_z \in (t_X,\ t_X + k\!\cdot\!\text{yr}]$ |
| $\mathcal{K}$ | Keyword universe — bulk patents matching the project's keyword definition (no date or IPC filter) |
| $P_a(t)$ | Portfolio of assignee $a$ truncated at time $t$: $\lbrace p : a \in A(p),\ t_p \leq t,\ p \neq X \rbrace$ |

For the ICE project specifically:

- **Universe**: 7,086 distinct USPTO granted patents from the union of
  Sinigaglia (2022) Applied Energy 306, 118003's 16 search strategies,
  with `patent_date` $\in$ [1980-01-01, 2020-12-31].
- $k = 5$ (5-year forward citation window).
- $\mathcal{K}$ = union over the 16 ICE query keyword sets matched
  against bulk title + abstract (no date or IPC filter; word-boundary
  regex). $|\mathcal{K}| = 100{,}084$.
- **Bulk currency**: `g_patent.patent_date` $\leq$ 2025-12-30, so
  $t_X + 5\,\text{yr} \leq 2025\text{-}12\text{-}31$ is fully observed
  for every patent in the cohort.

Citation direction (from `g_us_patent_citation.tsv`):
- `patent_id` is the **citing** patent
- `citation_patent_id` is the **cited** patent
- $BC(X)$ = rows where `patent_id == X`
- $FC_k(X)$ = rows where `citation_patent_id == X`, filtered by date

---

## Output: $Y$ (binary outcome)

Let $\phi_5(X) = |FC_5(X)|$ be the 5-year forward citation count of $X$,
and let $q_{90}$ be the 90-th percentile of $\phi_5$ over the universe.

$$
Y(X) = \mathbb{1}\!\left[\phi_5(X) \geq q_{90}\right]
$$

**Boundary rule**: $\geq$ (ties at $q_{90}$ resolve to $Y=1$, matches AV
convention).

For the ICE cohort: $q_{90} = 6.0$, positives $= 885 / 7{,}086 = 12.49\%$
(slightly above 10% due to ties).

---

## Bibliometric variables (16)

Lifted from `../Autonomous_Vehicle/variables.md` with the forward window
adjusted from 3 years to 5 years. All other formulas identical.

In each variable's section, the formula uses these per-cited-patent
helpers:
- $\sigma(c)$ = primary IPC subclass of cited patent $c$ (the subclass
  of $c$'s row in `g_ipc_at_issue` with smallest `ipc_sequence`,
  no `classification_value` filter). Empty if the row's subclass
  components are empty.
- $\mu(c)$ = primary IPC main_group of $c$ (analogous, at main_group
  level).
- IPC subclass = `section + ipc_class + subclass` (e.g. F02D).
- IPC main_group = `section + ipc_class + subclass + main_group`
  (e.g. F02D041).

### Group 1: Novelty (3)

#### 1. CTO — Class-level Technological Originality

Herfindahl over the IPC subclass of each backward-cited patent
(Trajtenberg / Jaffe formulation: one primary class per cited patent).

Let $S(X) = \lbrace c \in BC(X) : \sigma(c) \neq \emptyset \rbrace$ and $n = |S(X)|$.
For each subclass $j$:

$$
B_j = \frac{|\lbrace c \in S(X) : \sigma(c) = j \rbrace|}{n}
$$

$$
\boxed{\ \mathrm{CTO}(X) = 1 - \sum_j B_j^2\ }
$$

If $n = 0$: $\mathrm{CTO}(X) = \mathrm{NaN}$.
By construction $\sum_j B_j = 1$, so $\mathrm{CTO} \in [0,\ 1 - 1/n]$.

#### 2. STO — Main-group-level Technological Originality

Same as CTO at IPC main_group level. Let
$T(X) = \lbrace c \in BC(X) : \mu(c) \neq \emptyset \rbrace$ and $m = |T(X)|$.
For each main_group $j$:

$$
B_j = \frac{|\lbrace c \in T(X) : \mu(c) = j \rbrace|}{m}, \qquad
\boxed{\ \mathrm{STO}(X) = 1 - \sum_j B_j^2\ }
$$

NaN if $m = 0$.

#### 3. PK — Prior Knowledge

$$
\boxed{\ \mathrm{PK}(X) = |BC(X)|\ }
$$

Count of rows in `g_us_patent_citation` where `patent_id == X`.

### Group 2: Science Intensity (1)

#### 4. SK — Scientific Knowledge

$$
\boxed{\ \mathrm{SK}(X) = |\lbrace r \in \mathcal{R} : r.\mathrm{patent\,id} = X \rbrace|\ }
$$

where $\mathcal{R}$ is the table `g_other_reference.tsv` (NPL refs).
Proxy for science linkage.

### Group 3: Growth Speed (1)

#### 5. TCT — Technology Cycle Time

Let $D(X) = \lbrace c \in BC(X) : t_c \text{ is known} \rbrace$.

$$
\boxed{\ \mathrm{TCT}(X) = \mathrm{median}\!\left\lbrace  (t_X - t_c) : c \in D(X) \right \rbrace\ \text{(in days)}\ }
$$

NaN if $D(X) = \emptyset$.

### Group 4: Scope and Coverage (3)

#### 6. MF — Main Field (categorical)

$$
\boxed{\ \mathrm{MF}(X) = \mu^{*}(X)\ }
$$

where $\mu^{*}(X)$ is the IPC main_group of $X$'s row with smallest
`ipc_sequence`. Categorical. In modeling: encoded as top-10 one-hot
indicators plus an `MF_other` fallback (so 11 columns in feature space).

#### 7. TS — Technological Scope

$$
\boxed{\ \mathrm{TS}(X) = |\lbrace \mathrm{main\ group}(\rho) : \rho \in \mathrm{IPC}(X) \rbrace|\ }
$$

Number of distinct main_groups across all IPC rows of $X$.

#### 8. NC — Number of Claims

$$
\boxed{\ \mathrm{NC}(X) = \mathrm{num\ claims}(X)\ }
$$

From `g_patent.num_claims`.

### Group 5: Development Effort and Capability (8)

All eight apply a **time cutoff at $t_X$** to the assignee portfolio
to prevent leakage. Define:

$$
P_a(t_X) = \lbrace p : a \in A(p),\ t_p \leq t_X,\ p \neq X \rbrace
$$

$$
P(X) = \bigcup_{a \in A(X)} P_a(t_X) \qquad (\text{de-duplicated})
$$

#### 9. COL — Collaboration

$$
\boxed{\ \mathrm{COL}(X) = \mathbb{1}\!\left[|A(X)| \geq 2\right]\ }
$$

#### 10. INV — Inventors

$$
\boxed{\ \mathrm{INV}(X) = |I(X)|\ }
$$

#### 11. TKH — Total Know-How

$$
\boxed{\ \mathrm{TKH}(X) = |P(X)|\ }
$$

Total assignee portfolio size up to $t_X$, de-duplicated across $X$'s
multiple assignees.

#### 12. CKH — Core Area Know-How

$$
\boxed{\ \mathrm{CKH}(X) = |P(X) \cap \mathcal{K}|\ }
$$

Portion of the portfolio that lies in the keyword universe (i.e., is
"ICE-area" in this study).

#### 13. PKH — Peripheral Area Know-How

$$
\boxed{\ \mathrm{PKH}(X) = \mathrm{TKH}(X) - \mathrm{CKH}(X)\ }
$$

#### 14. TTS — Total Technological Strength

For $p \in P(X)$, define the **bounded forward count** with
citing-side cutoff $\tau$:

$$
F_{\le}(p,\ \tau) = |\lbrace z : (z,\ p) \in \mathrm{Citation},\ t_z \leq \tau \rbrace|
$$

Then:

$$
\boxed{\ \mathrm{TTS}(X) = \sum_{p \in P(X)} F_{\le}(p,\ t_X)\ }
$$

**Leakage safety**: $Y$ counts citing events with $t_z > t_X$ (within
$(t_X,\ t_X + 5\,\text{yr}]$); $\mathrm{TTS}$ counts events with
$t_z \leq t_X$. The two windows are disjoint, so no overlap with $Y$.

#### 15. CTS — Core Area Technological Strength

$$
\boxed{\ \mathrm{CTS}(X) = \sum_{p \in P(X) \cap \mathcal{K}} F_{\le}(p,\ t_X)\ }
$$

#### 16. PTS — Peripheral Area Technological Strength

$$
\boxed{\ \mathrm{PTS}(X) = \mathrm{TTS}(X) - \mathrm{CTS}(X)\ }
$$

### Algebraic invariants (verified at extraction time)

- $\mathrm{TKH} = \mathrm{CKH} + \mathrm{PKH}$
- $\mathrm{TTS} = \mathrm{CTS} + \mathrm{PTS}$
- $\mathrm{CKH} \leq \mathrm{TKH}$, $\mathrm{CTS} \leq \mathrm{TTS}$
- $\mathrm{CTO},\ \mathrm{STO} \in [0, 1]$
- $\mathrm{TS},\ \mathrm{NC} \geq 1$, $\mathrm{COL} \in \lbrace 0, 1 \rbrace$, $\mathrm{INV} \geq 0$

### Missing-data convention

$\mathrm{CTO},\ \mathrm{STO},\ \mathrm{TCT}$ are NaN when $X$ has
$BC(X) = \emptyset$ or all cited patents lack IPC / grant date.
In ICE: 7.17% (508 / 7,086 patents) hit this case. Downstream median
imputation is applied per train fold.

---

## Debate variables (25)

Source: `../Autonomous_Vehicle/debate/variables.py`. Adapted to a
**5-persona panel** (Technology, Application, User, Ecosystem,
BusinessModel) with 4-case moderator termination.

### Debate-specific notation

| Symbol | Meaning |
|---|---|
| $\Pi$ | Persona set, $|\Pi| = 5$ |
| $a, b$ | A persona (element of $\Pi$) |
| $R$ | Total number of rounds played, $R \in \lbrace 2, 3, 4, 5, 6 \rbrace$ (round 0 + 1..5) |
| $r$ | A round index, $r \in [0, R)$ |
| $r_0,\ r_f$ | First (initial) round, final round |
| $\pi_a(r) \in \lbrace 0, 1 \rbrace$ | Persona $a$'s prediction in round $r$ (1 = Promising) |
| $c_a(r) \in [0, 1]$ | Persona $a$'s `confidence_for` in round $r$ |
| $\alpha_{a \to b}(r) \in \lbrace \mathrm{Sup}, \mathrm{Att} \rbrace$ | $a$'s action targeting $b$ in round $r$ |
| $\phi_a(r) \in \lbrace 0, 1 \rbrace$ | Validator's binary fact-based judgment of $a$'s reason in round $r$ |
| $D = R - 1$ | Debate-round count (rounds 1..$R-1$, excluding round 0) |
| $H(p)$ | Binary entropy: $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$, with $H(0) = H(1) = 0$ |

Helper sets:

$$
\mathcal{A}_+ = \lbrace (a,r) : \pi_a(r) = 1 \rbrace,\qquad
\mathcal{A}_- = \lbrace (a,r) : \pi_a(r) = 0 \rbrace
$$

$$
\mathrm{gap}(r) = \left|\,\mathrm{mean}\lbrace c_a(r) : \pi_a(r) = 1 \rbrace
                   - \mathrm{mean}\lbrace c_a(r) : \pi_a(r) = 0 \rbrace\,\right|
$$

### Group 1: Confidence (7)

#### 1. mean_conf_pro

$$
\boxed{\ \frac{1}{|\mathcal{A}_+|} \sum_{(a,r) \in \mathcal{A}_+} c_a(r)\ }
$$

Mean confidence-for value among (persona, round) pairs where the
persona predicted promising. Returns 0 if $\mathcal{A}_+ = \emptyset$.

#### 2. mean_conf_anti

$$
\boxed{\ \frac{1}{|\mathcal{A}_-|} \sum_{(a,r) \in \mathcal{A}_-} c_a(r)\ }
$$

#### 3. var_conf_pro

$$
\boxed{\ \mathrm{Var}\!\left(\lbrace c_a(r) : (a,r) \in \mathcal{A}_+ \rbrace\right)\ \text{with ddof}=0\ }
$$

Returns 0 if $|\mathcal{A}_+| \leq 1$.

#### 4. var_conf_anti

$$
\boxed{\ \mathrm{Var}\!\left(\lbrace c_a(r) : (a,r) \in \mathcal{A}_- \rbrace\right)\ \text{with ddof}=0\ }
$$

#### 5. conf_gap_change

$$
\boxed{\ \mathrm{gap}(r_0) - \mathrm{gap}(r_f)\ }
$$

How much the pro / anti confidence gap shrunk from the initial round to
the final round. Positive = personas converged.

#### 6. H_final

$$
\boxed{\ \frac{1}{|\Pi|} \sum_{a \in \Pi} H(c_a(r_f))\ }
$$

Average per-persona binary entropy in the final round.

#### 7. delta_H

Define round-level mean entropy
$\bar{H}(r) = \frac{1}{|\Pi|}\sum_{a \in \Pi} H(c_a(r))$.

$$
\boxed{\ \bar{H}(r_0) - \bar{H}(r_f)\ }
$$

How much average uncertainty decreased over the debate. Positive =
debate produced more confident predictions overall.

### Group 2: Argument Graph (5)

For round $r \in [1, R)$, each persona issues actions targeting other
personas (or self). Aggregate counts over all triples $(a, b, r)$ where
$a$ issued an action targeting $b$ in round $r$:

$$
\begin{aligned}
\mathrm{cs} &= |\lbrace (a,b,r) : a \neq b,\ \alpha_{a \to b}(r) = \mathrm{Sup} \rbrace|\\
\mathrm{ca} &= |\lbrace (a,b,r) : a \neq b,\ \alpha_{a \to b}(r) = \mathrm{Att} \rbrace|\\
\mathrm{ss} &= |\lbrace (a,b,r) : a = b,\ \alpha_{a \to b}(r) = \mathrm{Sup} \rbrace|\\
\mathrm{sa} &= |\lbrace (a,b,r) : a = b,\ \alpha_{a \to b}(r) = \mathrm{Att} \rbrace|
\end{aligned}
$$

Receiving counts:

$$
\begin{aligned}
\mathrm{rp} &= |\lbrace (a,b,r) : \alpha_{a \to b}(r) = \mathrm{Sup} \land \pi_b(r-1) = 1 \rbrace|\\
\mathrm{rn} &= |\lbrace (a,b,r) : \alpha_{a \to b}(r) = \mathrm{Sup} \land \pi_b(r-1) = 0 \rbrace|
\end{aligned}
$$

If $D = 0$: all five variables = 0.0. Else:

#### 8. cross_domain_support

$$
\boxed{\ \mathrm{cs} / D\ }
$$

#### 9. cross_domain_attack

$$
\boxed{\ \mathrm{ca} / D\ }
$$

#### 10. same_domain_support

$$
\boxed{\ \mathrm{ss} / D\ }
$$

#### 11. same_domain_attack

$$
\boxed{\ \mathrm{sa} / D\ }
$$

#### 12. acceptability_gap

$$
\boxed{\ (\mathrm{rp} - \mathrm{rn}) / D\ }
$$

Positive value = pro side received more support across the debate.

### Group 3: Validator (2)

For each round $r$ and persona $a$, the validator emits
$\phi_a(r) \in \lbrace 0, 1 \rbrace$. Aggregate:

$$
\Phi_+ = \sum_{r \in [0, R)} \sum_{a \,:\, \pi_a(r) = 1} \phi_a(r),
\qquad
\Phi_- = \sum_{r \in [0, R)} \sum_{a \,:\, \pi_a(r) = 0} \phi_a(r)
$$

#### 13. fact_ratio_pro

$$
\boxed{\ \Phi_+ / R\ }
$$

#### 14. fact_ratio_anti

$$
\boxed{\ \Phi_- / R\ }
$$

(Both 0 if $R = 0$.)

### Group 4: Prediction (7)

#### 15. final_prediction

The moderator's final decision after termination, $\in \lbrace 0, 1 \rbrace$:
majority vote (5:0 or 4:1) when consensus is reached, or
forced-majority (3:2) at round 5.

$$
\boxed{\ \mathrm{state.final\ prediction}\ }
$$

#### 16-20. final_pred_{technology, application, user, ecosystem, businessmodel}

Each persona's prediction in the final round, $\in \lbrace 0, 1 \rbrace$:

$$
\boxed{\ \pi_a(r_f) \quad \forall a \in \Pi\ }
$$

(5 columns, one per persona.)

#### 21. prediction_volatility

For each round $r$, define the pro-supporter ratio:

$$
\rho(r) = \frac{|\lbrace a \in \Pi : \pi_a(r) = 1 \rbrace|}{|\Pi|}
$$

If $D = 0$: 0. Else:

$$
\boxed{\ \frac{1}{D} \sum_{i=1}^{R-1} |\rho(i) - \rho(i-1)|\ }
$$

Captures how much the pro-supporter ratio fluctuated across rounds.

### Group 5: Moderator (3)

Termination reason $\tau$ takes one of four values:
{`unanimous`, `consensus_mid`, `consensus_late`, `forced_majority`}.

#### 22. total_rounds

$$
\boxed{\ \mathrm{state.termination\ round} \in \lbrace 1, 2, 3, 4, 5 \rbrace\ }
$$

The round at which termination occurred.

#### 23. term_unanimous

$$
\boxed{\ \mathbb{1}\!\left[\tau = \mathtt{unanimous}\right]\ }
$$

Round-1 immediate consensus (5:0 or 4:1 in the very first debate
round).

#### 24. term_extended_debate

$$
\boxed{\ 1 - \mathbb{1}\!\left[\tau = \mathtt{unanimous}\right]\ }
$$

Indicator of any termination after round 1
(= `consensus_mid` ∨ `consensus_late` ∨ `forced_majority`).

### Group 6: Text (1)

#### 25. semantic_coherence

Embed the 5 reason strings $\lbrace \mathrm{reason}_a(r_f) : a \in \Pi \rbrace$
using OpenAI `text-embedding-3-small` to obtain
$e_1, \ldots, e_5 \in \mathbb{R}^d$. Compute pairwise cosine
similarity for all $\binom{5}{2} = 10$ pairs and average:

$$
\boxed{\ \frac{1}{\binom{5}{2}} \sum_{1 \leq i < j \leq 5}
       \frac{e_i \cdot e_j}{\|e_i\|\,\|e_j\|}\ }
$$

Higher value = personas' reasoning text more similar in latent
embedding space.

---

## Variable count summary

| Category | Count | Variables |
|---|---|---|
| Bibliometric (Phase 0) | **16** | CTO, STO, PK, SK, TCT, MF, TS, NC, COL, INV, TKH, CKH, PKH, TTS, CTS, PTS |
| Debate confidence | 7 | mean_conf_pro / anti, var_conf_pro / anti, conf_gap_change, H_final, delta_H |
| Debate argument graph | 5 | cross_domain_support / attack, same_domain_support / attack, acceptability_gap |
| Debate validator | 2 | fact_ratio_pro, fact_ratio_anti |
| Debate prediction | 7 | final_prediction, final_pred_{technology, application, user, ecosystem, businessmodel}, prediction_volatility |
| Debate moderator | 3 | total_rounds, term_unanimous, term_extended_debate |
| Debate text | 1 | semantic_coherence |
| **Total debate** | **25** | |
| **Output** | 1 | $Y$ (top-decile of $\phi_5$) |
| **Grand total per patent** | **42** | |

In modeling, MF is one-hot-encoded as 11 binary columns (top-10 +
"other"), so the **feature dimension** in the augmented (BIBLIO + 25
debate) model is $15 + 11 + 25 = 51$.

---

## Implementation references

| Component | File |
|---|---|
| Bibliometric extraction | `data_collection/compute_variables.py` |
| Bibliometric verification (per-variable AV mapping) | `data_collection/VERIFICATION.md` |
| Debate orchestration | `debate/main.py`, `debate/debate.py`, `debate/agents.py` |
| Debate variable extraction | `debate/variables.py` (class `VariableExtractor`) |
| Persona prompts | `debate/config.py` (`AGENT_PERSONAS`) |
| Validator prompt | `debate/config.py` (`VALIDATOR_SYSTEM_PROMPT`) |
| Moderator termination logic | `debate/agents.py` (class `Moderator`) |
| ICE-specific decisions and overrides | `CLAUDE.md` |

## Key references in the literature

- Sinigaglia, T., Martins, M. E. S., & Siluk, J. C. M. (2022).
  Technological evolution of internal combustion engine vehicle: A
  patent data analysis. *Applied Energy*, 306, 118003.
- Trajtenberg, M., Henderson, R., & Jaffe, A. (1997). University versus
  corporate patents: A window on the basicness of invention. *Economics
  of Innovation and New Technology*, 5(1), 19–50. (CTO / STO
  formulation)
- Kapoor, R., et al. — 5 sources of uncertainty framework (basis for
  the 5 personas).
