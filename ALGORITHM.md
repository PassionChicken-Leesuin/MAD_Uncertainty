# ALGORITHM.md — Multi-Agent Debate Protocol (n = 5)

Operational specification of the debate algorithm used in the ICE
multi-agent uplift study. Per-patent algorithm: takes a single patent
(title + abstract) as input, runs a 5-persona structured debate with
a fact-validator and rule-based moderator, and produces a
`DebateState` containing the full round-by-round trace plus a final
binary prediction.

The variable extraction layer that consumes `DebateState` is
documented separately in `VARIABLES.md`.

---

## 1. Components

**Expert personas** (n = 5, $\Pi$):

| Persona | Source of uncertainty (Kapoor et al.) |
|---|---|
| Technology | Scientific discovery and technical feasibility |
| Application | Domain-deployment value and reachability |
| User | Adoption, behavior, willingness-to-pay |
| Ecosystem | Complementors, infrastructure, standards |
| BusinessModel | Capture mechanisms, supply chain, pricing |

**Validator**: independent LLM that judges whether each persona's
reason text is fact-based (grounded in the patent disclosure) or
speculative.

**Moderator**: rule-based logic (no LLM call) that decides
termination at the end of each round.

**Domain-leakage rule**: persona / validator prompts must not contain
words identifying the dataset (`autonomous`, `vehicle`, `EV`, `ICE`,
`combustion`, `engine`, `hybrid`, `turbo`, etc.). Personas evaluate
patent text alone.

## 2. LLM settings

| Setting | Value | Source |
|---|---|---|
| `MODEL_NAME` | `gpt-4o-mini` | `debate/config.py` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | (used for `semantic_coherence`) |
| `TEMPERATURE_AGENT` | 0.7 | personas |
| `TEMPERATURE_VALIDATOR` | 0.3 | validator |
| `MAX_DEBATE_ROUNDS` | 6 | round 0 + debate rounds 1..5 |
| `MODERATOR_FORCE_ROUND` | 5 | last round always terminates |
| `MAX_RETRIES` | 20 | per single LLM call |
| `RANDOM_SEED` | 42 | for shuffles inside the orchestrator |

## 3. Output schema (`DebateState`)

| Field | Type | Description |
|---|---|---|
| `patent_id` | str | USPTO grant id |
| `patent_title` | str | from `g_patent` |
| `patent_abstract` | str | from `g_patent_abstract` |
| `rounds` | `list[list[dict]]` | per-round per-persona outputs |
| `validations` | `list[list[dict]]` | per-round per-persona `fact_based` ∈ {0, 1} |
| `termination_round` | int | round at which moderator stopped, ∈ {1, 2, 3, 4, 5} |
| `termination_reason` | str | one of {`unanimous`, `consensus_mid`, `consensus_late`, `forced_majority`} |
| `final_prediction` | int | ∈ {0, 1} |
| `final_confidence` | float | aggregate confidence of the winning side |

Each persona output (one `dict` in `rounds[r]`) carries:

| Key | Type | Constraint |
|---|---|---|
| `agent` | str | one of the 5 persona names |
| `prediction` | int | ∈ {0, 1} (1 = Promising) |
| `confidence_for` | float | ∈ [0.50, 1.00] (own-prediction confidence) |
| `confidence_against` | float | = 1.0 − `confidence_for` |
| `reason` | str | ≤ 2 sentences |
| `actions` | `list[dict]` | round 0: empty; round ≥ 1: 5 entries (one per target including self) |
| `fact_based` | int | ∈ {0, 1} (validator-injected after the round) |

Each `actions[i]` carries:

| Key | Type | Constraint |
|---|---|---|
| `target_agent` | str | persona name |
| `action` | str | ∈ {`Support`, `Attack`} |
| `action_reason` | str | 1 sentence |

---

## 4. Pseudocode

```
입력: 특허 (제목, 초록)
출력: DebateState (전체 라운드 기록, 종료 사유, 종료 라운드, 최종 예측값)

에이전트: {Technology, Application, User, Ecosystem, BusinessModel}  (n = 5)
검증자: Validator (각 에이전트 근거 텍스트의 사실 기반 여부 판정)
조정자: Moderator (n=5 종료 규칙, 순수 로직)

LLM 설정 (gpt-4o-mini):
    TEMPERATURE_AGENT      = 0.7
    TEMPERATURE_VALIDATOR  = 0.3
    MAX_DEBATE_ROUNDS      = 6        # 라운드 0 + 토론 라운드 1..5
    MODERATOR_FORCE_ROUND  = 5        # 5라운드에서 강제 종료

════════════════════════════

# 라운드 0: 초기 진술
각 에이전트(5명)에 대해 (병렬):
    수신:
        특허 (제목, 초록)
    출력 (JSON):
        prediction         ∈ {0: 비유망, 1: 유망}
        confidence_for     ∈ [0.50, 1.00]   # 자신의 예측에 대한 확신도
        confidence_against = 1.0 − confidence_for
        reason             ≤ 2 문장

검증자: 5개 에이전트의 reason → fact_based ∈ {0, 1} 5개 판정
라운드 0 결과 + 검증 결과 저장

# 조정자: 라운드 0 → 어떤 분포여도 무조건 토론 진행
#          (evaluate_round(0, ·) → False)

════════════════════════════

# 라운드 1 ~ 5: 토론 라운드
round_num ∈ {1, 2, 3, 4, 5}:

    각 에이전트(5명)에 대해 (병렬):
        수신:
            특허 (제목, 초록)
            직전 라운드 5개 에이전트의 (prediction, confidence_for, reason)

        출력 (JSON):
            actions[5]:                            # 5개 타겟 모두에 행동 (자신 포함)
                각 타겟마다:
                    target_agent  ∈ Π
                    action        ∈ {Support, Attack}
                    action_reason ≤ 1 문장
            prediction         ∈ {0, 1}
            confidence_for     ∈ [0.50, 1.00]
            confidence_against = 1.0 − confidence_for
            reason             ≤ 2 문장 (자신의 actions를 종합)

    검증자: 5개 에이전트의 reason → fact_based ∈ {0, 1} 5개 판정
    라운드 결과 + 검증 결과 저장

    # 조정자 판단 (n = 5)
    pro_count    ← prediction = 1 인 에이전트 수
    anti_count   ← 5 − pro_count
    is_consensus ← (pro_count ≥ 4 ∨ anti_count ≥ 4)   # 5:0 또는 4:1
    is_split     ← (pro_count = 3 ∧ anti_count = 2) 또는 그 반대  # 3:2
    # n=5라 짝수 동률(예: 2:2)은 불가능

    종료 판정:
        if round_num = 1 ∧ is_consensus:
            terminate, reason = "unanimous"               # 즉시 합의

        elif round_num ∈ {2, 3, 4} ∧ is_consensus:
            terminate, reason = "consensus_mid"           # 토론 중 합의

        elif round_num = 5 ∧ is_consensus:
            terminate, reason = "consensus_late"          # 마지막에 합의

        elif round_num = 5 ∧ is_split:                    # 강제 종료 (3:2)
            terminate, reason = "forced_majority"         # 다수 측 승

        else:                                              # 라운드 1−4의 3:2
            continue

    # 종료 시 최종 예측 결정
    if terminate:
        final_prediction ← majority side (pro 또는 anti)
        # is_consensus    면 다수가 명백 (≥ 4명)
        # forced_majority 면 3명 측이 승 (단순 다수)
        state.termination_round  ← round_num
        state.termination_reason ← reason
        state.final_prediction   ← final_prediction
        state.final_confidence   ← winning-side confidence aggregate
        break

════════════════════════════

# 종료 보장 (Round 5 forced)
# Round 5에서는 is_consensus 또는 is_split 둘 중 하나가 무조건 참
# 따라서 Round 5에 도달하면 반드시 종료됨

assert state.termination_round  ∈ {1, 2, 3, 4, 5}
assert state.termination_reason ∈ {"unanimous", "consensus_mid",
                                    "consensus_late", "forced_majority"}

DebateState 반환
```

---

## 5. Termination cases (4 reasons, 2 emitted indicators)

The four `termination_reason` values map to two complementary binary
indicator columns in the output dataset:

| Round | Verdict | `termination_reason` | `term_unanimous` | `term_extended_debate` |
|---|---|---|---|---|
| 1 | 5:0 or 4:1 | `unanimous`        | 1 | 0 |
| 2–4 | 5:0 or 4:1 | `consensus_mid`    | 0 | 1 |
| 5 | 5:0 or 4:1 | `consensus_late`   | 0 | 1 |
| 5 | 3:2 (forced) | `forced_majority`  | 0 | 1 |

By construction `term_unanimous + term_extended_debate ≡ 1`. The
string `termination_reason` is preserved on `DebateState` for
forensic post-hoc analysis (e.g., comparing case-3 vs case-4 patents).

3:2 in rounds 1–4 → continue. Tie (2.5 : 2.5) is impossible at n = 5
(odd panel size).

## 6. Why round 0 always continues

Round 0 is the **initial silent disclosure**: each persona reads the
patent and predicts independently, without seeing other personas'
reasoning. Even if all 5 agree at this stage, the protocol forces at
least one debate round (round 1) so that:

1. Every persona's reasoning is exposed to scrutiny.
2. Validator gets to judge fact-based reasoning across rounds.
3. Argument-graph variables (`cross_*`, `same_*`, `acceptability_gap`)
   have at least one round of actions to aggregate over.

The minimum total round count is therefore 2 (round 0 + round 1) and
the maximum is 6 (round 0 + rounds 1..5).

## 7. Validator and Moderator: separation of concerns

The Validator and Moderator are **deliberately separate components**
with distinct roles:

| | Validator | Moderator |
|---|---|---|
| Type | LLM agent | Rule-based logic |
| Input | 5 personas' reason texts | 5 personas' (prediction, confidence) tuples |
| Output | `fact_based[5]` ∈ {0, 1} | termination decision + reason string |
| Effect on debate flow | None — does not stop or continue debate | Decides termination |
| Effect on output vars | Source of `fact_ratio_pro`, `fact_ratio_anti` | Source of `total_rounds`, `term_unanimous`, `term_extended_debate` |

The Validator's per-round 5 fact_based judgments are stored in
`state.validations` and merged back into each round's persona dicts
as a `fact_based` key, but they do **not** feed back into the next
round's prompts (the next-round prompt sees only previous-round
predictions, confidences, reasons — not validator labels).

---

## 8. References

| Component | Location |
|---|---|
| Variable extraction from `DebateState` | `VARIABLES.md` |
| Persona prompts | `debate/config.py` (`AGENT_PERSONAS`) |
| Validator prompt | `debate/config.py` (`VALIDATOR_SYSTEM_PROMPT`) |
| Round-0 / debate-round JSON formats | `debate/config.py` (`ROUND0_FORMAT`, `DEBATE_FORMAT`, `VALIDATOR_FORMAT`) |
| Orchestrator | `debate/debate.py` (`DebateOrchestrator`) |
| Moderator logic | `debate/agents.py` (`Moderator`) |
| ICE-specific decisions | `CLAUDE.md` |

## Literature

- Kapoor, R., et al. — 5-source-of-uncertainty framework (basis for
  the persona panel).
- Du et al. (2023). *Improving Factuality and Reasoning in Language
  Models through Multiagent Debate.* (multi-agent debate as a
  reasoning protocol)
