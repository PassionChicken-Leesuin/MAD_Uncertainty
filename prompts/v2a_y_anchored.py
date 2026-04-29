"""v2a — Round-0 question anchored on the Y definition (ICE, 5-year window).

Identical to AV's v2a_y_anchored.py except the citation window is
"five years" to match ICE's Y definition (top 10% by 5-year forward
citation count, per ICE_Domain/CLAUDE.md §5.1).

Diff vs v0:
- AGENT_PERSONAS unchanged.
- ROUND0_USER_TEMPLATE rephrased to ask about top-10% forward-citation
  probability in 5 years.
- DEBATE_USER_TEMPLATE updated to match.
- FORMAT blocks unchanged.

Banned-token compliance (ICE list: combustion, engine, fuel, hybrid,
turbo, cylinder, piston, injection, ignition, valve, EGR, ICE, vehicle):
- v0_baseline persona texts contain none of these tokens.
- The Y-anchored framing below contains none of these tokens.
"""
from __future__ import annotations

from prompts.v0_baseline import (
    AGENT_PERSONAS,
    ROUND0_FORMAT,
    DEBATE_FORMAT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_FORMAT,
)


ROUND0_USER_TEMPLATE = (
    "Predict whether the following patent will be cited at a top-decile rate "
    "by future patents within five years of its grant date — that is, will "
    "the number of forward citations it accumulates over the five years "
    "after grant place it in the top 10% of its cohort.\n\n"
    "From your assigned uncertainty perspective, consider what features of "
    "this disclosure would (or would not) drive subsequent inventors to "
    "build on it.\n\n"
    "Patent Title: {title}\n"
    "Patent Abstract: {abstract}\n\n"
    "{format_block}"
)

DEBATE_USER_TEMPLATE = (
    "Continue the debate. The question is whether this patent will land in "
    "the top 10% of its cohort by 5-year forward citations.\n\n"
    "Patent Title: {title}\n"
    "Patent Abstract: {abstract}\n\n"
    "{prev_context}"
    "For each agent's argument above, decide whether to Support or Attack "
    "it, then provide your updated prediction.\n\n"
    "{format_block}"
)
