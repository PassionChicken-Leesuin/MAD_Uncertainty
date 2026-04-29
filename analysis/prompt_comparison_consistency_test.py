"""Definitive head-to-head: v0 vs v2a in matching/exceeding BIBLIO.

For every (field × feature_set × model), compute across 20 seeds:
  - win_rate(prompt) = #seeds where (BIBLIO+DEBATE) > BIBLIO  / 20
  - tie_or_win(prompt) = #seeds where (BIBLIO+DEBATE) >= BIBLIO / 20
  - mean_delta(prompt) = mean of (BIBLIO+DEBATE) - BIBLIO across seeds

Then count cells where v2a wins (higher win rate / mean delta) vs v0 wins.
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np

ICE_ROOT = Path(__file__).resolve().parents[2]
RAW = ICE_ROOT / "debate" / "runs" / "perfield_multiseed_raw.csv"

raw = pd.read_csv(RAW)

FIELDS = ["GLOBAL","01","03","07","08","09","12","13","14"]
fname = {
    "GLOBAL":"GLOBAL","01":"Q01 ICE","03":"Q03 HCCI","07":"Q07 Hybrids",
    "08":"Q08 EGR","09":"Q09 Turbo","12":"Q12 VVA","13":"Q13 AltFuels","14":"Q14 DI"
}
MODELS = ["RF", "GBT", "LogReg"]
DEBATE_FSETS = ["BIBLIO+DEBATE5", "BIBLIO+DEBATE25"]

print("=" * 110)
print("HEAD-TO-HEAD: v0 vs v2a — does (BIBLIO+DEBATE) match-or-exceed BIBLIO?")
print(f"For each (field × feature_set × model): 20 seeds, paired comparison")
print("=" * 110)

cells = []
for f in FIELDS:
    for fset in DEBATE_FSETS:
        for mname in MODELS:
            row = {"field": fname[f], "feature_set": fset, "model": mname}
            for ds in ["v0_baseline", "v2a_y_anchored"]:
                bib = raw[(raw.dataset==ds) & (raw.field==f) & (raw.model==mname) & (raw.feature_set=="BIBLIO")].set_index("seed")["auc"]
                aug = raw[(raw.dataset==ds) & (raw.field==f) & (raw.model==mname) & (raw.feature_set==fset)].set_index("seed")["auc"]
                common = bib.index.intersection(aug.index)
                if len(common) < 5:
                    row[f"{ds}_win"] = float("nan"); row[f"{ds}_tieW"] = float("nan"); row[f"{ds}_d"] = float("nan")
                    continue
                d = aug[common] - bib[common]
                row[f"{ds}_win"]  = int((d > 0).sum())     # strict win
                row[f"{ds}_tieW"] = int((d >= 0).sum())     # tie-or-win (≥0 within 1e-6)
                row[f"{ds}_d"]   = float(d.mean())
                row["n_seeds"] = len(common)
            cells.append(row)

df = pd.DataFrame(cells)
df["v2a_better_d"]   = df["v2a_y_anchored_d"] > df["v0_baseline_d"]
df["v2a_better_win"] = df["v2a_y_anchored_win"] > df["v0_baseline_win"]

# Print per-cell table
print("\n{:<14}{:<18}{:<7} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10}    Winner".format(
    "field","feature_set","model","v0 win/20","v0 mean Δ","v0 ≥0/20","v2a win/20","v2a mean Δ","v2a ≥0/20"
))
print("-" * 130)
for _, r in df.iterrows():
    if pd.isna(r["v0_baseline_d"]):
        continue
    winner = "v2a" if r["v2a_y_anchored_win"] > r["v0_baseline_win"] else (
             "v0"  if r["v2a_y_anchored_win"] < r["v0_baseline_win"] else "tie")
    if r["v2a_y_anchored_win"] == r["v0_baseline_win"]:
        winner = "v2a" if r["v2a_y_anchored_d"] > r["v0_baseline_d"] else (
                 "v0"  if r["v2a_y_anchored_d"] < r["v0_baseline_d"] else "tie")
    print(f"{r['field']:<14}{r['feature_set']:<18}{r['model']:<7} "
          f"{int(r['v0_baseline_win']):>10}/20  {r['v0_baseline_d']:>+12.4f} {int(r['v0_baseline_tieW']):>8}/20  "
          f"{int(r['v2a_y_anchored_win']):>10}/20  {r['v2a_y_anchored_d']:>+12.4f} {int(r['v2a_y_anchored_tieW']):>8}/20    {winner}")

# Global tally
print("\n" + "=" * 110)
print("AGGREGATE TALLY")
print("=" * 110)
valid = df.dropna(subset=["v0_baseline_d", "v2a_y_anchored_d"])
n_total = len(valid)
v2a_better_d   = int(valid["v2a_better_d"].sum())
v2a_better_win = int(valid["v2a_better_win"].sum())
v0_better_d    = int((~valid["v2a_better_d"]).sum())
v0_better_win  = int((~valid["v2a_better_win"]).sum())
ties_d   = int((valid["v2a_y_anchored_d"]   == valid["v0_baseline_d"]).sum())
ties_win = int((valid["v2a_y_anchored_win"] == valid["v0_baseline_win"]).sum())
print(f"Cells (field × feature_set × model): {n_total}")
print(f"\n  Mean Δ AUC comparison:")
print(f"    v2a > v0:  {v2a_better_d}/{n_total}  ({v2a_better_d/n_total*100:.1f}%)")
print(f"    v0 > v2a:  {v0_better_d - ties_d}/{n_total}")
print(f"    tie:       {ties_d}/{n_total}")
print(f"\n  Win rate (#seeds where DEBATE>BIBLIO) comparison:")
print(f"    v2a wins more often: {v2a_better_win}/{n_total}  ({v2a_better_win/n_total*100:.1f}%)")
print(f"    v0 wins more often:  {v0_better_win - ties_win}/{n_total}")
print(f"    tie:                  {ties_win}/{n_total}")

# How many cells have augmented ≥ BIBLIO at least 50% of the time (≥ 10/20)?
v0_50plus  = int((valid["v0_baseline_win"]   >= 10).sum())
v2a_50plus = int((valid["v2a_y_anchored_win"] >= 10).sum())
v0_60plus  = int((valid["v0_baseline_win"]   >= 12).sum())
v2a_60plus = int((valid["v2a_y_anchored_win"] >= 12).sum())
v0_70plus  = int((valid["v0_baseline_win"]   >= 14).sum())
v2a_70plus = int((valid["v2a_y_anchored_win"] >= 14).sum())
print(f"\n  Cells where DEBATE > BIBLIO in ≥10/20 seeds (50%+):")
print(f"    v0:  {v0_50plus}/{n_total}  ({v0_50plus/n_total*100:.1f}%)")
print(f"    v2a: {v2a_50plus}/{n_total}  ({v2a_50plus/n_total*100:.1f}%)")
print(f"\n  Cells where DEBATE > BIBLIO in ≥12/20 seeds (60%+):")
print(f"    v0:  {v0_60plus}/{n_total}  ({v0_60plus/n_total*100:.1f}%)")
print(f"    v2a: {v2a_60plus}/{n_total}  ({v2a_60plus/n_total*100:.1f}%)")
print(f"\n  Cells where DEBATE > BIBLIO in ≥14/20 seeds (70%+ - 'robust'):")
print(f"    v0:  {v0_70plus}/{n_total}  ({v0_70plus/n_total*100:.1f}%)")
print(f"    v2a: {v2a_70plus}/{n_total}  ({v2a_70plus/n_total*100:.1f}%)")

# Mean of mean-deltas
print(f"\n  Average of mean-Δ across {n_total} cells:")
print(f"    v0:  {valid['v0_baseline_d'].mean():+.4f}")
print(f"    v2a: {valid['v2a_y_anchored_d'].mean():+.4f}")
print(f"    paired diff: {(valid['v2a_y_anchored_d'] - valid['v0_baseline_d']).mean():+.4f}")

df.to_csv(ICE_ROOT / "debate" / "runs" / "prompt_consistency_test.csv", index=False)
print(f"\nSaved: {ICE_ROOT / 'debate' / 'runs' / 'prompt_consistency_test.csv'}")
