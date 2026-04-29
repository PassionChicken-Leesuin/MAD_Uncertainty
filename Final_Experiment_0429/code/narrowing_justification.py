"""25 -> 6 narrowing justification.

Loads three pre-existing artefacts:
  - debate_decompose_univariate_delta.csv : univariate add-Δ AUROC per var
  - debate_decompose_perm_importance.csv  : permutation importance per var
  - forward_selection/ms_selection_path.csv : 5-seed forward-selection path

For each of the 25 debate variables, computes 4 ranking signals and
combines them by Borda count (sum of ranks; lower = stronger). Compares
the empirical top-6 against the pre-selected 6 variables.

Pre-selected 6 (theoretical, 4-group taxonomy):
  cross_domain_attack, conf_gap_change, var_conf_pro,
  H_final, delta_H, semantic_coherence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIV_FILE = OUT_DIR / "debate_decompose_univariate_delta.csv"
PERM_FILE = OUT_DIR / "debate_decompose_perm_importance.csv"
FS_FILE   = OUT_DIR / "forward_selection" / "ms_selection_path.csv"

DEBATE_ALL = [
    "mean_conf_pro", "mean_conf_anti", "var_conf_pro", "var_conf_anti",
    "conf_gap_change", "H_final", "delta_H",
    "cross_domain_support", "cross_domain_attack",
    "same_domain_support", "same_domain_attack", "acceptability_gap",
    "fact_ratio_pro", "fact_ratio_anti",
    "final_prediction", "prediction_volatility",
    "final_pred_technology", "final_pred_application",
    "final_pred_user", "final_pred_ecosystem", "final_pred_businessmodel",
    "total_rounds", "term_unanimous", "term_extended_debate",
    "semantic_coherence",
]

GROUP = {
    "cross_domain_attack":   "A_interaction",
    "cross_domain_support":  "A_interaction",
    "same_domain_attack":    "A_interaction",
    "same_domain_support":   "A_interaction",
    "conf_gap_change":       "B_dynamic",
    "delta_H":                "B_dynamic",
    "prediction_volatility": "B_dynamic",
    "mean_conf_pro":         "C_individual",
    "mean_conf_anti":        "C_individual",
    "var_conf_pro":          "C_individual",
    "var_conf_anti":         "C_individual",
    "H_final":               "C_individual",
    "semantic_coherence":    "D_consensus",
    "fact_ratio_pro":        "D_consensus",
    "fact_ratio_anti":       "D_consensus",
    "acceptability_gap":     "D_consensus",
    "final_prediction":      "E_prediction",
    "final_pred_technology": "E_prediction",
    "final_pred_application":"E_prediction",
    "final_pred_user":       "E_prediction",
    "final_pred_ecosystem":  "E_prediction",
    "final_pred_businessmodel": "E_prediction",
    "total_rounds":          "F_meta",
    "term_unanimous":        "F_meta",
    "term_extended_debate":  "F_meta",
}

PRE_SELECTED_6 = [
    "cross_domain_attack", "conf_gap_change", "var_conf_pro",
    "H_final", "delta_H", "semantic_coherence",
]


def main():
    print("=" * 82)
    print("25 -> 6 narrowing justification")
    print("=" * 82)
    print(f"\nPre-selected 6 (theoretical):")
    for v in PRE_SELECTED_6:
        print(f"  {v:<25s}  group={GROUP[v]}")

    # ─── Load data ────────────────────────────────────────────────────────
    univ = pd.read_csv(UNIV_FILE, encoding="utf-8-sig")
    perm = pd.read_csv(PERM_FILE, encoding="utf-8-sig")
    fs   = pd.read_csv(FS_FILE,   encoding="utf-8-sig")

    print(f"\nLoaded:")
    print(f"  univariate-delta: {len(univ):,} rows  (groups={univ['group'].unique().tolist()})")
    print(f"  perm-importance:  {len(perm):,} rows  (seeds={sorted(perm['seed'].unique().tolist())})")
    print(f"  ms_selection_path: {len(fs):,} rows  (steps={sorted(fs['step'].unique().tolist())[:5]}...)")

    # ─── Signal 1: univariate add-Δ AUROC (mean across seeds, GLOBAL group) ──
    sig1 = (univ[univ["group"] == "GLOBAL"]
            .groupby("var")["delta"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "univariate_delta_mean",
                             "std":  "univariate_delta_std",
                             "count":"univariate_n_seeds"}))

    # ─── Signal 2: permutation importance (mean across seeds, GLOBAL) ───────
    sig2 = (perm[perm["group"] == "GLOBAL"]
            .groupby("var")["importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "perm_importance_mean",
                             "std":  "perm_importance_std",
                             "count":"perm_n_seeds"}))

    # ─── Signal 3: FS step-1 AUROC (each var added alone to BIBLIO) ─────────
    fs1 = fs[fs["step"] == 1].copy()
    sig3 = fs1[["candidate", "driver_AUROC_mean", "driver_AUROC_std",
                "uplift_mean"]].rename(columns={
        "candidate": "var",
        "driver_AUROC_mean": "fs_step1_AUROC_mean",
        "driver_AUROC_std":  "fs_step1_AUROC_std",
        "uplift_mean":       "fs_step1_uplift",
    })

    # ─── Signal 4: FS first-winner step (smaller = picked earlier = stronger) ─
    winners = fs[fs["is_winner"] == True].sort_values("step").drop_duplicates("candidate")
    sig4 = winners[["candidate", "step"]].rename(columns={
        "candidate": "var", "step": "fs_first_winner_step"
    })

    # ─── Merge all 4 signals ─────────────────────────────────────────────
    R = pd.DataFrame({"var": DEBATE_ALL})
    R["group"] = R["var"].map(GROUP)
    R = R.merge(sig1, on="var", how="left")
    R = R.merge(sig2, on="var", how="left")
    R = R.merge(sig3, on="var", how="left")
    R = R.merge(sig4, on="var", how="left")
    R["is_pre_selected_6"] = R["var"].isin(PRE_SELECTED_6)

    # ─── Compute Borda ranks ──────────────────────────────────────────────
    # Rank 1 = best per signal
    R["rk_univariate"]   = R["univariate_delta_mean"].rank(ascending=False, method="min")
    R["rk_perm"]         = R["perm_importance_mean"].rank(ascending=False, method="min")
    R["rk_fs_step1"]     = R["fs_step1_AUROC_mean"].rank(ascending=False, method="min")
    # Earlier winners are stronger; NaN means never selected as winner
    R["rk_fs_winner"]    = R["fs_first_winner_step"].rank(ascending=True, method="min", na_option="bottom")

    R["borda_sum"] = R[["rk_univariate", "rk_perm", "rk_fs_step1",
                         "rk_fs_winner"]].sum(axis=1)
    R["borda_rank"] = R["borda_sum"].rank(ascending=True, method="min").astype(int)

    R = R.sort_values("borda_rank").reset_index(drop=True)

    # ─── Print key tables ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("Per-variable rankings on 4 signals + Borda combined rank (lower = stronger)")
    print("=" * 110)
    cols = ["borda_rank", "var", "group", "is_pre_selected_6",
            "rk_univariate", "rk_perm", "rk_fs_step1", "rk_fs_winner",
            "univariate_delta_mean", "perm_importance_mean",
            "fs_step1_AUROC_mean", "fs_first_winner_step"]
    print(R[cols].to_string(index=False))

    # ─── Top-6 vs pre-selected 6 ──────────────────────────────────────────
    top6_borda = R.head(6)["var"].tolist()
    overlap = set(top6_borda) & set(PRE_SELECTED_6)
    only_borda = set(top6_borda) - set(PRE_SELECTED_6)
    only_pre = set(PRE_SELECTED_6) - set(top6_borda)

    print("\n" + "=" * 82)
    print("Top-6 by Borda combined rank")
    print("=" * 82)
    print(f"  Borda top-6: {top6_borda}")
    print(f"  Pre-selected 6: {PRE_SELECTED_6}")
    print(f"\n  Overlap: {len(overlap)}/6  ({sorted(overlap)})")
    print(f"  In Borda top-6 only: {sorted(only_borda)}")
    print(f"  In pre-selected only: {sorted(only_pre)}")
    if only_pre:
        print(f"\n  Pre-selected vars NOT in Borda top-6 (and their Borda rank):")
        for v in sorted(only_pre):
            r = R[R["var"] == v].iloc[0]
            print(f"    {v:<22s}  Borda rank={r['borda_rank']}  "
                  f"univ_Δ={r['univariate_delta_mean']:.4f}  "
                  f"perm={r['perm_importance_mean']:.4f}  "
                  f"FS_step1={r['fs_step1_AUROC_mean']:.4f}  "
                  f"first_winner_step={r['fs_first_winner_step']}")

    # ─── Group coverage ──────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("Group coverage of pre-selected 6 vs Borda top-6")
    print("=" * 82)
    g_pre = R[R["is_pre_selected_6"]].groupby("group").size().to_dict()
    g_borda = R.head(6).groupby("group").size().to_dict()
    all_groups = sorted(set(GROUP.values()))
    cov_rows = []
    for g in all_groups:
        cov_rows.append({"group": g,
                         "pre_selected_6": g_pre.get(g, 0),
                         "borda_top_6":    g_borda.get(g, 0),
                         "n_vars_in_group": sum(1 for v in DEBATE_ALL if GROUP[v] == g)})
    cov = pd.DataFrame(cov_rows)
    print(cov.to_string(index=False))

    # ─── Top-10 by Borda for context ─────────────────────────────────────
    print("\n" + "=" * 82)
    print("Borda top-10 (= empirical strongest 10 of 25 by combined evidence)")
    print("=" * 82)
    print(R.head(10)[["borda_rank", "var", "group", "is_pre_selected_6",
                       "rk_univariate", "rk_perm", "rk_fs_step1", "rk_fs_winner",
                       "borda_sum"]].to_string(index=False))

    # Save
    R.to_csv(OUT_DIR / "narrow6_evidence.csv",
             index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_DIR / 'narrow6_evidence.csv'}")

    # ─── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    overlap_rate = len(overlap) / 6
    if overlap_rate >= 5/6:
        verdict = "STRONG"
        msg = "Pre-selected 6 essentially matches empirical Borda top-6 -> narrowing well-supported."
    elif overlap_rate >= 4/6:
        verdict = "MODERATE"
        msg = "Pre-selected 6 mostly aligns with empirical evidence; 1-2 vars differ -> theoretically defensible."
    else:
        verdict = "WEAK"
        msg = "Pre-selected 6 diverges from empirical evidence -> narrowing should be revisited."
    print(f"  Overlap rate: {len(overlap)}/6 = {overlap_rate*100:.1f}% ({verdict})")
    print(f"  {msg}")


if __name__ == "__main__":
    main()
