"""Exhaustive 7-var search with FULL metric grid: Precision/Recall/DOR @ k%.

Same 7-var search as `exhaustive_7vars.py` but ranks subsets by
*each* of 12 metric slots (3 metrics x 4 k-pcts) instead of just AUROC.

Metric slots (12):
    Precision @ k%  for k in {5, 10, 15, 20}
    Recall    @ k%  for k in {5, 10, 15, 20}
    DOR       @ k%  for k in {5, 10, 15, 20}

For each subset (128 = BIBLIO + 127 non-empty), train 6 models x 5 seeds
and store per-(subset, model, seed, k_pct) the metric values. Then for
each metric slot, aggregate to grand mean across (6 models x 5 seeds)
and rank the 128 subsets.

Outputs (analysis/outputs/forward_selection/):
- exh7g_metric_grid.csv      raw long-form: subset, model, seed, k_pct, metrics
- exh7g_subset_per_slot.csv  per (subset, slot) grand mean (Precision/Recall/DOR @ k)
- exh7g_winners.csv          per slot: top-5 subsets ranked
- exh7g_summary.md           consensus analysis: do slots agree on a winner?
"""

from __future__ import annotations

import math
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEEDS = [42, 0, 1, 2, 3]
K_PCTS = [5, 10, 15, 20]
METRICS = ["Precision", "Recall", "DOR"]

CANDIDATES = [
    "cross_domain_attack", "conf_gap_change", "var_conf_pro",
    "H_final", "delta_H", "semantic_coherence", "prediction_volatility",
]
MODELS_ORDER = ["RF", "LogReg", "GBT", "SVM", "FFN", "XGB"]

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "debate" / "runs" / "v2a_y_anchored" / "results" / "variables_full.csv"
OUT_DIR = HERE / "outputs" / "forward_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATENT_NUMERIC = ["CTO", "STO", "PK", "SK", "TCT",
                  "TS", "NC",
                  "COL", "INV", "TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]


def build_X(df, debate_cols, mf_top):
    blocks = [df[PATENT_NUMERIC].astype(float)]
    mf_blk = pd.DataFrame(index=df.index)
    for mf in mf_top:
        mf_blk[f"MF_{mf}"] = (df["MF"] == mf).astype(int)
    mf_blk["MF_other"] = (~df["MF"].isin(mf_top)).astype(int)
    blocks.append(mf_blk)
    if debate_cols:
        blocks.append(df[list(debate_cols)].astype(float))
    X = pd.concat(blocks, axis=1)
    X.columns = X.columns.astype(str)
    return X


def fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.decision_function(Xte)


def make_full_models(seed: int, n_features: int):
    h = max(1, math.ceil(math.sqrt(max(1, n_features) * 2)))
    return {
        "RF":  RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
        "LogReg": LogisticRegression(
            max_iter=2000, random_state=seed, n_jobs=-1),
        "GBT": HistGradientBoostingClassifier(
            max_iter=300, random_state=seed),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "FFN": MLPClassifier(
            hidden_layer_sizes=(h,), activation="logistic",
            solver="lbfgs", alpha=0.001, max_iter=1000,
            random_state=seed),
        "XGB": XGBClassifier(
            n_estimators=300, tree_method="hist", random_state=seed,
            eval_metric="logloss", n_jobs=-1, verbosity=0),
    }


def haldane_dor(tp, tn, fp, fn):
    return ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))


def metrics_at_topk_pct(y_true, p_score, k_pct):
    n = len(y_true)
    k_abs = max(1, int(round(n * k_pct / 100.0)))
    order = np.argsort(-p_score, kind="mergesort")
    top_idx = order[:k_abs]
    y_pred = np.zeros(n, dtype=int)
    y_pred[top_idx] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    dor = haldane_dor(tp, tn, fp, fn)
    return prec, rec, dor


def main():
    t0 = time.time()
    print("=" * 82)
    print(f"Exhaustive 7-var search with metric grid")
    print("=" * 82)
    print(f"Candidates: {CANDIDATES}")
    print(f"k% values: {K_PCTS}")
    print(f"Metrics: {METRICS}")
    print(f"Models: {MODELS_ORDER}, Seeds: {SEEDS}")

    df_full = pd.read_csv(DATA, dtype={"patent_id": str, "fields": str},
                          encoding="utf-8-sig")
    df = df_full[df_full["mean_conf_pro"].notna()].reset_index(drop=True).copy()
    print(f"  cohort: {len(df):,}  Y rate: {df['Y'].mean()*100:.2f}%")
    y = df["Y"].astype(int).values

    # Generate all subsets including empty (BIBLIO_ONLY)
    subsets = [tuple()]
    for k in range(1, len(CANDIDATES) + 1):
        for c in combinations(CANDIDATES, k):
            subsets.append(tuple(c))
    print(f"  total subsets: {len(subsets)}")

    # Long-form storage: one row per (subset, seed, model, k_pct)
    # Plus AUROC/AUPRC per (subset, seed, model)
    rows_k = []
    rows_auc = []

    for i, sub in enumerate(subsets):
        t_s = time.time()
        debate_cols = list(sub) if sub else None
        sub_label = "+".join(sub) if sub else "BIBLIO_ONLY"

        for seed in SEEDS:
            idx = np.arange(len(df))
            idx_tr, idx_te = train_test_split(
                idx, test_size=0.2, stratify=y, random_state=seed)
            df_tr = df.iloc[idx_tr]
            df_te = df.iloc[idx_te]
            ytr, yte = y[idx_tr], y[idx_te]
            mf_top = df_tr["MF"].value_counts().head(10).index.tolist()
            Xtr = build_X(df_tr, debate_cols, mf_top)
            Xte = build_X(df_te, debate_cols, mf_top)
            imp = SimpleImputer(strategy="median")
            sc = MinMaxScaler()
            Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
            Xte_s = sc.transform(imp.transform(Xte))
            for mname, m in make_full_models(seed, Xtr_s.shape[1]).items():
                p = fit_predict(m, Xtr_s, ytr, Xte_s)
                rows_auc.append({
                    "subset_id": i, "subset_label": sub_label,
                    "subset_size": len(sub),
                    "seed": seed, "model": mname,
                    "AUROC": roc_auc_score(yte, p),
                    "AUPRC": average_precision_score(yte, p),
                })
                for k in K_PCTS:
                    prec, rec, dor = metrics_at_topk_pct(yte, p, k)
                    rows_k.append({
                        "subset_id": i, "subset_label": sub_label,
                        "subset_size": len(sub),
                        "seed": seed, "model": mname, "k_pct": k,
                        "Precision": prec, "Recall": rec, "DOR": dor,
                    })
        if (i + 1) % 16 == 0 or i == 0 or i == len(subsets) - 1:
            print(f"  [{i+1}/{len(subsets)}] {sub_label[:60]:<60s}  "
                  f"({time.time()-t_s:.1f}s)")

    print(f"\nTraining loop done in {(time.time()-t0):.0f}s. Aggregating ...")

    K = pd.DataFrame(rows_k)
    A = pd.DataFrame(rows_auc)
    K.to_csv(OUT_DIR / "exh7g_metric_grid.csv",
             index=False, encoding="utf-8-sig")

    # ─── Aggregate per (subset, k_pct): grand mean across 6 models x 5 seeds ──
    AGG = (K.groupby(["subset_id", "subset_label", "subset_size", "k_pct"])
            .agg(
                Precision_mean=("Precision", "mean"),
                Precision_std =("Precision", "std"),
                Recall_mean   =("Recall", "mean"),
                Recall_std    =("Recall", "std"),
                DOR_mean      =("DOR", "mean"),
                DOR_std       =("DOR", "std"),
                n_obs         =("Precision", "count"),
            )
            .round(5).reset_index())
    AGG.to_csv(OUT_DIR / "exh7g_subset_per_slot.csv",
               index=False, encoding="utf-8-sig")

    # AUROC aggregate (for tie-breaking / context)
    AGG_A = (A.groupby(["subset_id", "subset_label", "subset_size"])
              .agg(AUROC_mean=("AUROC", "mean"),
                   AUROC_std =("AUROC", "std"))
              .round(5).reset_index())

    # BIBLIO baseline rows
    base_K = AGG[AGG["subset_label"] == "BIBLIO_ONLY"].set_index("k_pct")
    base_AUROC = AGG_A[AGG_A["subset_label"] == "BIBLIO_ONLY"]["AUROC_mean"].iloc[0]

    # ─── Per-slot rankings: for each (k_pct, metric), rank subsets ──────────
    winner_rows = []
    print("\n" + "=" * 100)
    print("Per-slot winners (top 5 per metric × k%)")
    print("=" * 100)
    for k in K_PCTS:
        for met in METRICS:
            base_val = base_K.loc[k, f"{met}_mean"]
            sub = AGG[AGG["k_pct"] == k].copy()
            sub[f"delta_{met}"] = (sub[f"{met}_mean"] - base_val).round(5)
            sub = sub.sort_values(f"{met}_mean", ascending=False).reset_index(drop=True)
            sub.insert(0, "slot_rank", sub.index + 1)
            print(f"\n--- {met} @ top-{k}%   (BIBLIO baseline={base_val:.5f}) ---")
            cols = ["slot_rank", "subset_size", "subset_label",
                    f"{met}_mean", f"{met}_std", f"delta_{met}"]
            print(sub.head(5)[cols].to_string(index=False))
            for _, r in sub.head(5).iterrows():
                winner_rows.append({
                    "k_pct": k, "metric": met,
                    "rank_in_slot": int(r["slot_rank"]),
                    "subset_size": int(r["subset_size"]),
                    "subset_label": r["subset_label"],
                    "value_mean": float(r[f"{met}_mean"]),
                    "value_std":  float(r[f"{met}_std"]),
                    "delta_vs_BIBLIO": float(r[f"delta_{met}"]),
                    "BIBLIO_baseline": float(base_val),
                })
    W = pd.DataFrame(winner_rows)
    W.to_csv(OUT_DIR / "exh7g_winners.csv", index=False, encoding="utf-8-sig")

    # ─── Consensus analysis ────────────────────────────────────────────────
    # For each subset: how many slots is it in top 1? top 3? top 5?
    in_topk = {1: {}, 3: {}, 5: {}}
    for k_pct in K_PCTS:
        for met in METRICS:
            sub = AGG[AGG["k_pct"] == k_pct].copy()
            sub = sub.sort_values(f"{met}_mean", ascending=False).reset_index(drop=True)
            for cap in (1, 3, 5):
                for label in sub["subset_label"].head(cap):
                    in_topk[cap][label] = in_topk[cap].get(label, 0) + 1

    consensus = pd.DataFrame({
        "subset_label": list(set(in_topk[5].keys())),
    })
    for cap in (1, 3, 5):
        consensus[f"in_top{cap}_count"] = consensus["subset_label"].map(
            lambda s: in_topk[cap].get(s, 0))
    consensus["subset_size"] = consensus["subset_label"].apply(
        lambda s: 0 if s == "BIBLIO_ONLY" else len(s.split("+")))
    consensus = consensus.sort_values(
        ["in_top1_count", "in_top3_count", "in_top5_count"],
        ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"Consensus across all 12 metric slots ({len(K_PCTS)} k% × {len(METRICS)} metrics)")
    print("=" * 100)
    print(f"  Total slots: {len(K_PCTS) * len(METRICS)}")
    print(f"\nSubsets ranked by appearances in top-1 across slots (then top-3, top-5):")
    print(consensus.head(15).to_string(index=False))
    consensus.to_csv(OUT_DIR / "exh7g_consensus.csv",
                     index=False, encoding="utf-8-sig")

    # AUROC ranking for context
    AGG_A_sorted = AGG_A.sort_values("AUROC_mean", ascending=False).reset_index(drop=True)
    AGG_A_sorted["delta_AUROC"] = (AGG_A_sorted["AUROC_mean"] - base_AUROC).round(5)
    print(f"\nAUROC ranking (top 10) for context:")
    print(AGG_A_sorted.head(10)[["subset_size", "subset_label", "AUROC_mean", "delta_AUROC"]].to_string(index=False))

    # ─── Markdown summary ──────────────────────────────────────────────────
    lines = []
    lines.append("# 7-var exhaustive search — full metric grid\n")
    lines.append(f"Candidates: {CANDIDATES}\n")
    lines.append(f"\nMetric slots evaluated: 3 metrics × 4 k% = **12 slots**")
    lines.append(f"(Precision / Recall / DOR @ k% for k ∈ {K_PCTS})\n")
    lines.append(f"\nCohort: n={len(df):,}, Y rate {df['Y'].mean()*100:.2f}%. "
                 f"6 models × 5 seeds = 30 fits per subset.\n")

    lines.append("\n## Consensus winners across 12 slots\n")
    lines.append("Subsets ranked by appearances in top-1 across all slots "
                 "(higher = robust winner across metric choices).\n")
    lines.append(consensus.head(15).to_markdown(index=False))

    lines.append("\n\n## Per-slot top-3 winners\n")
    for k_pct in K_PCTS:
        for met in METRICS:
            base_val = base_K.loc[k_pct, f"{met}_mean"]
            sub = AGG[AGG["k_pct"] == k_pct].sort_values(
                f"{met}_mean", ascending=False).head(3)
            lines.append(f"\n### {met} @ top-{k_pct}%  (BIBLIO = {base_val:.5f})\n")
            display = sub[["subset_size", "subset_label", f"{met}_mean", f"{met}_std"]].copy()
            display[f"delta_vs_BIBLIO"] = (display[f"{met}_mean"] - base_val).round(5)
            lines.append(display.to_markdown(index=False))

    lines.append(f"\n\n## AUROC ranking (top 10) for reference\n")
    lines.append(AGG_A_sorted.head(10)[["subset_size", "subset_label",
                                          "AUROC_mean", "delta_AUROC"]].to_markdown(index=False))

    (OUT_DIR / "exh7g_summary.md").write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nTotal wall: {elapsed:.0f}s = {elapsed/60:.1f} min")
    for fn in ("exh7g_metric_grid.csv", "exh7g_subset_per_slot.csv",
               "exh7g_winners.csv", "exh7g_consensus.csv", "exh7g_summary.md"):
        print(f"  {OUT_DIR / fn}")


if __name__ == "__main__":
    main()
