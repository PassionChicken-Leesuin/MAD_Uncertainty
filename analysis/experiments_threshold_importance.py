"""Per-threshold feature importance and SHAP for the +6focal config.

For each T in {5, 10, 15, 20}% × 6 models (RF/GBT/XGB/LogReg/SVM/FFN)
× 20 seeds:
  - Train on +6focal features (BIBLIO 26 + 6 focal debate = 32 cols)
  - Compute sklearn permutation importance on the test set
    (n_repeats=10, scoring=roc_auc)
  - For XGB only: compute SHAP values via TreeExplainer

Output per-seed CSV plus aggregated mean ± std.

Output files (in outputs/experiments_summary/):
  threshold_importance_perm_full.csv
  threshold_importance_perm_summary.csv  (aggregated mean ± std over 20 seeds)
  threshold_importance_shap_full.csv     (XGB only)
  threshold_importance_shap_summary.csv  (aggregated)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

THRESHOLDS_PCT = [5, 10, 15, 20]
SEEDS = [42] + list(range(19))

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PARTIAL = PROJECT / "debate" / "runs" / "v2a_y_anchored" / "results" / "variables_full_partial.csv"
OUT_DIR = HERE / "outputs" / "experiments_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATENT_NUMERIC = ["CTO", "STO", "PK", "SK", "TCT",
                  "TS", "NC",
                  "COL", "INV", "TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]

DEBATE_FOCAL_6 = [
    "var_conf_pro", "H_final", "delta_H",
    "conf_gap_change", "cross_domain_attack", "semantic_coherence",
]


def build_X(df, debate_cols, mf_top):
    blocks = [df[PATENT_NUMERIC].astype(float)]
    mf_blk = pd.DataFrame(index=df.index)
    for mf in mf_top:
        mf_blk[f"MF_{mf}"] = (df["MF"] == mf).astype(int)
    mf_blk["MF_other"] = (~df["MF"].isin(mf_top)).astype(int)
    blocks.append(mf_blk)
    if debate_cols:
        blocks.append(df[debate_cols].astype(float))
    X = pd.concat(blocks, axis=1)
    X.columns = X.columns.astype(str)
    return X


def make_models(n_features, seed):
    h = max(1, math.ceil(math.sqrt(max(1, n_features) * 2)))
    return {
        "RF":  RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "GBT": GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                           random_state=seed),
        "XGB": XGBClassifier(n_estimators=300, tree_method="hist",
                              random_state=seed, eval_metric="logloss",
                              n_jobs=-1, verbosity=0),
        "LogReg": LogisticRegression(solver="lbfgs", max_iter=1000,
                                      random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "FFN": MLPClassifier(hidden_layer_sizes=(h,), activation="logistic",
                              solver="lbfgs", alpha=0.001, max_iter=1000,
                              random_state=seed),
    }


def main():
    print("Loading data ...")
    full = pd.read_csv(PARTIAL, dtype={"patent_id": str, "fields": str},
                       encoding="utf-8-sig")
    df = full[full["mean_conf_pro"].notna()].reset_index(drop=True).copy()
    print(f"  cohort: {len(df):,}")

    # Compute Y_T columns for each threshold
    print("\nThreshold-specific Y labels:")
    for T in THRESHOLDS_PCT:
        thr = np.percentile(df["forward5"].values, 100 - T)
        Y_T = (df["forward5"] >= thr).astype(int)
        rate = Y_T.mean() * 100
        print(f"  T={T:>2}%  cutoff forward5 >= {thr:>5.1f}  Y rate = {rate:.2f}%")
        df[f"Y_T{T}"] = Y_T

    perm_rows = []  # detail
    shap_rows = []  # detail (XGB only)

    for T in THRESHOLDS_PCT:
        print(f"\n{'='*60}\nThreshold T = {T}%\n{'='*60}")
        y = df[f"Y_T{T}"].values
        idx = np.arange(len(df))

        for s_i, seed in enumerate(SEEDS):
            idx_tr, idx_te = train_test_split(
                idx, test_size=0.2, stratify=y, random_state=seed)
            df_tr = df.iloc[idx_tr].reset_index(drop=True)
            df_te = df.iloc[idx_te].reset_index(drop=True)
            ytr, yte = y[idx_tr], y[idx_te]
            mf_top = df_tr["MF"].value_counts().head(10).index.tolist()

            Xtr = build_X(df_tr, DEBATE_FOCAL_6, mf_top)
            Xte = build_X(df_te, DEBATE_FOCAL_6, mf_top)
            feature_names = list(Xtr.columns)
            imp = SimpleImputer(strategy="median")
            sc = MinMaxScaler()
            Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
            Xte_s = sc.transform(imp.transform(Xte))

            models = make_models(Xtr_s.shape[1], seed=seed)
            for mname, m in models.items():
                m.fit(Xtr_s, ytr)

                # Permutation importance on test set
                r = permutation_importance(
                    m, Xte_s, yte, n_repeats=10,
                    random_state=seed, n_jobs=-1, scoring="roc_auc")
                for fn, imp_mean, imp_std in zip(feature_names,
                                                   r.importances_mean,
                                                   r.importances_std):
                    perm_rows.append({
                        "T_pct": T, "seed": seed, "model": mname,
                        "feature": fn,
                        "importance_mean": float(imp_mean),
                        "importance_std": float(imp_std),
                    })

                # SHAP only for XGB
                if mname == "XGB":
                    explainer = shap.TreeExplainer(m)
                    sv = explainer.shap_values(Xte_s)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]
                    mean_abs = np.abs(sv).mean(axis=0)
                    for fn, ma in zip(feature_names, mean_abs):
                        shap_rows.append({
                            "T_pct": T, "seed": seed,
                            "feature": fn,
                            "mean_abs_shap": float(ma),
                        })

            if (s_i + 1) % 5 == 0:
                print(f"  T={T}%, seed {seed}: done ({s_i+1}/{len(SEEDS)})")

    # Save raw
    P = pd.DataFrame(perm_rows)
    P.to_csv(OUT_DIR / "threshold_importance_perm_full.csv",
             index=False, encoding="utf-8-sig")
    S = pd.DataFrame(shap_rows)
    S.to_csv(OUT_DIR / "threshold_importance_shap_full.csv",
             index=False, encoding="utf-8-sig")

    # Aggregate perm
    perm_agg = P.groupby(["T_pct", "model", "feature"]).agg(
        n_seeds=("seed", "count"),
        imp_mean=("importance_mean", "mean"),
        imp_std=("importance_mean", "std"),
        imp_within_std=("importance_std", "mean"),
    ).round(5).reset_index()
    perm_agg.to_csv(OUT_DIR / "threshold_importance_perm_summary.csv",
                    index=False, encoding="utf-8-sig")

    # Aggregate SHAP
    shap_agg = S.groupby(["T_pct", "feature"]).agg(
        n_seeds=("seed", "count"),
        mean_abs_shap_mean=("mean_abs_shap", "mean"),
        mean_abs_shap_std=("mean_abs_shap", "std"),
    ).round(5).reset_index()
    shap_agg.to_csv(OUT_DIR / "threshold_importance_shap_summary.csv",
                    index=False, encoding="utf-8-sig")

    # ─── Print top features per (T, model) — perm importance ───────
    print("\n" + "=" * 100)
    print("PERMUTATION IMPORTANCE — top 10 features per (T, model)")
    print("=" * 100)
    for T in THRESHOLDS_PCT:
        print(f"\n--- T = {T}% ---")
        for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
            sub = perm_agg[(perm_agg["T_pct"] == T) & (perm_agg["model"] == mname)]
            sub = sub.sort_values("imp_mean", ascending=False).head(10)
            print(f"\n  {mname}:")
            for _, r in sub.iterrows():
                marker = " [DEBATE]" if r["feature"] in DEBATE_FOCAL_6 else ""
                print(f"    {r['feature']:<25}{r['imp_mean']:>10.4f} ± {r['imp_std']:.4f}{marker}")

    # ─── Print SHAP top per T (XGB) ────────────────────────────────
    print("\n" + "=" * 100)
    print("SHAP (XGB) — top 15 features per T")
    print("=" * 100)
    for T in THRESHOLDS_PCT:
        sub = shap_agg[shap_agg["T_pct"] == T]
        sub = sub.sort_values("mean_abs_shap_mean", ascending=False).head(15)
        print(f"\n--- T = {T}% ---")
        for _, r in sub.iterrows():
            marker = " [DEBATE]" if r["feature"] in DEBATE_FOCAL_6 else ""
            print(f"    {r['feature']:<25}{r['mean_abs_shap_mean']:>10.5f} ± "
                  f"{r['mean_abs_shap_std']:.5f}{marker}")

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'threshold_importance_perm_full.csv'}")
    print(f"  {OUT_DIR / 'threshold_importance_perm_summary.csv'}")
    print(f"  {OUT_DIR / 'threshold_importance_shap_full.csv'}")
    print(f"  {OUT_DIR / 'threshold_importance_shap_summary.csv'}")


if __name__ == "__main__":
    main()
