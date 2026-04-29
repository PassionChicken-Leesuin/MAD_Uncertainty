"""Threshold sweep — does debate's marginal contribution depend on
how strict the "promising" Y label is?

Sweeps the Y-defining threshold across T ∈ {5%, 10%, 15%, 20%} of
forward5 and re-runs the same multi-seed protocol at each threshold.

For each threshold T:
  - Y_T = (forward5 >= percentile(100-T))
  - Y rate slightly above T due to ties at the boundary
  - 6 models (RF / GBT / XGB / LogReg / SVM / FFN), 3 configs (BIBLIO / +25 / +6focal)
  - 20 seeds, 80/20 stratified random on Y_T
  - Metrics at top-T% prediction threshold:
      AUROC, AUPRC (threshold-free)
      Precision, Recall, DOR (Haldane), F1, Accuracy (top-T% threshold)

Output:
  outputs/experiments_summary/threshold_sweep_full.csv (per seed × T × cfg × model)
  outputs/experiments_summary/threshold_sweep_summary.csv (aggregated mean ± std)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
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

DEBATE_FOCAL_6 = [
    "var_conf_pro", "H_final", "delta_H",
    "conf_gap_change", "cross_domain_attack", "semantic_coherence",
]

CONFIGS = {
    "BIBLIO":  None,
    "+25":     DEBATE_ALL,
    "+6focal": DEBATE_FOCAL_6,
}


def haldane_dor(tp, tn, fp, fn):
    return ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))


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


def metrics_at_threshold(y_true, p_score, top_pct):
    """Compute AUROC/AUPRC plus confusion-matrix metrics at top-T% threshold."""
    auroc = roc_auc_score(y_true, p_score)
    auprc = average_precision_score(y_true, p_score)
    thr = np.quantile(p_score, 1 - top_pct / 100)
    y_pred = (p_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "DOR": haldane_dor(tp, tn, fp, fn),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def main():
    print("Loading data ...")
    full = pd.read_csv(PARTIAL, dtype={"patent_id": str, "fields": str},
                       encoding="utf-8-sig")
    df = full[full["mean_conf_pro"].notna()].reset_index(drop=True).copy()
    print(f"  cohort: {len(df):,}")

    # Compute Y labels for each threshold
    print("\nThreshold-specific Y labels:")
    for T in THRESHOLDS_PCT:
        thr = np.percentile(df["forward5"].values, 100 - T)
        Y_T = (df["forward5"] >= thr).astype(int)
        rate = Y_T.mean() * 100
        print(f"  T={T:>2}%  cutoff forward5 >= {thr:>5.1f}  "
              f"Y rate = {rate:.2f}%  n_pos={Y_T.sum()}")
        df[f"Y_T{T}"] = Y_T

    detail_rows = []
    for T in THRESHOLDS_PCT:
        print(f"\n{'='*60}\nThreshold T = {T}%\n{'='*60}")
        y = df[f"Y_T{T}"].values
        idx = np.arange(len(df))
        Y_rate = y.mean() * 100

        for s_i, seed in enumerate(SEEDS):
            idx_tr, idx_te = train_test_split(
                idx, test_size=0.2, stratify=y, random_state=seed)
            df_tr = df.iloc[idx_tr].reset_index(drop=True)
            df_te = df.iloc[idx_te].reset_index(drop=True)
            ytr, yte = y[idx_tr], y[idx_te]
            mf_top = df_tr["MF"].value_counts().head(10).index.tolist()

            for cfg_name, debate_cols in CONFIGS.items():
                Xtr = build_X(df_tr, debate_cols, mf_top)
                Xte = build_X(df_te, debate_cols, mf_top)
                imp = SimpleImputer(strategy="median")
                sc = MinMaxScaler()
                Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
                Xte_s = sc.transform(imp.transform(Xte))
                models = make_models(Xtr_s.shape[1], seed=seed)
                for mname, m in models.items():
                    m.fit(Xtr_s, ytr)
                    p = m.predict_proba(Xte_s)[:, 1]
                    mt = metrics_at_threshold(yte, p, top_pct=T)
                    mt.update({
                        "T_pct": T, "Y_rate_pct": round(Y_rate, 2),
                        "seed": seed, "config": cfg_name, "model": mname,
                        "n_test": len(yte), "Y_pos_test": int(yte.sum()),
                    })
                    detail_rows.append(mt)
            if (s_i + 1) % 5 == 0:
                print(f"  T={T}%, seed {seed}: done ({s_i+1}/{len(SEEDS)})")

    D = pd.DataFrame(detail_rows)
    D.to_csv(OUT_DIR / "threshold_sweep_full.csv",
             index=False, encoding="utf-8-sig")

    # Aggregate
    metric_cols = ["AUROC", "AUPRC", "Precision", "Recall",
                    "F1", "Accuracy", "DOR"]
    agg = D.groupby(["T_pct", "config", "model"]).agg(
        Y_rate_pct=("Y_rate_pct", "first"),
        n_seeds=("seed", "count"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_std": (c, "std") for c in metric_cols},
    ).round(4).reset_index()
    agg.to_csv(OUT_DIR / "threshold_sweep_summary.csv",
               index=False, encoding="utf-8-sig")

    # ─── Print summary tables ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Threshold sweep summary — mean over 20 seeds")
    print("=" * 100)
    for T in THRESHOLDS_PCT:
        sub = agg[agg["T_pct"] == T]
        Y_rate = sub["Y_rate_pct"].iloc[0]
        print(f"\n--- T = {T}% (Y rate = {Y_rate}%) ---")
        print(f"  {'cfg/model':<14}{'AUROC':>10}{'AUPRC':>10}{'Prec':>9}{'Recall':>9}{'F1':>9}{'DOR':>9}")
        for cfg in ["BIBLIO", "+6focal", "+25"]:
            for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
                r = sub[(sub["config"] == cfg) & (sub["model"] == mname)].iloc[0]
                print(f"  {cfg+'/'+mname:<14}"
                      f"{r['AUROC_mean']:>10.4f}{r['AUPRC_mean']:>10.4f}"
                      f"{r['Precision_mean']:>9.4f}{r['Recall_mean']:>9.4f}"
                      f"{r['F1_mean']:>9.4f}{r['DOR_mean']:>9.2f}")

    # Δ tables: +6focal vs BIBLIO and +25 vs BIBLIO
    print("\n" + "=" * 100)
    print("Δ AUROC vs BIBLIO across thresholds (paired by seed)")
    print("=" * 100)
    delta_rows = []
    for T in THRESHOLDS_PCT:
        for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
            for cfg in ["+6focal", "+25"]:
                paired = []
                for s in SEEDS:
                    base = D.query(f"T_pct == {T} and seed == @s and config == 'BIBLIO' and model == @mname")["AUROC"]
                    aug  = D.query(f"T_pct == {T} and seed == @s and config == @cfg and model == @mname")["AUROC"]
                    if len(base) and len(aug):
                        paired.append(aug.iloc[0] - base.iloc[0])
                paired = np.array(paired)
                wins = int((paired > 0).sum())
                delta_rows.append({
                    "T_pct": T, "model": mname, "config": cfg,
                    "delta_mean": round(paired.mean(), 4),
                    "delta_std":  round(paired.std(), 4),
                    "wins":       wins,
                    "n":          len(paired),
                })
    DR = pd.DataFrame(delta_rows)
    DR.to_csv(OUT_DIR / "threshold_sweep_delta.csv",
              index=False, encoding="utf-8-sig")

    print(f"\n  {'T':>3} {'model':>5} {'config':>9} {'Δmean':>9} {'Δstd':>9} {'wins/20':>10}")
    for _, r in DR.iterrows():
        print(f"  {r['T_pct']:>3} {r['model']:>5} {r['config']:>9} "
              f"{r['delta_mean']:>9.4f} {r['delta_std']:>9.4f} "
              f"{r['wins']:>4}/{r['n']:<3}")

    # Compact: show debate-uplift Δ AUROC trend for +6focal across T
    print("\n" + "=" * 100)
    print("Δ AUROC (+6focal vs BIBLIO) across thresholds — compact view")
    print("=" * 100)
    print(f"  {'model':<6}", end="")
    for T in THRESHOLDS_PCT:
        print(f"{'T='+str(T)+'%':>14}", end="")
    print()
    for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
        print(f"  {mname:<6}", end="")
        for T in THRESHOLDS_PCT:
            r = DR.query(f"T_pct == {T} and model == @mname and config == '+6focal'").iloc[0]
            mark = "+" if r["delta_mean"] > 0 else " "
            print(f"  {mark}{r['delta_mean']:+.4f} ({r['wins']}/{r['n']})", end="")
        print()

    print("\n" + "=" * 100)
    print("Δ DOR (+6focal vs BIBLIO) across thresholds")
    print("=" * 100)
    delta_dor = []
    for T in THRESHOLDS_PCT:
        for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
            for cfg in ["+6focal", "+25"]:
                paired = []
                for s in SEEDS:
                    base = D.query(f"T_pct == {T} and seed == @s and config == 'BIBLIO' and model == @mname")["DOR"]
                    aug  = D.query(f"T_pct == {T} and seed == @s and config == @cfg and model == @mname")["DOR"]
                    if len(base) and len(aug):
                        paired.append(aug.iloc[0] - base.iloc[0])
                paired = np.array(paired)
                delta_dor.append({
                    "T_pct": T, "model": mname, "config": cfg,
                    "dor_delta_mean": round(paired.mean(), 3),
                    "dor_delta_std":  round(paired.std(), 3),
                    "wins": int((paired > 0).sum()),
                    "n":    len(paired),
                })
    DD = pd.DataFrame(delta_dor)
    DD.to_csv(OUT_DIR / "threshold_sweep_dor_delta.csv",
              index=False, encoding="utf-8-sig")

    print(f"  {'model':<6}", end="")
    for T in THRESHOLDS_PCT:
        print(f"{'T='+str(T)+'%':>14}", end="")
    print()
    for mname in ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]:
        print(f"  {mname:<6}", end="")
        for T in THRESHOLDS_PCT:
            r = DD.query(f"T_pct == {T} and model == @mname and config == '+6focal'").iloc[0]
            print(f"  {r['dor_delta_mean']:+8.2f} ({r['wins']}/{r['n']})", end="")
        print()

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'threshold_sweep_full.csv'}")
    print(f"  {OUT_DIR / 'threshold_sweep_summary.csv'}")
    print(f"  {OUT_DIR / 'threshold_sweep_delta.csv'}")
    print(f"  {OUT_DIR / 'threshold_sweep_dor_delta.csv'}")


if __name__ == "__main__":
    main()
