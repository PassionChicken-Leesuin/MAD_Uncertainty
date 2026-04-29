"""Per-field threshold sweep — does debate's marginal contribution
(BIBLIO vs BIBLIO+6focal) vary across the 8 ICE fields with n>100?

Field membership is explode-style: a patent tagged "08,09" is included
in BOTH field 08 and field 09's per-field experiments.

Y is GLOBAL top-T% (computed once on the full 7,086 cohort), shared
across all field subsets, so each field inherits its natural class
imbalance — the imbalance itself becomes part of the signal.

Sweep:
  fields ∈ {01, 03, 07, 08, 09, 12, 13, 14}
  T      ∈ {5, 10, 15, 20}%
  seeds  ∈ [42, 0..18]   (20 seeds)
  cfg    ∈ {BIBLIO, +6focal}
  models ∈ {RF, GBT, XGB, LogReg, SVM, FFN}
  → 8 × 4 × 20 × 2 × 6 = 7,680 fits

For each (field, T, seed): per-field stratified 80/20 on Y_T.
MF top-10 is computed from the per-field train fold.

Metrics: AUROC, AUPRC, Precision, Recall, F1, Accuracy, DOR (Haldane).

Outputs (in outputs/):
  per_field_threshold_full.csv      — per (field, T, seed, cfg, model)
  per_field_threshold_summary.csv   — seed mean ± std
  per_field_delta_auroc.csv         — paired Δ + paired-t p + BH-FDR per T
  per_field_delta_dor.csv           — same for DOR
"""

from __future__ import annotations

import io
import math
import sys
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
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

FIELDS_8 = ["01", "03", "07", "08", "09", "12", "13", "14"]
THRESHOLDS_PCT = [5, 10, 15, 20]
SEEDS = [42] + list(range(19))

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PARTIAL = PROJECT / "data" / "variables_full_partial.csv"
OUT_DIR = PROJECT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATENT_NUMERIC = ["CTO", "STO", "PK", "SK", "TCT",
                  "TS", "NC",
                  "COL", "INV", "TKH", "CKH", "PKH", "TTS", "CTS", "PTS"]

DEBATE_FOCAL_6 = [
    "var_conf_pro", "H_final", "delta_H",
    "conf_gap_change", "cross_domain_attack", "semantic_coherence",
]

CONFIGS = {
    "BIBLIO":  None,
    "+6focal": DEBATE_FOCAL_6,
}

MODEL_NAMES = ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]


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
    auroc = roc_auc_score(y_true, p_score) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, p_score) if len(np.unique(y_true)) > 1 else float("nan")
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


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def main():
    print("Loading data ...")
    full = pd.read_csv(PARTIAL, dtype={"patent_id": str, "fields": str},
                       encoding="utf-8-sig")
    df = full[full["mean_conf_pro"].notna()].reset_index(drop=True).copy()
    print(f"  cohort (debate completed): {len(df):,}")

    # GLOBAL Y for each T (computed once on full cohort)
    print("\nGlobal Y labels (top-T% on full cohort):")
    for T in THRESHOLDS_PCT:
        thr = np.percentile(df["forward5"].values, 100 - T)
        df[f"Y_T{T}"] = (df["forward5"] >= thr).astype(int)
        print(f"  T={T:>2}%  forward5 cutoff = {thr:>5.1f}  "
              f"global Y rate = {df[f'Y_T{T}'].mean()*100:.2f}%")

    # Tag membership: explode-style. df_field[f] = subset.
    print("\nField membership (explode-style):")
    field_idx = {}
    for f in FIELDS_8:
        mask = df["fields"].str.split(",").apply(lambda L: f in L)
        field_idx[f] = df.index[mask].to_numpy()
        print(f"  field {f}: n = {len(field_idx[f]):>5}")

    detail_rows = []
    for f in FIELDS_8:
        idx_field = field_idx[f]
        df_f = df.loc[idx_field].reset_index(drop=True)
        print(f"\n{'='*60}\nField {f}  (n = {len(df_f):,})\n{'='*60}")
        for T in THRESHOLDS_PCT:
            y = df_f[f"Y_T{T}"].values
            n_pos = int(y.sum())
            if n_pos < 2 or n_pos == len(y):
                print(f"  T={T}%: skipping — n_pos={n_pos}")
                continue
            Y_rate = y.mean() * 100
            for s_i, seed in enumerate(SEEDS):
                idx = np.arange(len(df_f))
                idx_tr, idx_te = train_test_split(
                    idx, test_size=0.2, stratify=y, random_state=seed)
                df_tr = df_f.iloc[idx_tr].reset_index(drop=True)
                df_te = df_f.iloc[idx_te].reset_index(drop=True)
                ytr, yte = y[idx_tr], y[idx_te]
                if int(yte.sum()) == 0 or int(yte.sum()) == len(yte):
                    continue
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
                            "field": f, "T_pct": T,
                            "Y_rate_pct": round(Y_rate, 2),
                            "seed": seed, "config": cfg_name, "model": mname,
                            "n_train": len(ytr), "n_test": len(yte),
                            "Y_pos_train": int(ytr.sum()),
                            "Y_pos_test": int(yte.sum()),
                        })
                        detail_rows.append(mt)
            print(f"  T={T}%  Y_rate={Y_rate:.2f}%  pos={n_pos}  done")

    D = pd.DataFrame(detail_rows)
    full_path = OUT_DIR / "per_field_threshold_full.csv"
    D.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"\n[wrote] {full_path}  ({len(D):,} rows)")

    # ─── Aggregate: seed mean ± std per (field, T, cfg, model) ───────
    metric_cols = ["AUROC", "AUPRC", "Precision", "Recall",
                   "F1", "Accuracy", "DOR"]
    agg = D.groupby(["field", "T_pct", "config", "model"]).agg(
        n_seeds=("seed", "count"),
        Y_rate_pct=("Y_rate_pct", "first"),
        n_test=("n_test", "first"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_std": (c, "std") for c in metric_cols},
    ).round(4).reset_index()
    summ_path = OUT_DIR / "per_field_threshold_summary.csv"
    agg.to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"[wrote] {summ_path}")

    # ─── Δ tables: +6focal vs BIBLIO, paired by seed ─────────────────
    for metric in ["AUROC", "DOR"]:
        rows = []
        for T in THRESHOLDS_PCT:
            for f in FIELDS_8:
                for mname in MODEL_NAMES:
                    base = D.query(
                        "T_pct == @T and field == @f and config == 'BIBLIO' and model == @mname"
                    ).set_index("seed")[metric]
                    aug = D.query(
                        "T_pct == @T and field == @f and config == '+6focal' and model == @mname"
                    ).set_index("seed")[metric]
                    common = sorted(set(base.index) & set(aug.index))
                    if len(common) < 3:
                        continue
                    paired = (aug.loc[common] - base.loc[common]).values
                    try:
                        tstat, pval = stats.ttest_rel(
                            aug.loc[common].values, base.loc[common].values)
                    except Exception:
                        tstat, pval = float("nan"), float("nan")
                    rows.append({
                        "T_pct": T, "field": f, "model": mname,
                        "delta_mean": float(np.mean(paired)),
                        "delta_std":  float(np.std(paired, ddof=1)),
                        "wins":       int((paired > 0).sum()),
                        "n_seeds":    len(paired),
                        "t_stat":     float(tstat),
                        "p_paired":   float(pval),
                    })
        DR = pd.DataFrame(rows, columns=[
            "T_pct", "field", "model", "delta_mean", "delta_std",
            "wins", "n_seeds", "t_stat", "p_paired",
        ])
        # BH-FDR per T (over 8 field × 6 model = 48 cells)
        DR["p_bh"] = np.nan
        if not DR.empty:
            for T in THRESHOLDS_PCT:
                mask = DR["T_pct"] == T
                if mask.sum() == 0:
                    continue
                DR.loc[mask, "p_bh"] = bh_fdr(DR.loc[mask, "p_paired"].values)
        DR["sig_bh_05"] = (DR["p_bh"] < 0.05).astype(int)
        DR["sig_bh_10"] = (DR["p_bh"] < 0.10).astype(int)
        DR = DR.round({
            "delta_mean": 4, "delta_std": 4,
            "t_stat": 3, "p_paired": 4, "p_bh": 4,
        })
        out = OUT_DIR / f"per_field_delta_{metric.lower()}.csv"
        DR.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[wrote] {out}  ({len(DR)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
