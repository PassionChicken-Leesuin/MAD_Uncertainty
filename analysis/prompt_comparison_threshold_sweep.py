"""FOCAL-6 + 6 models + Y Threshold Sweep multi-seed evaluation.

FOCAL-6 (per ICE manuscript spec):
  cross_domain_attack, conf_gap_change, var_conf_pro,
  H_final, delta_H, semantic_coherence

Models (6):
  LogReg, RF, GBT, XGB, SVM, FFN

Feature sets (3):
  BIBLIO          : 16 input vars (26 dim with MF top-10)
  FOCAL6          : 6 debate vars only
  BIBLIO+FOCAL6   : 26 + 6 = 32 dim

Y threshold sweep (T = 5, 10, 15, 20 %):
  At top-T% cutoff by predicted score:
    Precision, Recall, F1, DOR (Haldane-corrected)

Threshold-independent:
  AUROC, AUPRC

20 seeds × 80/20 stratified split. Train on 80%, evaluate on 20% test.
Aggregate mean ± std across seeds.
"""
from __future__ import annotations
import sys, io, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

ICE_ROOT = Path(__file__).resolve().parents[2]
V2A = ICE_ROOT / "debate" / "runs" / "v2a_y_anchored" / "results" / "variables_full.csv"
V0  = ICE_ROOT / "debate" / "runs" / "v0_baseline"   / "results" / "variables_full_v0baseline.csv"
OUT_DIR = ICE_ROOT / "debate" / "runs"

NUM_INPUT = ["CTO","STO","PK","SK","TCT","TS","NC","COL","INV",
             "TKH","CKH","PKH","TTS","CTS","PTS"]
FOCAL6 = ["cross_domain_attack","conf_gap_change","var_conf_pro",
          "H_final","delta_H","semantic_coherence"]

N_SEEDS = 20
T_PERCENTAGES = [5, 10, 15, 20]


def make_models(seed, n_features):
    h = max(2, math.ceil(math.sqrt(n_features * 2)))
    return {
        "LogReg": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "RF":     RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1),
        "GBT":    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed),
        "XGB":    XGBClassifier(n_estimators=300, tree_method="hist", random_state=seed,
                                eval_metric="logloss", verbosity=0, n_jobs=-1),
        "SVM":    SVC(kernel="rbf", probability=True, random_state=seed),
        "FFN":    MLPClassifier(hidden_layer_sizes=(h,), activation="logistic",
                                solver="lbfgs", alpha=0.001, max_iter=1000, random_state=seed),
    }


def metrics_at_T(y_true, scores, T_pct):
    """Top-T% cutoff metrics."""
    n = len(y_true); k = max(1, int(np.ceil(n * T_pct / 100)))
    top = np.argsort(-scores)[:k]
    yh = np.zeros(n, dtype=int); yh[top] = 1
    tp = int(((y_true == 1) & (yh == 1)).sum())
    fp = int(((y_true == 0) & (yh == 1)).sum())
    fn = int(((y_true == 1) & (yh == 0)).sum())
    tn = int(((y_true == 0) & (yh == 0)).sum())
    n_pos = int(y_true.sum())
    P = tp / k if k > 0 else 0.0
    R = tp / n_pos if n_pos > 0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    DOR = ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))
    return {"precision": P, "recall": R, "f1": F1, "DOR": DOR,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "k": k}


def build_biblio(df, train_idx):
    df = df.copy()
    train = df.iloc[train_idx]
    top10 = train["MF"].value_counts().head(10).index.tolist()
    for mf in top10:
        df[f"MF_{mf}"] = (df["MF"] == mf).astype(int)
    df["MF_other"] = (~df["MF"].isin(top10)).astype(int)
    cat_cols = [f"MF_{m}" for m in top10] + ["MF_other"]
    imp = SimpleImputer(strategy="median").fit(df.iloc[train_idx][NUM_INPUT])
    df[NUM_INPUT] = imp.transform(df[NUM_INPUT])
    sc = MinMaxScaler().fit(df.iloc[train_idx][NUM_INPUT])
    df[NUM_INPUT] = sc.transform(df[NUM_INPUT])
    return df[NUM_INPUT + cat_cols].values.astype(float)


def build_debate(df, train_idx, cols):
    df = df.copy()
    imp = SimpleImputer(strategy="median").fit(df.iloc[train_idx][cols])
    df[cols] = imp.transform(df[cols])
    sc = MinMaxScaler().fit(df.iloc[train_idx][cols])
    df[cols] = sc.transform(df[cols])
    return df[cols].values.astype(float)


def run_dataset(label, df):
    t0 = time.time()
    print(f"\n=== {label}: {N_SEEDS} seeds × 6 models × 3 fsets × 4 thresholds ===", flush=True)
    y_full = df["Y"].astype(int).values
    rows = []
    for seed in range(N_SEEDS):
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=0.20, stratify=y_full, random_state=seed)
        X_b   = build_biblio(df, train_idx)
        X_f6  = build_debate(df, train_idx, FOCAL6)
        SETS = {
            "BIBLIO":         X_b,
            "FOCAL6":         X_f6,
            "BIBLIO+FOCAL6":  np.hstack([X_b, X_f6]),
        }
        for fset, X in SETS.items():
            models = make_models(seed, X.shape[1])
            for mname, model in models.items():
                try:
                    model.fit(X[train_idx], y_full[train_idx])
                    p = model.predict_proba(X[test_idx])[:, 1]
                except Exception as e:
                    print(f"  seed {seed} {mname} {fset} FAILED: {e}", flush=True)
                    continue
                y_te = y_full[test_idx]
                auroc = roc_auc_score(y_te, p)
                auprc = average_precision_score(y_te, p)
                for T in T_PERCENTAGES:
                    m = metrics_at_T(y_te, p, T)
                    rows.append({
                        "dataset": label, "feature_set": fset, "model": mname,
                        "seed": seed, "T_pct": T,
                        "auroc": auroc, "auprc": auprc,
                        **{k: v for k, v in m.items()}
                    })
        elapsed = time.time() - t0
        print(f"  seed {seed+1:>2}/{N_SEEDS} done  ({elapsed/60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


# ─── Run ──────────────────────────────────────────────────────────────
v0_df  = pd.read_csv(V0,  dtype={"patent_id": str})
v2a_df = pd.read_csv(V2A, dtype={"patent_id": str})

s_v0  = run_dataset("v0_baseline",   v0_df)
s_v2a = run_dataset("v2a_y_anchored", v2a_df)
raw = pd.concat([s_v0, s_v2a], ignore_index=True)
raw.to_csv(OUT_DIR / "focal6_threshold_sweep_raw.csv", index=False)

# ─── Aggregate ────────────────────────────────────────────────────────
# AUROC/AUPRC are threshold-independent — group without T
auc_agg = raw.groupby(["dataset","feature_set","model"]).agg(
    auroc_mean=("auroc","mean"), auroc_std=("auroc","std"),
    auprc_mean=("auprc","mean"), auprc_std=("auprc","std"),
).reset_index()

# Threshold-dependent
T_agg = raw.groupby(["dataset","feature_set","model","T_pct"]).agg(
    P_mean=("precision","mean"),  P_std=("precision","std"),
    R_mean=("recall","mean"),     R_std=("recall","std"),
    F1_mean=("f1","mean"),        F1_std=("f1","std"),
    DOR_mean=("DOR","mean"),      DOR_std=("DOR","std"),
).reset_index()

auc_agg.to_csv(OUT_DIR / "focal6_AUROC_AUPRC_agg.csv", index=False)
T_agg.to_csv(  OUT_DIR / "focal6_threshold_metrics_agg.csv", index=False)

# ─── Print main tables ───────────────────────────────────────────────
print("\n" + "=" * 110)
print("AUROC & AUPRC — threshold-independent (20-seed mean ± std)")
print("=" * 110)
for ds in ["v0_baseline", "v2a_y_anchored"]:
    print(f"\n[{ds}]")
    print(f"{'feature_set':<18}{'model':<8}  {'AUROC':>15}  {'AUPRC':>15}")
    sub = auc_agg[auc_agg["dataset"]==ds]
    for fset in ["BIBLIO","FOCAL6","BIBLIO+FOCAL6"]:
        for m in ["LogReg","RF","GBT","XGB","SVM","FFN"]:
            r = sub[(sub["feature_set"]==fset) & (sub["model"]==m)]
            if len(r) == 0: continue
            r = r.iloc[0]
            print(f"  {fset:<16}{m:<8}  {r['auroc_mean']:.4f}±{r['auroc_std']:.4f}   "
                  f"{r['auprc_mean']:.4f}±{r['auprc_std']:.4f}")

print("\n" + "=" * 110)
print("Threshold sweep — Precision / Recall / F1 / DOR  (20-seed mean ± std)")
print("=" * 110)
for ds in ["v0_baseline", "v2a_y_anchored"]:
    print(f"\n[{ds}]")
    for fset in ["BIBLIO","FOCAL6","BIBLIO+FOCAL6"]:
        print(f"\n  ── {fset} ──")
        print(f"  {'model':<8}{'T':>4}  {'P':>15}  {'R':>15}  {'F1':>15}  {'DOR':>15}")
        sub = T_agg[(T_agg["dataset"]==ds) & (T_agg["feature_set"]==fset)]
        for m in ["LogReg","RF","GBT","XGB","SVM","FFN"]:
            for T in T_PERCENTAGES:
                r = sub[(sub["model"]==m) & (sub["T_pct"]==T)]
                if len(r)==0: continue
                r = r.iloc[0]
                print(f"  {m:<7}{T:>4}%  {r['P_mean']:.3f}±{r['P_std']:.3f}    "
                      f"{r['R_mean']:.3f}±{r['R_std']:.3f}    "
                      f"{r['F1_mean']:.3f}±{r['F1_std']:.3f}    "
                      f"{r['DOR_mean']:>5.2f}±{r['DOR_std']:>4.2f}")

# ─── v2a vs v0 paired Δ AUROC and Δ DOR@10 head-to-head ──────────────
print("\n" + "=" * 110)
print("HEAD-TO-HEAD: v2a vs v0 — paired Δ AUROC and Δ metric @ each T (20 seeds)")
print("=" * 110)

cells_auc = []
for fset in ["FOCAL6","BIBLIO+FOCAL6"]:
    for m in ["LogReg","RF","GBT","XGB","SVM","FFN"]:
        v0_b = raw[(raw.dataset=="v0_baseline") & (raw.feature_set=="BIBLIO") & (raw.model==m) & (raw.T_pct==10)].set_index("seed")["auroc"]
        v0_a = raw[(raw.dataset=="v0_baseline") & (raw.feature_set==fset) & (raw.model==m) & (raw.T_pct==10)].set_index("seed")["auroc"]
        v2_b = raw[(raw.dataset=="v2a_y_anchored") & (raw.feature_set=="BIBLIO") & (raw.model==m) & (raw.T_pct==10)].set_index("seed")["auroc"]
        v2_a = raw[(raw.dataset=="v2a_y_anchored") & (raw.feature_set==fset) & (raw.model==m) & (raw.T_pct==10)].set_index("seed")["auroc"]
        common = v0_b.index.intersection(v0_a.index).intersection(v2_b.index).intersection(v2_a.index)
        d_v0  = (v0_a[common] - v0_b[common])
        d_v2a = (v2_a[common] - v2_b[common])
        cells_auc.append({
            "feature_set": fset, "model": m,
            "v0_d_AUROC_mean":  d_v0.mean(),  "v0_win":  int((d_v0 > 0).sum()),
            "v2a_d_AUROC_mean": d_v2a.mean(), "v2a_win": int((d_v2a > 0).sum()),
        })

print("\nΔ AUROC vs BIBLIO (paired, 20 seeds):")
print(f"  {'feature_set':<18}{'model':<8}  {'v0 mean Δ':>12}  {'v0 win':>10}  {'v2a mean Δ':>12}  {'v2a win':>10}  Winner")
for c in cells_auc:
    winner = "v2a" if c["v2a_d_AUROC_mean"] > c["v0_d_AUROC_mean"] else (
             "v0"  if c["v2a_d_AUROC_mean"] < c["v0_d_AUROC_mean"] else "tie")
    print(f"  {c['feature_set']:<16}{c['model']:<8}  {c['v0_d_AUROC_mean']:>+12.4f}  "
          f"{c['v0_win']:>5}/20    {c['v2a_d_AUROC_mean']:>+12.4f}  "
          f"{c['v2a_win']:>5}/20    {winner}")

# ─── Aggregated head-to-head tally ───────────────────────────────────
df_cells = pd.DataFrame(cells_auc)
df_cells["v2a_better"] = df_cells["v2a_d_AUROC_mean"] > df_cells["v0_d_AUROC_mean"]
n = len(df_cells)
print(f"\n{'='*110}")
print(f"AGGREGATE: 12 cells (2 fsets × 6 models)")
print(f"  v2a > v0 (paired Δ AUROC): {int(df_cells['v2a_better'].sum())}/{n}  ({df_cells['v2a_better'].mean()*100:.1f}%)")
print(f"  Mean v0 Δ:  {df_cells['v0_d_AUROC_mean'].mean():+.4f}")
print(f"  Mean v2a Δ: {df_cells['v2a_d_AUROC_mean'].mean():+.4f}")
print(f"  Paired diff: {(df_cells['v2a_d_AUROC_mean']-df_cells['v0_d_AUROC_mean']).mean():+.4f}")

df_cells.to_csv(OUT_DIR / "focal6_headtohead.csv", index=False)
print(f"\nSaved: focal6_threshold_sweep_raw.csv, focal6_AUROC_AUPRC_agg.csv, "
      f"focal6_threshold_metrics_agg.csv, focal6_headtohead.csv")
