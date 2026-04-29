"""Per-field 20-seed evaluation for ICE.

Procedure (mirrors manuscript Q3):
  - Global 80/20 stratified split (random_state=seed)
  - Train models on the full 80% train set
  - For each of 8 main fields, score the test patents whose `fields`
    contains that field id (test slice; sizes vary from ~26 to ~559)
  - Compute AUC, P@k, R@k, DOR@k per field

Aggregate across 20 seeds.
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

ICE_ROOT = Path(__file__).resolve().parents[2]
V2A = ICE_ROOT / "debate" / "runs" / "v2a_y_anchored" / "results" / "variables_full.csv"
V0  = ICE_ROOT / "debate" / "runs" / "v0_baseline"   / "results" / "variables_full_v0baseline.csv"
OUT_DIR = ICE_ROOT / "debate" / "runs"

NUM_INPUT = ["CTO","STO","PK","SK","TCT","TS","NC","COL","INV",
             "TKH","CKH","PKH","TTS","CTS","PTS"]
DEBATE5 = ["semantic_coherence","conf_gap_change","var_conf_pro","H_final","delta_H"]
DEBATE25 = ["mean_conf_pro","mean_conf_anti","var_conf_pro","var_conf_anti",
            "conf_gap_change","H_final","delta_H",
            "cross_domain_support","cross_domain_attack",
            "same_domain_support","same_domain_attack","acceptability_gap",
            "fact_ratio_pro","fact_ratio_anti",
            "final_prediction","prediction_volatility",
            "final_pred_technology","final_pred_application",
            "final_pred_user","final_pred_ecosystem","final_pred_businessmodel",
            "total_rounds","term_unanimous","term_extended_debate",
            "semantic_coherence"]
N_SEEDS = 20
K_PCT = 10  # top-10% of field test slice
FIELDS = ["01", "03", "07", "08", "09", "12", "13", "14"]

MODELS = {
    "RF":  lambda seed: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1),
    "GBT": lambda seed: GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed),
    "LogReg": lambda seed: LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
}


def field_in(fields_str, target):
    if pd.isna(fields_str): return False
    return target in {t.strip().zfill(2) for t in str(fields_str).split(",") if t.strip().isdigit()}


def metrics_at_k(y_true, scores, k_pct):
    n = len(y_true); k = max(1, int(np.ceil(n * k_pct / 100)))
    top = np.argsort(-scores)[:k]
    yh = np.zeros(n, dtype=int); yh[top] = 1
    tp = int(((y_true == 1) & (yh == 1)).sum())
    fp = int(((y_true == 0) & (yh == 1)).sum())
    fn = int(((y_true == 1) & (yh == 0)).sum())
    tn = int(((y_true == 0) & (yh == 0)).sum())
    n_pos = int(y_true.sum())
    P = tp / k if k > 0 else 0.0
    R = tp / n_pos if n_pos > 0 else 0.0
    DOR = ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))
    return P, R, DOR


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


def run_one_dataset(label, df):
    print(f"\n=== {label}: {N_SEEDS} seeds × 8 fields × 3 models × 3 feature sets ===", flush=True)
    y_full = df["Y"].astype(int).values
    fields_col = df["fields"].astype(str).values
    # Pre-compute field membership masks
    field_mask = {f: np.array([field_in(fs, f) for fs in fields_col]) for f in FIELDS}

    rows = []
    for seed in range(N_SEEDS):
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=0.20, stratify=y_full, random_state=seed)
        X_b   = build_biblio(df, train_idx)
        X_d5  = build_debate(df, train_idx, DEBATE5)
        X_d25 = build_debate(df, train_idx, DEBATE25)
        SETS = {
            "BIBLIO":          X_b,
            "BIBLIO+DEBATE5":  np.hstack([X_b, X_d5]),
            "BIBLIO+DEBATE25": np.hstack([X_b, X_d25]),
        }
        for fset, X in SETS.items():
            for mname, mfn in MODELS.items():
                m = mfn(seed).fit(X[train_idx], y_full[train_idx])
                p_test = m.predict_proba(X[test_idx])[:, 1]
                # global eval (for sanity)
                rows.append({
                    "dataset": label, "field": "GLOBAL", "seed": seed,
                    "feature_set": fset, "model": mname,
                    "n_test": len(test_idx), "n_pos": int(y_full[test_idx].sum()),
                    "auc": roc_auc_score(y_full[test_idx], p_test),
                    **dict(zip(["P@10","R@10","DOR@10"],
                               metrics_at_k(y_full[test_idx], p_test, K_PCT))),
                })
                # per-field slices
                for f in FIELDS:
                    f_mask = field_mask[f][test_idx]
                    if f_mask.sum() < 5 or y_full[test_idx][f_mask].sum() < 1:
                        continue  # skip degenerate
                    y_slice = y_full[test_idx][f_mask]
                    p_slice = p_test[f_mask]
                    if len(np.unique(y_slice)) < 2:
                        continue
                    P, R, DOR = metrics_at_k(y_slice, p_slice, K_PCT)
                    rows.append({
                        "dataset": label, "field": f, "seed": seed,
                        "feature_set": fset, "model": mname,
                        "n_test": int(f_mask.sum()), "n_pos": int(y_slice.sum()),
                        "auc": roc_auc_score(y_slice, p_slice),
                        "P@10": P, "R@10": R, "DOR@10": DOR,
                    })
        if (seed + 1) % 5 == 0:
            print(f"  seed {seed+1}/{N_SEEDS} done", flush=True)
    return pd.DataFrame(rows)


# ─── Run ──────────────────────────────────────────────────────────────
v0_df  = pd.read_csv(V0,  dtype={"patent_id": str})
v2a_df = pd.read_csv(V2A, dtype={"patent_id": str})

s_v0  = run_one_dataset("v0_baseline",   v0_df)
s_v2a = run_one_dataset("v2a_y_anchored", v2a_df)
raw = pd.concat([s_v0, s_v2a], ignore_index=True)
raw.to_csv(OUT_DIR / "perfield_multiseed_raw.csv", index=False)

# ─── Aggregate ────────────────────────────────────────────────────────
agg = raw.groupby(["dataset","field","feature_set","model"]).agg(
    n_test_mean=("n_test","mean"),
    n_pos_mean=("n_pos","mean"),
    auc_mean=("auc","mean"), auc_std=("auc","std"),
    P_mean=("P@10","mean"), P_std=("P@10","std"),
    R_mean=("R@10","mean"), R_std=("R@10","std"),
    DOR_mean=("DOR@10","mean"), DOR_std=("DOR@10","std"),
).reset_index()
agg.to_csv(OUT_DIR / "perfield_multiseed_agg.csv", index=False)

# ─── Tables: per-field AUC and DOR (RF only, key feature sets) ───────
print("\n" + "=" * 110)
print(f"PER-FIELD AUC@RF — {N_SEEDS}-seed mean ± std (k={K_PCT}%)")
print("=" * 110)
field_names = {
    "01":"Q01 ICE",     "03":"Q03 HCCI",     "07":"Q07 Hybrids",  "08":"Q08 EGR",
    "09":"Q09 Turbo",   "12":"Q12 VVA",      "13":"Q13 AltFuels", "14":"Q14 DI",
    "GLOBAL":"GLOBAL",
}

for ds in ["v0_baseline", "v2a_y_anchored"]:
    print(f"\n────── {ds}, model=RF ──────")
    sub = agg[(agg["dataset"]==ds) & (agg["model"]=="RF")]
    print(f"{'field':<14}{'fset':<18}{'n_test':>8}{'n_pos':>7}  AUC mean±std       P@10 mean±std    DOR@10 mean±std")
    for f in ["GLOBAL"] + FIELDS:
        for fset in ["BIBLIO","BIBLIO+DEBATE5","BIBLIO+DEBATE25"]:
            r = sub[(sub["field"]==f) & (sub["feature_set"]==fset)]
            if len(r) == 0: continue
            r = r.iloc[0]
            print(f"  {field_names[f]:<12}{fset:<18}{r['n_test_mean']:>8.0f}{r['n_pos_mean']:>7.0f}  "
                  f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}   "
                  f"{r['P_mean']:.3f}±{r['P_std']:.3f}    "
                  f"{r['DOR_mean']:>5.2f}±{r['DOR_std']:.2f}")

# ─── Δ AUC: (BIBLIO+DEBATE) − BIBLIO per field, paired across seeds ──
print("\n" + "=" * 110)
print("PAIRED Δ AUC per field (RF) — same seed, BIBLIO+DEBATE vs BIBLIO")
print("=" * 110)
for ds in ["v0_baseline", "v2a_y_anchored"]:
    print(f"\n────── {ds} ──────")
    print(f"{'field':<14}{'fset':<18}{'mean Δ':>10}{'std Δ':>10}{'win':>8}{'AUC bib':>10}{'AUC aug':>10}")
    sub = raw[(raw["dataset"]==ds) & (raw["model"]=="RF")]
    for f in ["GLOBAL"] + FIELDS:
        for fset in ["BIBLIO+DEBATE5","BIBLIO+DEBATE25"]:
            bib = sub[(sub["field"]==f) & (sub["feature_set"]=="BIBLIO")].set_index("seed")["auc"]
            aug = sub[(sub["field"]==f) & (sub["feature_set"]==fset)].set_index("seed")["auc"]
            common = bib.index.intersection(aug.index)
            if len(common) < 5: continue
            d = aug[common] - bib[common]
            wins = int((d > 0).sum())
            print(f"  {field_names[f]:<12}{fset:<18}{d.mean():>+10.4f}{d.std():>+10.4f}"
                  f"{wins:>5}/{len(common):<2}  {bib[common].mean():>9.4f} {aug[common].mean():>9.4f}")

# ─── Cross-prompt: v2a − v0 paired, per field ──────────────────────
print("\n" + "=" * 110)
print(f"PAIRED Δ AUC per field — v2a vs v0 (same seed, RF)")
print("=" * 110)
for fset in ["BIBLIO","BIBLIO+DEBATE5","BIBLIO+DEBATE25"]:
    print(f"\n────── {fset}, model=RF ──────")
    print(f"{'field':<14}{'mean Δ':>10}{'std Δ':>10}{'win':>10}")
    for f in ["GLOBAL"] + FIELDS:
        v0_a  = raw[(raw["dataset"]=="v0_baseline") & (raw["field"]==f) &
                    (raw["model"]=="RF") & (raw["feature_set"]==fset)].set_index("seed")["auc"]
        v2a_a = raw[(raw["dataset"]=="v2a_y_anchored") & (raw["field"]==f) &
                    (raw["model"]=="RF") & (raw["feature_set"]==fset)].set_index("seed")["auc"]
        common = v0_a.index.intersection(v2a_a.index)
        if len(common) < 5: continue
        d = v2a_a[common] - v0_a[common]
        wins = int((d > 0).sum())
        print(f"  {field_names[f]:<12}{d.mean():>+10.4f}{d.std():>+10.4f}{wins:>7}/{len(common):<2}")
