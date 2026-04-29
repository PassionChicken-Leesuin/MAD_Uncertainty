"""Pretty-print top features for the threshold importance experiment.

Reads:
  outputs/experiments_summary/threshold_importance_perm_summary.csv
  outputs/experiments_summary/threshold_importance_shap_summary.csv

Prints (with ASCII separators only — Windows cp949 safe):
  - Top 10 permutation features per (T, model)
  - Top 15 SHAP features per T (XGB)
  - Per-T debate-variable rank fraction (within-each-model rank
    averaged across debate features), to see if debate vars
    rise/fall as T moves.
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs" / "experiments_summary"
perm = pd.read_csv(OUT / "threshold_importance_perm_summary.csv",
                   encoding="utf-8-sig")
shap_ = pd.read_csv(OUT / "threshold_importance_shap_summary.csv",
                    encoding="utf-8-sig")

THRESHOLDS = [5, 10, 15, 20]
MODELS = ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]
DEBATE_FOCAL_6 = {
    "var_conf_pro", "H_final", "delta_H",
    "conf_gap_change", "cross_domain_attack", "semantic_coherence",
}


def tag(f):
    return " [DEBATE]" if f in DEBATE_FOCAL_6 else ""


print("=" * 90)
print("PERMUTATION IMPORTANCE -- top 10 features per (T, model)")
print("=" * 90)
for T in THRESHOLDS:
    print(f"\n--- T = {T}% ---")
    for m in MODELS:
        sub = perm[(perm["T_pct"] == T) & (perm["model"] == m)]
        sub = sub.sort_values("imp_mean", ascending=False).head(10)
        print(f"\n  {m}:")
        for _, r in sub.iterrows():
            print(f"    {r['feature']:<25}{r['imp_mean']:>9.4f} "
                  f"+/- {r['imp_std']:.4f}{tag(r['feature'])}")

print("\n" + "=" * 90)
print("SHAP (XGB) -- top 15 features per T")
print("=" * 90)
for T in THRESHOLDS:
    sub = shap_[shap_["T_pct"] == T].sort_values(
        "mean_abs_shap_mean", ascending=False).head(15)
    print(f"\n--- T = {T}% ---")
    for _, r in sub.iterrows():
        print(f"    {r['feature']:<25}{r['mean_abs_shap_mean']:>9.5f} "
              f"+/- {r['mean_abs_shap_std']:.5f}{tag(r['feature'])}")

# Average rank of each debate variable per (T, model)
print("\n" + "=" * 90)
print("DEBATE-VARIABLE PERM RANK per (T, model) -- 1 = top, 32 = bottom")
print("=" * 90)
for m in MODELS:
    print(f"\n  {m}:")
    print(f"    {'feature':<25}" + "".join(f"{'T='+str(T)+'%':>10}" for T in THRESHOLDS))
    for d in ["cross_domain_attack", "conf_gap_change", "var_conf_pro",
              "H_final", "delta_H", "semantic_coherence"]:
        line = f"    {d:<25}"
        for T in THRESHOLDS:
            sub = perm[(perm["T_pct"] == T) & (perm["model"] == m)].copy()
            sub = sub.sort_values("imp_mean", ascending=False).reset_index(drop=True)
            rk = sub.index[sub["feature"] == d].tolist()
            line += f"{(rk[0]+1) if rk else -1:>10}"
        print(line)

# Per-T SHAP rank for debate variables
print("\n" + "=" * 90)
print("DEBATE-VARIABLE SHAP RANK (XGB) per T -- 1 = top, 32 = bottom")
print("=" * 90)
print(f"  {'feature':<25}" + "".join(f"{'T='+str(T)+'%':>10}" for T in THRESHOLDS))
for d in ["cross_domain_attack", "conf_gap_change", "var_conf_pro",
          "H_final", "delta_H", "semantic_coherence"]:
    line = f"  {d:<25}"
    for T in THRESHOLDS:
        sub = shap_[shap_["T_pct"] == T].copy()
        sub = sub.sort_values("mean_abs_shap_mean", ascending=False).reset_index(drop=True)
        rk = sub.index[sub["feature"] == d].tolist()
        line += f"{(rk[0]+1) if rk else -1:>10}"
    print(line)

# Aggregate: how much total imp_mean is captured by 6 debate vars
print("\n" + "=" * 90)
print("DEBATE share of total positive perm importance per (T, model)")
print("=" * 90)
for m in MODELS:
    print(f"\n  {m}:")
    for T in THRESHOLDS:
        sub = perm[(perm["T_pct"] == T) & (perm["model"] == m)].copy()
        pos = sub[sub["imp_mean"] > 0]
        total = pos["imp_mean"].sum()
        debate = pos[pos["feature"].isin(DEBATE_FOCAL_6)]["imp_mean"].sum()
        share = (debate / total * 100) if total > 0 else 0.0
        print(f"    T={T:>2}%  total={total:.4f}  debate={debate:.4f}  share={share:.1f}%")

# SHAP debate share
print("\n" + "=" * 90)
print("DEBATE share of total SHAP (XGB) per T")
print("=" * 90)
for T in THRESHOLDS:
    sub = shap_[shap_["T_pct"] == T].copy()
    total = sub["mean_abs_shap_mean"].sum()
    debate = sub[sub["feature"].isin(DEBATE_FOCAL_6)]["mean_abs_shap_mean"].sum()
    share = (debate / total * 100) if total > 0 else 0.0
    print(f"  T={T:>2}%  total SHAP={total:.4f}  debate={debate:.4f}  share={share:.1f}%")
