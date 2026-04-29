import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "experiments_summary"
agg = pd.read_csv(OUT / "threshold_sweep_summary.csv", encoding="utf-8-sig")
delta = pd.read_csv(OUT / "threshold_sweep_delta.csv", encoding="utf-8-sig")
dor_delta = pd.read_csv(OUT / "threshold_sweep_dor_delta.csv", encoding="utf-8-sig")

THRESHOLDS = [5, 10, 15, 20]
MODELS = ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]
CONFIGS = ["BIBLIO", "+6focal", "+25"]

# Y rates
print("Y rates by threshold:")
for T in THRESHOLDS:
    r = agg[agg["T_pct"] == T].iloc[0]
    print(f"  T={T}%: Y_rate={r['Y_rate_pct']}%")

# Per-T performance tables
for T in THRESHOLDS:
    print(f"\n=== T = {T}% (Y rate = {agg[agg['T_pct']==T]['Y_rate_pct'].iloc[0]}%) ===")
    sub = agg[agg["T_pct"] == T]
    print(f"  {'cfg/model':<14}{'AUROC':>10}{'AUPRC':>10}{'Prec':>9}{'Recall':>9}{'F1':>9}{'DOR':>9}")
    for cfg in CONFIGS:
        for m in MODELS:
            r = sub[(sub["config"] == cfg) & (sub["model"] == m)].iloc[0]
            print(f"  {cfg+'/'+m:<14}{r['AUROC_mean']:>10.4f}{r['AUPRC_mean']:>10.4f}"
                  f"{r['Precision_mean']:>9.4f}{r['Recall_mean']:>9.4f}"
                  f"{r['F1_mean']:>9.4f}{r['DOR_mean']:>9.2f}")

# Δ AUROC across T
print("\n=== Δ AUROC (config − BIBLIO) across thresholds ===")
print(f"  {'model':<6}{'config':<10}", end="")
for T in THRESHOLDS:
    print(f"{'T='+str(T)+'%':>14}", end="")
print()
for m in MODELS:
    for cfg in ["+6focal", "+25"]:
        print(f"  {m:<6}{cfg:<10}", end="")
        for T in THRESHOLDS:
            r = delta.query(f"T_pct == {T} and model == @m and config == @cfg").iloc[0]
            print(f"  {r['delta_mean']:+.4f} ({r['wins']:>2}/{r['n']})", end="")
        print()

# Δ DOR across T
print("\n=== Δ DOR (config − BIBLIO) across thresholds ===")
print(f"  {'model':<6}{'config':<10}", end="")
for T in THRESHOLDS:
    print(f"{'T='+str(T)+'%':>14}", end="")
print()
for m in MODELS:
    for cfg in ["+6focal", "+25"]:
        print(f"  {m:<6}{cfg:<10}", end="")
        for T in THRESHOLDS:
            r = dor_delta.query(f"T_pct == {T} and model == @m and config == @cfg").iloc[0]
            print(f"  {r['dor_delta_mean']:+7.2f} ({r['wins']:>2}/{r['n']})", end="")
        print()

# Key view: For each T, what's the best Δ?
print("\n=== Best Δ AUROC per T (across models × configs) ===")
for T in THRESHOLDS:
    sub = delta[delta["T_pct"] == T].copy()
    sub_pos = sub[sub["delta_mean"] > 0]
    if len(sub_pos) > 0:
        best = sub_pos.loc[sub_pos["delta_mean"].idxmax()]
        print(f"  T={T}%: best Δ AUROC = {best['delta_mean']:+.4f} "
              f"({best['model']}/{best['config']}, wins {best['wins']}/{best['n']})")
    else:
        print(f"  T={T}%: no positive Δ AUROC across all model/config combos")
