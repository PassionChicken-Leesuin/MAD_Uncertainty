"""Pretty-print per-field sweep results.

Loads:
  outputs/per_field_threshold_summary.csv
  outputs/per_field_delta_auroc.csv
  outputs/per_field_delta_dor.csv
  outputs/per_field_covariates.csv
  outputs/per_field_correlation.csv

Tables:
  1. Per-T compact: field × model grid of Δ AUROC (+6focal − BIBLIO)
  2. Per-T compact: field × model grid of Δ DOR
  3. BH-FDR significant cells per T
  4. Field-level covariate snapshot
  5. Top |ρ| Spearman correlations per (T, target)
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent / "outputs"

FIELDS_8 = ["01", "03", "07", "08", "09", "12", "13", "14"]
FIELD_LBL = {"01":"01-ICE","03":"03-HCCI","07":"07-Hyb","08":"08-EGR",
             "09":"09-Trb","12":"12-VVA","13":"13-Alt","14":"14-DI"}
THRESHOLDS = [5, 10, 15, 20]
MODELS = ["RF", "GBT", "XGB", "LogReg", "SVM", "FFN"]


def load(name):
    p = OUT / name
    if not p.exists():
        print(f"WARN: {p} missing — skipping")
        return None
    return pd.read_csv(p, dtype={"field": str}, encoding="utf-8-sig")


def grid_table(D, metric_col, label):
    """Render field × model grid for each T."""
    print(f"\n=== Δ {label} grid: rows = field, cols = model (mean across 20 seeds) ===")
    for T in THRESHOLDS:
        sub = D[D["T_pct"] == T]
        if sub.empty:
            continue
        print(f"\n--- T = {T}% ---")
        print(f"  {'field':<10}", end="")
        for m in MODELS:
            print(f"{m:>11}", end="")
        print()
        for f in FIELDS_8:
            print(f"  {FIELD_LBL[f]:<10}", end="")
            for m in MODELS:
                r = sub[(sub["field"] == f) & (sub["model"] == m)]
                if len(r) == 0:
                    print(f"{'—':>11}", end="")
                    continue
                v = float(r[metric_col].iloc[0])
                bh = float(r["p_bh"].iloc[0]) if "p_bh" in r.columns else float("nan")
                star = "*" if (not np.isnan(bh)) and bh < 0.05 else ("·" if (not np.isnan(bh)) and bh < 0.10 else " ")
                if metric_col == "delta_mean":
                    print(f"  {v:+.4f}{star} ", end="")
                else:
                    print(f"  {v:+7.2f}{star} ", end="")
            print()
        # FDR-sig count
        sig05 = int(sub["sig_bh_05"].sum()) if "sig_bh_05" in sub.columns else 0
        sig10 = int(sub["sig_bh_10"].sum()) if "sig_bh_10" in sub.columns else 0
        print(f"  BH-FDR sig: {sig05}/{len(sub)} at q<0.05, {sig10}/{len(sub)} at q<0.10")


def main():
    da = load("per_field_delta_auroc.csv")
    dd = load("per_field_delta_dor.csv")
    cov = load("per_field_covariates.csv")
    corr = load("per_field_correlation.csv")

    if da is not None:
        grid_table(da, "delta_mean", "AUROC")
    if dd is not None:
        grid_table(dd, "delta_mean", "DOR")

    # ─── Aggregated Δ per field (averaged over T × model) ────────────
    if da is not None:
        print("\n=== Field-averaged Δ AUROC (mean over 4 T × 6 model) ===")
        agg = da.groupby("field")["delta_mean"].agg(["mean", "std", "count"]).round(4)
        for f in FIELDS_8:
            if f in agg.index:
                r = agg.loc[f]
                print(f"  {FIELD_LBL[f]:<10}  Δ = {r['mean']:+.4f} ± {r['std']:.4f}  (n_cells={int(r['count'])})")

    # ─── Covariate snapshot ──────────────────────────────────────────
    if cov is not None:
        print("\n=== Field-level covariate snapshot ===")
        show_cols = [
            "field", "n", "Y_rate_T10", "BL_AUROC",
            "PK_mean", "TCT_mean", "asg_HHI",
            "ipc_subclass_entropy", "ipc_subclass_distinct",
            "H_final_mean", "semantic_coherence_mean",
            "zX", "zY",
        ]
        show_cols = [c for c in show_cols if c in cov.columns]
        c2 = cov[show_cols].copy()
        # pretty-print
        with pd.option_context("display.max_columns", None,
                                "display.width", 180,
                                "display.float_format", "{:>10.4f}".format):
            print(c2.to_string(index=False))

    # ─── Top correlations ────────────────────────────────────────────
    if corr is not None:
        corr["abs_rho"] = corr["spearman_rho"].abs()
        print("\n=== Top |ρ| per (T, target) ===")
        for T in THRESHOLDS:
            for metric in ["AUROC", "DOR"]:
                blk = corr[(corr["T_pct"] == T) & (corr["target_metric"] == metric)] \
                    .sort_values("abs_rho", ascending=False).head(6)
                if len(blk) == 0:
                    continue
                print(f"\n--- T = {T}%, target = Δ{metric} ---")
                for _, r in blk.iterrows():
                    sig = "*" if r["p_value"] < 0.05 else (" ·" if r["p_value"] < 0.10 else "  ")
                    print(f"  {sig} {r['covariate']:<32}  ρ = {r['spearman_rho']:+.3f}  p = {r['p_value']:.3f}")


if __name__ == "__main__":
    main()
