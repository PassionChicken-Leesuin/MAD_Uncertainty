"""Per-field covariates — explanatory variables for "why does debate help
in some fields but not others?".

Computes 17 field-level covariates × 8 fields (n>100 in cohort), then
Spearman ρ vs Δ AUROC and Δ DOR (BIBLIO+6focal − BIBLIO).

Run AFTER experiments_per_field_sweep.py (needs per_field_delta_{auroc,dor}.csv
and per_field_threshold_summary.csv).

Covariate groups:
  Size/rarity       n, Y_rate_T{5,10,15,20}
  Baseline strength BL_AUROC (mean over 6 model × 4 T, BIBLIO config)
  Knowledge depth   PK_mean, SK_mean, TCT_mean, CKH_mean
  Assignee concen.  asg_HHI, top5_share
  IPC diversity     subclass: Shannon H, distinct count, top-1 share, tags/patent
                    main_group: Shannon H, distinct count
  Debate signal     H_final_mean, delta_H_mean, semantic_coherence_mean
  Quadrant          zX, zY (from per_field_quadrant_placement.csv)

Outputs:
  per_field_covariates.csv     — 8 rows × covariates
  per_field_correlation.csv    — Spearman ρ (long: T × metric × covariate)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

FIELDS_8 = ["01", "03", "07", "08", "09", "12", "13", "14"]
THRESHOLDS_PCT = [5, 10, 15, 20]

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_CSV = PROJECT / "data" / "variables_full_partial.csv"
OUT_DIR = PROJECT / "outputs"

ICE_ROOT = Path(r"C:\Users\User\OneDrive\문서\이수인\서울대학교\AI Agent\ICE_Domain")
IPC_LOOKUP = ICE_ROOT / "data_collection" / "intermediate" / "ipc_lookup.parquet"

AV_ROOT = Path(r"C:\Users\User\OneDrive\문서\이수인\서울대학교\AI Agent\Autonomous_Vehicle")
QUADRANT_CSV = AV_ROOT / "scripts" / "per_field_quadrant_placement.csv"

QUADRANT_KEY = {
    "01": "Q01_ICE", "03": "Q03_HCCI", "07": "Q07_Hybrids",
    "08": "Q08_EGR", "09": "Q09_Turbo", "12": "Q12_VVA",
    "13": "Q13_AltFuels", "14": "Q14_DirectInj",
}


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def main():
    # ─── Load cohort ─────────────────────────────────────────────────
    df = pd.read_csv(DATA_CSV, dtype={"patent_id": str, "fields": str},
                     encoding="utf-8-sig")
    df = df[df["mean_conf_pro"].notna()].reset_index(drop=True).copy()
    print(f"cohort (debate completed): {len(df):,}")

    # global Y at each T
    for T in THRESHOLDS_PCT:
        thr = np.percentile(df["forward5"].values, 100 - T)
        df[f"Y_T{T}"] = (df["forward5"] >= thr).astype(int)

    # ─── Load IPC lookup ─────────────────────────────────────────────
    ipc = pd.read_parquet(IPC_LOOKUP)
    ipc["patent_id"] = ipc["patent_id"].astype(str)
    print(f"ipc_lookup rows: {len(ipc):,} ({ipc['patent_id'].nunique():,} patents)")

    # ─── Load quadrant placement (zX, zY, asg_HHI, top5_share) ───────
    quad = pd.read_csv(QUADRANT_CSV)
    quad = quad.set_index("cohort")

    # ─── Build per-field covariate table ─────────────────────────────
    rows = []
    for f in FIELDS_8:
        in_field = df["fields"].str.split(",").apply(lambda L: f in L)
        sub = df.loc[in_field].copy()
        n = len(sub)
        pids = set(sub["patent_id"])
        ipc_sub = ipc[ipc["patent_id"].isin(pids)]

        # IPC subclass diversity
        sc_counts = ipc_sub["subclass_full"].value_counts()
        sc_entropy = shannon_entropy(sc_counts.values)
        sc_distinct = int(sc_counts.shape[0])
        sc_top1 = float(sc_counts.iloc[0] / sc_counts.sum()) if len(sc_counts) else float("nan")

        # IPC main_group diversity
        mg_counts = ipc_sub["main_group_full"].value_counts()
        mg_entropy = shannon_entropy(mg_counts.values)
        mg_distinct = int(mg_counts.shape[0])

        # Mean IPC tags per patent (from ipc_lookup, not all cohort patents are present)
        tags_per = ipc_sub.groupby("patent_id").size()
        tags_per_pat = float(tags_per.mean()) if len(tags_per) else float("nan")

        # Quadrant-derived covariates
        qkey = QUADRANT_KEY[f]
        q = quad.loc[qkey] if qkey in quad.index else None

        row = {
            "field": f,
            "n": n,
            **{f"Y_rate_T{T}": float(sub[f"Y_T{T}"].mean()) for T in THRESHOLDS_PCT},
            "PK_mean":  float(sub["PK"].mean()),
            "SK_mean":  float(sub["SK"].mean()),
            "TCT_mean": float(sub["TCT"].mean()),
            "CKH_mean": float(sub["CKH"].mean()),
            "asg_HHI":      float(q["asg_HHI"])     if q is not None else float("nan"),
            "top5_share":   float(q["top5_share"])  if q is not None else float("nan"),
            "zX":           float(q["zX"])          if q is not None else float("nan"),
            "zY":           float(q["zY"])          if q is not None else float("nan"),
            "ipc_subclass_entropy":     sc_entropy,
            "ipc_subclass_distinct":    sc_distinct,
            "ipc_subclass_top1_share":  sc_top1,
            "ipc_tags_per_patent_mean": tags_per_pat,
            "ipc_maingroup_entropy":    mg_entropy,
            "ipc_maingroup_distinct":   mg_distinct,
            "H_final_mean":           float(sub["H_final"].mean()),
            "delta_H_mean":           float(sub["delta_H"].mean()),
            "semantic_coherence_mean": float(sub["semantic_coherence"].mean()),
        }
        rows.append(row)

    cov = pd.DataFrame(rows).round(4)

    # ─── BL_AUROC: mean over 6 model × 4 T, BIBLIO config ────────────
    summ_path = OUT_DIR / "per_field_threshold_summary.csv"
    if summ_path.exists():
        summ = pd.read_csv(summ_path, dtype={"field": str})
        bl = summ[summ["config"] == "BIBLIO"].groupby("field")["AUROC_mean"].mean()
        cov["BL_AUROC"] = cov["field"].map(bl).round(4)
    else:
        print(f"WARN: {summ_path} not found — BL_AUROC will be NaN")
        cov["BL_AUROC"] = float("nan")

    cov_path = OUT_DIR / "per_field_covariates.csv"
    cov.to_csv(cov_path, index=False, encoding="utf-8-sig")
    print(f"\n[wrote] {cov_path}")
    print(cov.to_string(index=False))

    # ─── Spearman ρ: Δ vs covariate ──────────────────────────────────
    # Δ is averaged over models per (field, T) so we have 8 points per T.
    ρ_rows = []
    cov_cols = [c for c in cov.columns if c != "field"]

    for metric in ["AUROC", "DOR"]:
        delta_path = OUT_DIR / f"per_field_delta_{metric.lower()}.csv"
        if not delta_path.exists():
            print(f"WARN: {delta_path} not found — skipping {metric}")
            continue
        D = pd.read_csv(delta_path, dtype={"field": str})
        # average Δ over models → one Δ per (field, T)
        D_agg = D.groupby(["T_pct", "field"])["delta_mean"].mean().reset_index()

        for T in THRESHOLDS_PCT:
            sub = D_agg[D_agg["T_pct"] == T].set_index("field")["delta_mean"]
            # align to FIELDS_8 order
            delta_vec = np.array([sub.get(f, np.nan) for f in FIELDS_8])
            for c in cov_cols:
                cov_vec = cov.set_index("field").loc[FIELDS_8, c].values
                mask = ~(np.isnan(delta_vec) | np.isnan(cov_vec.astype(float)))
                if mask.sum() < 4:
                    rho, p = float("nan"), float("nan")
                else:
                    try:
                        rho, p = stats.spearmanr(delta_vec[mask], cov_vec[mask].astype(float))
                    except Exception:
                        rho, p = float("nan"), float("nan")
                ρ_rows.append({
                    "T_pct": T, "target_metric": metric,
                    "covariate": c,
                    "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
                    "p_value": float(p) if not np.isnan(p) else np.nan,
                    "n_fields": int(mask.sum()),
                })
    R = pd.DataFrame(ρ_rows).round({"spearman_rho": 3, "p_value": 4})
    corr_path = OUT_DIR / "per_field_correlation.csv"
    R.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"\n[wrote] {corr_path}")

    # Print top hits
    print("\n" + "=" * 80)
    print("Top |ρ| at each (T, target_metric) — first 5 rows by abs(ρ)")
    print("=" * 80)
    R["abs_rho"] = R["spearman_rho"].abs()
    for T in THRESHOLDS_PCT:
        for metric in ["AUROC", "DOR"]:
            block = R[(R["T_pct"] == T) & (R["target_metric"] == metric)] \
                .sort_values("abs_rho", ascending=False).head(5)
            if len(block):
                print(f"\n--- T={T}%, Δ{metric} ---")
                for _, r in block.iterrows():
                    sig = "*" if r["p_value"] < 0.05 else " "
                    print(f"  {sig} {r['covariate']:<30} ρ={r['spearman_rho']:+.3f}  p={r['p_value']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
