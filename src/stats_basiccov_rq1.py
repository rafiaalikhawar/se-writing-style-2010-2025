from pathlib import Path
import pandas as pd

METRICS_DIR = Path("data/metrics")
OUT_SUMMARY = METRICS_DIR / "rq1_bucket_summary.csv"
OUT_MWU_ALL = METRICS_DIR / "rq1_mwu_all.csv"

order = ["2010-2014","2015-2019","2020-2025"]

# --- per-bucket medians (and n) ---
# readability
r = pd.read_csv(METRICS_DIR / "readability_per_paper.csv")
r = r[r["rq1_bucket"].isin(order)].copy()
r["rq1_bucket"] = pd.Categorical(r["rq1_bucket"], categories=order, ordered=True)
rb = (r.groupby("rq1_bucket")[["fre","fkgl"]]
        .agg(n=("fre","count"), fre_median=("fre","median"), fkgl_median=("fkgl","median"))
        .reset_index())

# passive
p = pd.read_csv(METRICS_DIR / "passive_per_paper.csv")
p = p[p["rq1_bucket"].isin(order)].copy()
p["rq1_bucket"] = pd.Categorical(p["rq1_bucket"], categories=order, ordered=True)
pb = (p.groupby("rq1_bucket")[["passive_ratio"]]
        .agg(passive_median=("passive_ratio","median")).reset_index())

# basic coverage (use the combined file you just made)
b = pd.read_csv(METRICS_DIR / "basiccov_summary_all.csv")
b = b[b["rq1_bucket"].isin(order)].copy()
b = b[["rq1_bucket","median"]].rename(columns={"median":"basiccov_median"})

summary = (rb.merge(pb, on="rq1_bucket").merge(b, on="rq1_bucket"))
summary = summary.round(4)
summary.to_csv(OUT_SUMMARY, index=False)
print("✅ Wrote", OUT_SUMMARY)
display(summary)

# --- MWU results all metrics together ---
parts = []
for fn in ["readability_mwu_rq1.csv", "passive_mwu_rq1_only.csv", "basiccov_mwu_rq1.csv"]:
    f = METRICS_DIR / fn
    if f.exists():
        df = pd.read_csv(f)
        parts.append(df)
    else:
        print("⚠️ Missing:", f)
mwu_all = pd.concat(parts, ignore_index=True)
mwu_all.to_csv(OUT_MWU_ALL, index=False)
print("✅ Wrote", OUT_MWU_ALL)
display(mwu_all.round(4))
