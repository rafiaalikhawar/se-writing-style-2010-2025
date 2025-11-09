# === Passive Voice Ratio: Mann–Whitney U (RQ1) ===

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

METRICS_DIR = Path("data/metrics")
IN_CSV  = METRICS_DIR / "passive_per_paper.csv"
OUT_CSV = METRICS_DIR / "passive_mwu_rq1_only.csv"

# Load the per-paper passive ratios
df = pd.read_csv(IN_CSV)

# Ensure the three buckets are present and ordered
order = ["2010-2014", "2015-2019", "2020-2025"]
df = df[df["rq1_bucket"].isin(order)].copy()
df["rq1_bucket"] = pd.Categorical(df["rq1_bucket"], categories=order, ordered=True)

# Pairwise comparisons: A vs B, A vs C, B vs C
pairs = [
    ("2010-2014", "2015-2019"),
    ("2010-2014", "2020-2025"),
    ("2015-2019", "2020-2025"),
]

def holm_correction(pvals):
    """Holm step-down adjusted p-values (returns in original order)."""
    m = len(pvals)
    order_idx = np.argsort(pvals)
    p_sorted = np.array(pvals)[order_idx]
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i in range(m):
        adj_i = (m - i) * p_sorted[i]
        running = max(running, adj_i)
        adj[order_idx[i]] = min(1.0, running)
    return adj.tolist()

rows = []
pvals = []
tmp   = []

for g1, g2 in pairs:
    A = df.loc[df["rq1_bucket"]==g1, "passive_ratio"].dropna().to_numpy()
    B = df.loc[df["rq1_bucket"]==g2, "passive_ratio"].dropna().to_numpy()
    if len(A)==0 or len(B)==0:
        tmp.append((g1, g2, np.nan, np.nan, np.nan, len(A), len(B), np.nan, np.nan))
        pvals.append(np.nan)
        continue
    U, p = mannwhitneyu(A, B, alternative="two-sided")
    # Rank-biserial effect size (direction: group_A vs group_B)
    rbc = 1.0 - 2.0 * U / (len(A)*len(B))
    medA, medB = float(np.median(A)), float(np.median(B))
    tmp.append((g1, g2, float(U), float(p), float(rbc), len(A), len(B), medA, medB))
    pvals.append(float(p))

# Holm adjust across the three pairwise tests
valid_idx = [i for i,p in enumerate(pvals) if not np.isnan(p)]
adj_all = [np.nan]*len(pvals)
if valid_idx:
    adj_vals = holm_correction([pvals[i] for i in valid_idx])
    for i, adj in zip(valid_idx, adj_vals):
        adj_all[i] = adj

for (g1, g2, U, p, rbc, nA, nB, medA, medB), p_holm in zip(tmp, adj_all):
    rows.append({
        "metric": "passive_ratio",
        "group_A": g1, "group_B": g2,
        "n_A": nA, "n_B": nB,
        "median_A": medA, "median_B": medB,
        "U": U, "p": p, "p_holm": p_holm,
        "effect_size_rbc": rbc
    })

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)

print("✅ Wrote:", OUT_CSV)
display(res.round(4))

print("\nInterpretation tips:")
print("• effect_size_rbc > 0 → group_A has higher passive_ratio than group_B.")
print("• Use p_holm for significance after multiple comparisons.")
