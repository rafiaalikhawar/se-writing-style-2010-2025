# === Passive Voice Ratio (RQ1: 2010–2014, 2015–2019, 20205) ===

#   - passive_per_paper.csv
#   - passive_summary.csv
#   - passive_mwu_rq1.csv
#   - passive_differences.csv
#   - figs/passive_rq1.(png|svg)
from pathlib import Path
import re
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy
from scipy.stats import mannwhitneyu

# ---------- paths ----------
BUCKETS = {
    "2010-2014": Path("data/cleaned/2010"),
    "2015-2019": Path("data/cleaned/2015"),
    "2020-2025": Path("data/cleaned/2020"),
}
OUT_DIR  = Path("data/metrics")
FIGS_DIR = OUT_DIR / "figs"
LOGS_DIR = OUT_DIR / "logs"
for p in [OUT_DIR, FIGS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

PER_PAPER_CSV = OUT_DIR / "passive_per_paper.csv"
SUMMARY_CSV   = OUT_DIR / "passive_summary.csv"
DIFFS_CSV     = OUT_DIR / "passive_differences.csv"
MWU_CSV       = OUT_DIR / "passive_mwu_rq1.csv"
SHORT_LOG     = LOGS_DIR / "too_short_passive.log"
SHORT_LOG.write_text("", encoding="utf-8")

MIN_CHARS = 500  # log short files

# ---------- helpers ----------
def find_year(path: Path) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(path))
    return int(m.group(0)) if m else None

def sentence_is_passive(sent) -> bool:
    """
    Passive if:
      - any token has dep_ == 'nsubjpass', OR
      - (any aux/auxpass with lemma 'be') AND (any token with tag_ == 'VBN')
    """
    has_nsubjpass = any(t.dep_ == "nsubjpass" for t in sent)
    if has_nsubjpass:
        return True
    has_be_aux = any((t.dep_ in ("aux","auxpass")) and (t.lemma_ == "be") for t in sent)
    has_vbn    = any(t.tag_ == "VBN" for t in sent)
    return bool(has_be_aux and has_vbn)

def holm_correction(pvals: List[float]) -> List[float]:
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

# Load spaCy (parser on for sents; disable NER for speed)
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ---------- compute per paper ----------
rows: List[Dict] = []
for bucket, root in BUCKETS.items():
    if not root.exists():
        print(f"⚠️ Missing folder: {root} (skipping {bucket})")
        continue
    txts = list(root.rglob("*.txt"))
    if not txts:
        print(f"ℹ️ No .txt files under {root}")
        continue

    for p in tqdm(txts, desc=f"Passive ratio {bucket}"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if len(text) < MIN_CHARS:
                with SHORT_LOG.open("a", encoding="utf-8") as f:
                    f.write(str(p) + "\n")

            doc = nlp(text)
            sents = list(doc.sents)
            total_sents = len(sents)
            passive_ratio = np.nan
            if total_sents > 0:
                flags = [sentence_is_passive(s) for s in sents]
                passive_ratio = float(sum(flags) / total_sents)

            rows.append({
                "paper_id": p.stem,
                "path": str(p),
                "year": find_year(p),
                "rq1_bucket": bucket,
                "passive_ratio": passive_ratio,
                "total_sentences": total_sents,
            })
        except Exception as e:
            print(f"❌ Failed {p}: {e}")

df = pd.DataFrame(rows)
df.to_csv(PER_PAPER_CSV, index=False)
print(f"✅ Saved per-paper passive ratios → {PER_PAPER_CSV}")

# ---------- summary per bucket (shown on screen) ----------
order = ["2010-2014","2015-2019","2020-2025"]
df["rq1_bucket"] = pd.Categorical(df["rq1_bucket"], categories=order, ordered=True)
summary = (
    df.groupby("rq1_bucket")[["passive_ratio"]]
      .agg(n=("passive_ratio","count"),
           mean=("passive_ratio","mean"),
           median=("passive_ratio","median"))
      .round(4)
)
summary.to_csv(SUMMARY_CSV, index=False)
print(f"✅ Saved summary → {SUMMARY_CSV}")
print("\n=== Passive Voice Ratio (per bucket) ===")
display(summary)

# ---------- differences (means) ----------
pairs = [("2010-2014","2015-2019"), ("2010-2014","2020-2025"), ("2015-2019","2020-2025")]
diff_rows = []
for g1, g2 in pairs:
    s1, s2 = summary.set_index("rq1_bucket").loc[g1], summary.set_index("rq1_bucket").loc[g2]
    diff_rows.append({
        "comparison": f"{g1} vs {g2}",
        "mean_diff": float(s1["mean"] - s2["mean"]),       # + means g1 has higher passive ratio
        "median_diff": float(s1["median"] - s2["median"]),
    })
diffs = pd.DataFrame(diff_rows).round(4)
diffs.to_csv(DIFFS_CSV, index=False)
print(f"\n✅ Saved differences → {DIFFS_CSV}")
print("=== Mean/Median Differences (g1 - g2) ===")
display(diffs)

# ---------- Mann–Whitney U with Holm correction ----------
rows_stats = []
for metric in ["passive_ratio"]:
    pvals = []
    tmp = []
    for g1, g2 in pairs:
        A = df.loc[df["rq1_bucket"]==g1, metric].dropna().to_numpy()
        B = df.loc[df["rq1_bucket"]==g2, metric].dropna().to_numpy()
        if len(A)==0 or len(B)==0:
            tmp.append((g1,g2,np.nan,np.nan,np.nan,len(A),len(B),np.nan,np.nan))
            pvals.append(np.nan)
            continue
        U, p = mannwhitneyu(A, B, alternative="two-sided")
        rbc = 1.0 - 2.0 * U / (len(A)*len(B))  # rank-biserial effect size
        medA, medB = float(np.median(A)), float(np.median(B))
        tmp.append((g1,g2,float(U),float(p),float(rbc),len(A),len(B),medA,medB))
        pvals.append(float(p))
    # Holm within this metric
    valid_idx = [i for i,p in enumerate(pvals) if not np.isnan(p)]
    adj_all = [np.nan]*len(pvals)
    if valid_idx:
        adj_vals = holm_correction([pvals[i] for i in valid_idx])
        for i, adj in zip(valid_idx, adj_vals):
            adj_all[i] = adj
    for (g1,g2,U,p,rbc,nA,nB,medA,medB), p_holm in zip(tmp, adj_all):
        rows_stats.append({
            "metric": metric,
            "group_A": g1, "group_B": g2,
            "n_A": nA, "n_B": nB,
            "median_A": medA, "median_B": medB,
            "U": U, "p": p, "p_holm": p_holm,
            "effect_size_rbc": rbc
        })

mwu_df = pd.DataFrame(rows_stats)
mwu_df.to_csv(MWU_CSV, index=False)
print(f"\n✅ Saved MWU results → {MWU_CSV}")
print("=== Mann–Whitney U (two-sided), Holm-adjusted p-values ===")
display(mwu_df.round(4))

# ---------- boxplot ----------
def boxplot_passive(df, order, title, fname_stub):
    data = [df.loc[df["rq1_bucket"]==b, "passive_ratio"].dropna().to_numpy() for b in order]
    if all(len(d)==0 for d in data):
        print("⚠️ No passive_ratio data, skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    bp = ax.boxplot(data, labels=order, showmeans=False, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Passive Voice Ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    # annotate medians & n
    for i, arr in enumerate(data, start=1):
        if len(arr):
            med = np.median(arr); n = len(arr)
            ax.text(i, bp["medians"][i-1].get_ydata()[0], f" med={med:.2f}\n n={n}", fontsize=9)
    for ext in ["png","svg"]:
        fig.savefig(FIGS_DIR / f"{fname_stub}.{ext}", bbox_inches="tight")
    plt.close(fig)

boxplot_passive(df, order, "Passive Voice Ratio by RQ1", "passive_rq1")
print(f"🎨 Figure saved in → {FIGS_DIR}")
print(f"📝 Short files logged at → {SHORT_LOG}")
