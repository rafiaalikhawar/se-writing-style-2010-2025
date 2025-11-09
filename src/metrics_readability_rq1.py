# === Readability (FRE & FKGL) for 2010 / 2015 / 2020 ===
#          - readability_per_paper.csv
#          - readability_summary.csv
#          - readability_differences.csv
#          - figs/fre_rq1.(png|svg), figs/fkgl_rq1.(png|svg)

from pathlib import Path
import re
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import textstat

# -------- paths --------
BUCKETS = {
    "2010-2014": Path("data/cleaned/2010"),
    "2015-2019": Path("data/cleaned/2015"),
    "2020-2025": Path("data/cleaned/2020"),
}
OUT_DIR   = Path("data/metrics")
FIGS_DIR  = OUT_DIR / "figs"
LOGS_DIR  = OUT_DIR / "logs"
for p in [OUT_DIR, FIGS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

PER_PAPER_CSV = OUT_DIR / "readability_per_paper.csv"
SUMMARY_CSV   = OUT_DIR / "readability_summary.csv"
DIFFS_CSV     = OUT_DIR / "readability_differences.csv"
SHORT_LOG     = LOGS_DIR / "too_short_readability.log"
SHORT_LOG.write_text("", encoding="utf-8")
MIN_CHARS = 500

# -------- helpers --------
def find_year(path: Path) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(path))
    return int(m.group(0)) if m else None

def counts(text: str) -> Tuple[int,int,int]:
    words = textstat.lexicon_count(text, removepunct=True)
    sentences = textstat.sentence_count(text)
    syllables = textstat.syllable_count(text)
    return int(words), int(sentences), int(syllables)

def fre_fkgl(words: int, sentences: int, syllables: int) -> Tuple[float,float]:
    if words == 0 or sentences == 0:
        return np.nan, np.nan
    fre  = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    fkgl = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    return float(fre), float(fkgl)

# -------- compute per paper --------
rows: List[Dict] = []
for bucket, root in BUCKETS.items():
    if not root.exists():
        print(f"⚠️ Missing folder: {root} (skipping {bucket})")
        continue
    txts = list(root.rglob("*.txt"))
    if not txts:
        print(f"ℹ️ No .txt files under {root}")
        continue

    for p in tqdm(txts, desc=f"Readability {bucket}"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if len(text) < MIN_CHARS:
                with SHORT_LOG.open("a", encoding="utf-8") as f:
                    f.write(str(p) + "\n")

            w, s, syl = counts(text)
            fre, fk   = fre_fkgl(w, s, syl)

            rows.append({
                "paper_id": p.stem,
                "path": str(p),
                "year": find_year(p),
                "rq1_bucket": bucket,
                "words": w,
                "sentences": s,
                "syllables": syl,
                "fre": fre,
                "fkgl": fk,
            })
        except Exception as e:
            print(f"❌ Failed {p}: {e}")

df = pd.DataFrame(rows)
df.to_csv(PER_PAPER_CSV, index=False)
print(f"✅ Saved per-paper metrics → {PER_PAPER_CSV}")

# -------- summary per bucket (show on screen) --------
order = ["2010-2014","2015-2019","2020-2025"]
df["rq1_bucket"] = pd.Categorical(df["rq1_bucket"], categories=order, ordered=True)
summary = (
    df.groupby("rq1_bucket")[["fre","fkgl"]]
      .agg(n=("fre","count"),
           fre_mean=("fre","mean"), fre_median=("fre","median"),
           fkgl_mean=("fkgl","mean"), fkgl_median=("fkgl","median"))
      .round(2)
)
summary.to_csv(SUMMARY_CSV)
print(f"✅ Saved summary → {SUMMARY_CSV}")
print("\n=== Readability Summary (per bucket) ===")
display(summary)

# -------- differences (mean deltas) printed & saved --------
# FRE: higher = easier; FKGL: higher = harder
pairs = [("2010-2014","2015-2019"), ("2010-2014","2020-2025"), ("2015-2019","2020-2025")]
diff_rows = []
for g1, g2 in pairs:
    s1, s2 = summary.loc[g1], summary.loc[g2]
    diff_rows.append({
        "comparison": f"{g1} vs {g2}",
        "fre_mean_diff": float(s1["fre_mean"] - s2["fre_mean"]),   # + means g1 easier than g2
        "fkgl_mean_diff": float(s1["fkgl_mean"] - s2["fkgl_mean"]) # + means g1 harder grade than g2
    })
diffs = pd.DataFrame(diff_rows).round(2)
diffs.to_csv(DIFFS_CSV, index=False)
print(f"\n✅ Saved differences → {DIFFS_CSV}")
print("=== Mean Differences (g1 - g2) ===")
display(diffs)

# -------- boxplots (saved under data/metrics/figs) --------
def boxplot_metric(metric: str, title: str, fname_stub: str):
    data = [df.loc[df["rq1_bucket"]==b, metric].dropna().to_numpy() for b in order]
    if all(len(d)==0 for d in data):
        print(f"⚠️ No data for {metric}, skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    bp = ax.boxplot(data, labels=order, showmeans=False, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.grid(True, linestyle="--", alpha=0.4)
    # annotate medians & n
    for i, arr in enumerate(data, start=1):
        if len(arr):
            med = np.median(arr); n = len(arr)
            ax.text(i, bp["medians"][i-1].get_ydata()[0], f" med={med:.2f}\n n={n}", fontsize=9)
    for ext in ["png","svg"]:
        fig.savefig(FIGS_DIR / f"{fname_stub}.{ext}", bbox_inches="tight")
    plt.close(fig)

boxplot_metric("fre",  "Flesch Reading Ease by RQ1", "fre_rq1")
boxplot_metric("fkgl", "Flesch-Kincaid Grade by RQ1", "fkgl_rq1")
print(f"🎨 Figures saved in → {FIGS_DIR}")
print(f"📝 Short files logged at → {SHORT_LOG}")
