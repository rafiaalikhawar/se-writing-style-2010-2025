# === BASIC 3k COVERAGE — PER BUCKET (run this cell once per bucket) ===
# Set which bucket to compute this run:
TARGET_BUCKET = "2020-2025"   # <-- change to "2015-2019", then "2020-2025" on later runs

from pathlib import Path
import re
from typing import Optional, Set, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy

BUCKET_TO_FOLDER = {
    "2010-2014": Path("data/cleaned/2010"),
    "2015-2019": Path("data/cleaned/2015"),
    "2020-2025": Path("data/cleaned/2020"),
}
BASIC_LIST = Path("data/resources/basic_english_3000.txt")
OUT_DIR = Path("data/metrics"); OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = OUT_DIR / "logs"; LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Per-bucket outputs
PER_PAPER_CSV = OUT_DIR / f"basiccov_per_paper_{TARGET_BUCKET.replace('-','_')}.csv"
SUMMARY_CSV   = OUT_DIR / f"basiccov_summary_{TARGET_BUCKET.replace('-','_')}.csv"
SHORT_LOG     = LOGS_DIR / f"too_short_basiccov_{TARGET_BUCKET.replace('-','_')}.log"
SHORT_LOG.write_text("", encoding="utf-8")

# ---- helpers ----
def find_year(path: Path) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(path))
    return int(m.group(0)) if m else None

def load_basic_list(path: Path) -> Set[str]:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            words.add(w)
    if not words:
        raise ValueError(f"No entries found in {path}")
    return words

def basic_coverage_from_doc(doc, basic: Set[str]) -> float:
    uniq = {t.lemma_.lower() for t in doc if t.is_alpha}
    return (len(uniq & basic) / len(uniq)) if uniq else float("nan")

# ---- load resources ----
root = BUCKET_TO_FOLDER.get(TARGET_BUCKET)
if not root or not root.exists():
    raise FileNotFoundError(f"Folder for {TARGET_BUCKET} not found: {root}")

txts = list(root.rglob("*.txt"))
if not txts:
    raise RuntimeError(f"No .txt files under: {root}")

BASIC = load_basic_list(BASIC_LIST)

# spaCy (small model is fine; restart kernel if model was just installed)
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

MIN_CHARS = 500

# ---- process this bucket only ----
rows: List[Dict] = []
for p in tqdm(txts, desc=f"Basic 3k coverage: {TARGET_BUCKET}"):
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) < MIN_CHARS:
            SHORT_LOG.write_text(SHORT_LOG.read_text() + str(p) + "\n", encoding="utf-8")
        doc = nlp(text)
        cov = basic_coverage_from_doc(doc, BASIC)
        rows.append({
            "paper_id": p.stem,
            "path": str(p),
            "year": find_year(p),
            "rq1_bucket": TARGET_BUCKET,
            "basic3000_coverage": float(cov),
        })
    except Exception as e:
        print(f"❌ Failed {p}: {e}")

df = pd.DataFrame(rows)
df.to_csv(PER_PAPER_CSV, index=False)

# quick summary for this bucket
summary = df["basic3000_coverage"].agg(n="count", mean="mean", median="median").to_frame().T.round(4)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"✅ Saved per-paper → {PER_PAPER_CSV}")
print(f"✅ Saved summary   → {SUMMARY_CSV}")
display(summary)
