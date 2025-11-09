from pathlib import Path
import regex as re
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)

BASE = Path(__file__).resolve().parents[1]
INPUT_FOLDER = Path("data/raw/2020")
OUTPUT_FOLDER = Path("data/cleaned/2020")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
TOO_SHORT_LOG = OUTPUT_FOLDER.parent / "too_short.log"
MIN_CHARS = 500  # log anything shorter after cleaning

SECTION_RE = re.compile(r'\b(references|bibliography|acknowledgements|acknowledgments)\b', re.I)

def undo_hyphenation(text: str) -> str:
    # "word-\nword" -> "wordword"
    return re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)

def strip_bullets(line_block: str) -> str:
    # remove bullets at start of lines (•, *, -, –)
    return re.sub(r'(?m)^\s*[•\*\-\u2013]\s+', '', line_block)

def clean_text(raw: str) -> str:
    # Normalize unicode (curly quotes, NBSP, ligatures → ASCII-friendly forms)
    t = unicodedata.normalize('NFKC', raw)

    # Early: standardize newlines
    t = t.replace('\r\n', '\n').replace('\r', '\n')

    # Undo hyphenated line breaks before removing newlines
    t = undo_hyphenation(t)

    # Drop everything after references/bibliography/acknowledgements (heuristic)
    t = SECTION_RE.split(t)[0]

    # Strip bullets at line starts (common in lists)
    t = strip_bullets(t)

    # Lowercase
    t = t.lower()

    # Remove URLs/DOIs/arXiv-ish links
    t = re.sub(r'(https?://\S+|www\.\S+|doi:\S+|doi\s*\S+)', ' ', t)

    # Remove citation brackets like [12], [3,7], (Fig. 2), (Table 3) loosely
    t = re.sub(r'\[[^\]\n]{1,50}\]', ' ', t)  # square-bracket citations
    t = re.sub(r'\(fig\.\s*\d+[a-z]?\)|\(table\s*\d+[a-z]?\)', ' ', t, flags=re.I)

    # Remove standalone numbers but keep alphanumerics like "h2o" or "e2e"
    t = re.sub(r'(?<!\w)\d+(?!\w)', ' ', t)

    # Keep letters, digits embedded in words, spaces, and sentence enders . ! ?
    # Also keep brackets for sentence boundaries? We'll remove most symbols but preserve .?! explicitly.
    t = re.sub(r"[^a-z0-9\s\.\!\?]", " ", t)

    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t

converted, skipped = 0, 0
TOO_SHORT_LOG.write_text("", encoding="utf-8")

# Process recursively: **/*.txt
for txt_file in INPUT_FOLDER.rglob("*.txt"):
    raw = txt_file.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)

    # Tokenize after cleaning (punkt uses .?!)
    sentences = sent_tokenize(cleaned)
    words = word_tokenize(cleaned)

    # Save as sentence-per-line (good for later sentence-level analysis)
    out_path = OUTPUT_FOLDER / txt_file.name
    out_path.write_text("\n".join(sentences), encoding="utf-8")

    if len(cleaned) < MIN_CHARS:
        with TOO_SHORT_LOG.open("a", encoding="utf-8") as f:
            f.write(str(txt_file) + "\n")

    print(f"Processed: {txt_file.name} — {len(sentences)} sentences, {len(words)} words")
    converted += 1

print(f"\n✅ Cleaned {converted} files → {OUTPUT_FOLDER}")
print(f"📝 Short/possibly-bad files logged at: {TOO_SHORT_LOG}")
