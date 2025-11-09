import re
from pathlib import Path
from PyPDF2 import PdfReader

def _slugify(name: str, maxlen: int = 120) -> str:
    s = name.lower()
    s = re.sub(r"[^\w\s-]+", "", s)      # remove punctuation except _ and -
    s = re.sub(r"\s+", "-", s).strip("-")# spaces -> dashes
    s = re.sub(r"-{2,}", "-", s)         # collapse dashes
    return s[:maxlen]

def _find_year(p: Path):
    m = re.search(r"(19|20)\d{2}", str(p))
    return int(m.group(0)) if m else None

def convert_pdfs_to_txt(folder_path: str,
                        dest_root: str = "data/raw",
                        recursive: bool = True,
                        overwrite: bool = False):
    """
    Convert PDFs in folder_path to TXT using PyPDF2.

    Saves to: DEST/<year>/<year>_<slug>.txt
    - year is inferred from any 4-digit 19xx/20xx in the path or filename.
    - if not found, falls back to DEST/unknown_year/<slug>.txt
    """
    src = Path(folder_path).expanduser().resolve()
    dest = Path(dest_root).expanduser().resolve()
    pdf_iter = src.rglob("*.pdf") if recursive else src.glob("*.pdf")

    found = False
    for pdf_file in pdf_iter:
        found = True
        year = _find_year(pdf_file) or "unknown_year"
        slug = _slugify(pdf_file.stem)
        out_dir = dest / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = out_dir / f"{year}_{slug}.txt" if isinstance(year, int) else out_dir / f"{slug}.txt"

        if out_txt.exists() and not overwrite:
            print(f"Skipping (exists): {out_txt}")
            continue

        try:
            reader = PdfReader(str(pdf_file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            out_txt.write_text(text, encoding="utf-8")
            print(f"Converted: {pdf_file}  →  {out_txt}")
        except Exception as e:
            print(f"Failed to convert {pdf_file}: {e}")

    if not found:
        print("No PDF files found.")

# 🟢 Change this to your bucket path
folder_path = "research/buckets/2020-2025"

# Example run (recursive scan, write to data/raw/<year>/...)
convert_pdfs_to_txt(folder_path, dest_root="data/raw", recursive=True, overwrite=False)
