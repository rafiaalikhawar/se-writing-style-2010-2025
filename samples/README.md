
We do not redistribute publisher PDFs. Place your own PDFs under data/raw/* locally.

cat > se-writing-style-2010-2025/src/utils.py <<'EOF'

from pathlib import Path
import re
def slugify(name: str, maxlen: int = 120) -> str:
    s = name.lower(); s = re.sub(r"[^\w\s-]+","",s); s = re.sub(r"\s+","-",s).strip("-"); s = re.sub(r"-{2,}","-",s); return s[:maxlen]
def find_year(p: Path):
    m = re.search(r"(19|20)\d{2}", str(p)); return int(m.group(0)) if m else None
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
