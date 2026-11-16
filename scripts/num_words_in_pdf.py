"""Sanity-check downloaded PDFs: report files that cannot be read or contain almost no text."""

import sys
from pathlib import Path

from src.config import PAPERS_DIR
from src.s04_extract_pdfs import extract_pdf_content


def process_first_n_papers(dir_path: Path = PAPERS_DIR, n: int = 1000, min_words: int = 10) -> dict:
    """Return ``{file name: word count}`` for the first ``n`` PDFs (``-1`` = unreadable)."""
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = sorted(f for f in p.iterdir() if f.suffix == ".pdf")[:n]

    results = {}
    for f in files:
        text = extract_pdf_content(f)
        results[f.name] = -1 if text is None else len(text.split())
        if 0 <= results[f.name] < min_words:
            print(f"{f.name} has fewer than {min_words} words (probably a scanned PDF).")
    bad = sum(v < min_words for v in results.values())
    print(f"{bad}/{len(files)} PDFs unreadable or nearly empty.")
    return results


if __name__ == "__main__":
    process_first_n_papers(n=int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
