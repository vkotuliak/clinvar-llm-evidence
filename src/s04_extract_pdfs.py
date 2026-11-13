"""Stage 4: extract plain text from downloaded full-text PDFs."""

from pathlib import Path

import pymupdf

from src.config import PAPERS_DIR


def doi_to_pdf_path(doi: str, pdf_dir: Path = PAPERS_DIR) -> Path:
    """Map a DOI (optionally as a doi.org URL) to the file name PyPaperBot uses with --use-doi-as-filename."""
    bare = str(doi).replace("https://doi.org/", "").strip()
    return Path(pdf_dir) / (bare.replace("/", "%2F") + ".pdf")


def extract_pdf_content(file) -> str | None:
    """Return the text of a PDF, or ``None`` if the file is missing or cannot be parsed."""
    try:
        with pymupdf.open(file) as pdf_file:
            return "".join(page.get_text() for page in pdf_file)
    except (FileNotFoundError, pymupdf.FileNotFoundError, pymupdf.FileDataError, RuntimeError) as e:
        print(f"Could not read {file}: {e}")
        return None


if __name__ == "__main__":
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    readable = sum(extract_pdf_content(p) is not None for p in pdfs)
    print(f"{readable}/{len(pdfs)} PDFs in {PAPERS_DIR} could be read.")
