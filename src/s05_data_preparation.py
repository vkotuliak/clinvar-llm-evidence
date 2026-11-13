"""Stage 5: build one text document per variant (ClinVar comments + abstract or full text per cited paper).

Papers are separated by ``PAPER_SEPARATOR`` so the LLM stage can process them one by one.
"""

import ast
from functools import lru_cache

import pandas as pd

from src.config import AGGREGATED_DIR, CLINVAR_CSV, PUBMED_ABSTRACTS_CSV
from src.s04_extract_pdfs import doi_to_pdf_path, extract_pdf_content
from src.text_cleaning import clean_raw_text, group_broken_paragraphs

PAPER_SEPARATOR = "#####"


def safe_literal_eval(s) -> list:
    """Parse a stringified Python list from the CSV; NaN or malformed values become ``[]``."""
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


@lru_cache(maxsize=1)
def _clinvar_index() -> pd.DataFrame:
    """Load the (large) ClinVar CSV once and index it by VariationID."""
    df = pd.read_csv(CLINVAR_CSV, usecols=["VariationID", "Comment", "Citations"])
    return df.set_index("VariationID")


@lru_cache(maxsize=1)
def _abstracts_index() -> pd.DataFrame:
    df = pd.read_csv(PUBMED_ABSTRACTS_CSV, dtype={"PMID": int})
    return df.drop_duplicates("PMID").set_index("PMID")


def paper_text(pmid: int, abstracts: pd.DataFrame) -> str:
    """Full text of the paper if its PDF is available and readable, otherwise its abstract."""
    if pmid not in abstracts.index:
        return "ABSTRACT:\nNot available\n"
    row = abstracts.loc[pmid]

    doi = row.get("DOI")
    if isinstance(doi, str) and doi:
        pdf_content = extract_pdf_content(doi_to_pdf_path(doi))
        if pdf_content and pdf_content.strip():
            return "FULL PAPER TEXT:\n" + clean_raw_text(group_broken_paragraphs(pdf_content)) + "\n"

    abstract = row.get("Abstract")
    abstract = abstract if isinstance(abstract, str) and abstract.strip() else "Not available"
    return "ABSTRACT:\n" + abstract + "\n"


def aggregate_data(variation_id: int, save: bool = True) -> str:
    """Aggregate ClinVar comments and the text of every cited paper for one variant."""
    clinvar = _clinvar_index()
    if variation_id not in clinvar.index:
        raise KeyError(f"VariationID {variation_id} not found in {CLINVAR_CSV}")
    row = clinvar.loc[variation_id]

    comments = safe_literal_eval(row["Comment"])
    parts = ["COMMENT:", ". ".join(comments) if comments else "Not Provided", "---"]

    abstracts = _abstracts_index()
    for i, pmid in enumerate(safe_literal_eval(row["Citations"]), start=1):
        parts.append(f"{PAPER_SEPARATOR}\nPAPER {i}\n{paper_text(int(pmid), abstracts)}")

    outcome = "\n".join(parts) + "\n"
    if save:
        AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
        (AGGREGATED_DIR / f"aggregated_data_{variation_id}.txt").write_text(outcome, encoding="utf-8")
    return outcome


if __name__ == "__main__":
    import sys

    vid = int(sys.argv[1]) if len(sys.argv) > 1 else 3254
    print(aggregate_data(vid))
