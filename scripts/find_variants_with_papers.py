"""List variants for which every cited paper that has a DOI was downloaded as a PDF.

1. Take each variant's PubMed citations from the ClinVar CSV.
2. Look up the DOI of every citation in the PubMed abstracts CSV.
3. Keep variants where all citations with a DOI have a PDF in ``papers_by_doi/``
   and at least ``min_downloaded`` PDFs exist.
"""

import sys

import pandas as pd

from src.config import CLINVAR_CSV, PAPERS_DIR, PUBMED_ABSTRACTS_CSV
from src.s04_extract_pdfs import doi_to_pdf_path
from src.s05_data_preparation import safe_literal_eval


def find_downloaded_papers(min_downloaded: int = 4) -> list[int]:
    clinvar_df = pd.read_csv(CLINVAR_CSV, usecols=["VariationID", "Citations"])
    pubmed_df = pd.read_csv(PUBMED_ABSTRACTS_CSV, usecols=["PMID", "DOI"]).dropna(subset=["DOI"])
    pmid_to_doi = dict(zip(pubmed_df["PMID"].astype(int), pubmed_df["DOI"], strict=True))
    downloaded = {p.name for p in PAPERS_DIR.glob("*.pdf")}

    selected = []
    total = len(clinvar_df)
    for n, (var_id, citations) in enumerate(
        zip(clinvar_df["VariationID"], clinvar_df["Citations"], strict=True), 1
    ):
        if n % 100_000 == 0:
            print(f"processed {n}/{total}")
        pmids = [int(p) for p in safe_literal_eval(citations)]
        dois = [pmid_to_doi[p] for p in pmids if pmid_to_doi.get(p)]
        if not dois:
            continue
        n_downloaded = sum(doi_to_pdf_path(d).name in downloaded for d in dois)
        if n_downloaded == len(dois) and n_downloaded >= min_downloaded:
            selected.append(int(var_id))
    return selected


if __name__ == "__main__":
    min_dl = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    variants = find_downloaded_papers(min_dl)
    print(f"{len(variants)} variants with all papers downloaded: {variants[:50]}")
