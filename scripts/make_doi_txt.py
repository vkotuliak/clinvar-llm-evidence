"""Write one bare DOI per line (input for PyPaperBot) from the PubMed abstracts CSV."""

import pandas as pd

from src.config import DOIS_TXT, PUBMED_ABSTRACTS_CSV


def main() -> None:
    dois = pd.read_csv(PUBMED_ABSTRACTS_CSV, usecols=["DOI"])["DOI"].dropna()
    dois = dois.astype(str).str.replace("https://doi.org/", "", n=1, regex=False).drop_duplicates()
    DOIS_TXT.write_text("\n".join(dois) + "\n")
    print(f"Wrote {len(dois)} DOIs to {DOIS_TXT}")


if __name__ == "__main__":
    main()
