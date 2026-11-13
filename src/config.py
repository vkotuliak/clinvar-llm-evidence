"""Central place for file-system paths so every stage reads/writes the same locations.

All paths are relative to the repository root, so scripts work no matter which
directory they are launched from (run them as modules: ``python -m src.s01_...``).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = DATA_DIR / "benchmark"
AGGREGATED_DIR = DATA_DIR / "aggregated"
PUBMED_XML_DIR = DATA_DIR / "pubmed_xmls"
PAPERS_DIR = ROOT / "papers_by_doi"

# Full-XML pipeline (s01-s06)
CLINVAR_XML_GZ = RAW_DIR / "clinvar_dataset.xml.gz"
CLINVAR_CSV = DATA_DIR / "clinvar_csv_dataset.csv"
PUBMED_ABSTRACTS_CSV = DATA_DIR / "pubmed_abstracts.csv"
DOIS_TXT = DATA_DIR / "dois.txt"

# Lightweight tab-delimited ClinVar exports used by the benchmark
CLINVAR_FTP = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited"
VARIANT_SUMMARY = RAW_DIR / "variant_summary.txt.gz"
VAR_CITATIONS = RAW_DIR / "var_citations.txt"

RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
DOCS_FIGURES_DIR = ROOT / "docs" / "figures"
