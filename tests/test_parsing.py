from pathlib import Path

from src.config import DATA_DIR
from src.s01_clinvar_xml_to_csv import FIELDS, iter_variation_records
from src.s03_fetch_pubmed_abstracts import parse_pubmed_xml
from src.s04_extract_pdfs import doi_to_pdf_path, extract_pdf_content
from src.text_cleaning import clean_raw_text, group_broken_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"


def test_clinvar_xml_records():
    with open(DATA_DIR / "example.xml", "rb") as f:
        records = list(iter_variation_records(f))
    assert [r["VariationID"] for r in records] == ["69", "722", "286"]
    first = records[0]
    assert set(first) == set(FIELDS)
    assert first["GeneSymbol"] == "UROD"
    assert first["ClinicalSignificance"] == "Pathogenic"
    assert first["HGVS_p"] == "p.Arg292Gly"
    assert first["Citations"] == ["1634232", "24175354"]
    assert all(c.isdigit() for r in records for c in r["Citations"])


def test_pubmed_parsing_nested_markup_and_doi_fallback():
    recs = parse_pubmed_xml((FIXTURES / "pubmed_efetch.xml").read_bytes())
    assert [r["pmid"] for r in recs] == ["12345", "67890"]
    a, b = recs
    assert a["title"] == "A BRCA1 variant in two families"
    # text nested two levels deep (<b><sup>) must not be lost
    assert "81 carriers" in a["abstract"]
    assert a["abstract"].startswith("BACKGROUND: We studied BRCA1 c.68_69del")
    assert "disease & was absent" in a["abstract"]
    assert a["doi"] == "https://doi.org/10.1000/xyz123"
    assert b["abstract"] == ""
    assert b["doi"] == "https://doi.org/10.2000/abc"  # from ArticleIdList


def test_doi_to_pdf_path():
    assert (
        doi_to_pdf_path("https://doi.org/10.1194/jlr.P081620", Path("p")).name == "10.1194%2Fjlr.P081620.pdf"
    )


def test_missing_pdf_returns_none(tmp_path):
    assert extract_pdf_content(tmp_path / "missing.pdf") is None
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf")
    assert extract_pdf_content(corrupt) is None


def test_text_cleaning():
    raw = "This is a para-\ngraph broken\nacross lines.\n\n“Quoted” – text"
    grouped = group_broken_paragraphs(raw)
    assert grouped.split("\n\n")[0] == "This is a paragraph broken across lines."
    assert clean_raw_text(grouped).endswith('"Quoted" - text')
