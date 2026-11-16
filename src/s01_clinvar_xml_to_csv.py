"""Stage 1: stream the full ClinVar XML release into a flat CSV (one row per VariationArchive)."""

import csv
import gzip

from lxml import etree

from src.config import CLINVAR_CSV, CLINVAR_XML_GZ

FIELDS = [
    "VariationID",
    "VariationName",
    "AlleleID",
    "Type",
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "Condition",
    "HGVS_c",
    "HGVS_p",
    "dbSNP_ID",
    "NumberOfSubmissions",
    "Comment",
    "Citations",
]


def extract_variation_data(variation_elem) -> dict:
    """Extract the fields in ``FIELDS`` from a ``VariationArchive`` element."""
    record = {
        "VariationID": variation_elem.get("VariationID", ""),
        "VariationName": variation_elem.get("VariationName", ""),
        "AlleleID": "",
        "Type": variation_elem.get("VariationType", ""),
        "GeneSymbol": "",
        "ClinicalSignificance": "",
        "ReviewStatus": "",
        "Condition": "",
        "HGVS_c": "",
        "HGVS_p": "",
        "dbSNP_ID": "",
        "NumberOfSubmissions": "",
        "Comment": [],
        "Citations": [],
    }

    simple_allele = variation_elem.find(".//SimpleAllele")
    if simple_allele is not None:
        record["AlleleID"] = simple_allele.get("AlleleID", "")

        gene_elem = simple_allele.find(".//Gene")
        if gene_elem is not None:
            record["GeneSymbol"] = gene_elem.get("Symbol", "")

        hgvs_coding = simple_allele.find('.//HGVS[@Type="coding"]/NucleotideExpression')
        if hgvs_coding is not None:
            record["HGVS_c"] = hgvs_coding.get("change", "")

        hgvs_protein = simple_allele.find('.//HGVS[@Type="coding"]/ProteinExpression')
        if hgvs_protein is not None:
            record["HGVS_p"] = hgvs_protein.get("change", "")

        dbsnp_xref = simple_allele.find('.//XRef[@DB="dbSNP"]')
        if dbsnp_xref is not None:
            record["dbSNP_ID"] = dbsnp_xref.get("ID", "")

    germline_class = variation_elem.find(".//Classifications/GermlineClassification")
    if germline_class is not None:
        desc_elem = germline_class.find("./Description")
        if desc_elem is not None:
            record["ClinicalSignificance"] = desc_elem.text or ""

        review_elem = germline_class.find("./ReviewStatus")
        if review_elem is not None:
            record["ReviewStatus"] = review_elem.text or ""

        record["NumberOfSubmissions"] = germline_class.get("NumberOfSubmissions", "") or ""

    classified_condition = variation_elem.find(".//ClassifiedCondition")
    if classified_condition is not None:
        record["Condition"] = classified_condition.text or ""

    for elem in variation_elem.findall('.//Attribute[@Type="Description"]'):
        if elem.text:
            record["Comment"].append(elem.text)
    for elem in variation_elem.findall(".//Comment"):
        if elem.text:
            record["Comment"].append(elem.text)

    pubmed_ids = {
        id_elem.text for id_elem in variation_elem.findall('.//Citation/ID[@Source="PubMed"]') if id_elem.text
    }
    record["Citations"] = sorted(pubmed_ids, key=int) if pubmed_ids else []

    return record


def iter_variation_records(xml_file):
    """Yield one record per ``VariationArchive``, clearing parsed elements to keep memory flat."""
    context = etree.iterparse(xml_file, events=("end",), tag="VariationArchive")
    for _, elem in context:
        yield extract_variation_data(elem)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def convert(in_path=CLINVAR_XML_GZ, out_path=CLINVAR_CSV) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(in_path, "rb") as f_in, open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDS)
        writer.writeheader()
        for record in iter_variation_records(f_in):
            writer.writerow(record)
            count += 1
            if count % 10_000 == 0:
                print(f"Processed {count} variations")
    print(f"Done. Total {count} variations written to {out_path}")
    return count


if __name__ == "__main__":
    convert()
