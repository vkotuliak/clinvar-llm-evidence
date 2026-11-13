"""Stage 3: fetch title, DOI and abstract for every PubMed ID cited in ClinVar (NCBI E-utilities)."""

import ast
import csv
import html
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import CLINVAR_CSV, PUBMED_ABSTRACTS_CSV, PUBMED_XML_DIR

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "clinvar-llm-evidence"
BATCH_SIZE = 200
DELAY_S = 0.34  # NCBI allows 3 requests/s without an API key


def collect_unique_pmids(csv_path=CLINVAR_CSV) -> list[str]:
    """Collect unique PubMed IDs from the ``Citations`` column of the ClinVar CSV."""
    all_pmids: set[str] = set()
    df = pd.read_csv(csv_path, usecols=["Citations"])

    for cit in df["Citations"].dropna():
        try:
            pmids = ast.literal_eval(cit)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing citation: {cit}, Error: {e}")
            continue
        all_pmids.update(str(p) for p in pmids if str(p).isdigit())

    print(f"Collected {len(all_pmids)} unique PMIDs")
    return sorted(all_pmids, key=int)


def create_session() -> requests.Session:
    """Create an HTTP session that retries on rate-limit and server errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def extract_text_content(element) -> str:
    """All text inside an element (including arbitrarily nested markup), whitespace-normalised."""
    text = html.unescape("".join(element.itertext()))
    return " ".join(text.split())


def _find_doi(article) -> str:
    doi = article.findtext(".//ELocationID[@EIdType='doi']") or article.findtext(
        ".//ArticleIdList/ArticleId[@IdType='doi']"
    )
    if not doi:
        return ""
    doi = doi.strip()
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def parse_pubmed_xml(xml_content: str | bytes) -> list[dict]:
    """Parse an efetch ``PubmedArticleSet`` into ``{pmid, doi, title, abstract}`` dicts."""
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")
    root = ET.fromstring(xml_content)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID") or article.findtext(".//PMID")
        title_elem = article.find(".//ArticleTitle")
        title = extract_text_content(title_elem) if title_elem is not None else ""

        parts = []
        for el in article.findall(".//Abstract/AbstractText"):
            text = extract_text_content(el)
            if not text:
                continue
            label = el.get("Label")
            parts.append(f"{label}: {text}" if label else text)

        records.append(
            {"pmid": str(pmid), "doi": _find_doi(article), "title": title, "abstract": " ".join(parts)}
        )
    return records


def load_abstracts(
    pmids: list,
    session: requests.Session,
    batch_number: int = 0,
    xml_dir: Path | None = PUBMED_XML_DIR,
) -> list[dict]:
    """Fetch one batch of PMIDs from PubMed. Optionally keeps the raw XML in ``xml_dir``."""
    valid_pmids = [str(p) for p in pmids if p and str(p).isdigit()]
    if not valid_pmids:
        return []

    params = {"db": "pubmed", "id": ",".join(valid_pmids), "retmode": "xml", "tool": TOOL_NAME}
    try:
        response = session.post(EFETCH_URL, data=params, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for batch {batch_number}: {e}")
        return []

    if xml_dir is not None:
        xml_dir.mkdir(parents=True, exist_ok=True)
        name = f"batch_{batch_number:04d}_{valid_pmids[0]}_to_{valid_pmids[-1]}.xml"
        (xml_dir / name).write_bytes(response.content)

    try:
        records = parse_pubmed_xml(response.content)
    except ET.ParseError as e:
        print(f"XML parsing failed for batch {batch_number}: {e}")
        return []

    missing = set(valid_pmids) - {r["pmid"] for r in records}
    if missing:
        print(f"Batch {batch_number}: {len(missing)} PMIDs not returned by PubMed")
    return records


def fetch_abstracts(
    pmids: list,
    session: requests.Session | None = None,
    batch_size: int = BATCH_SIZE,
    delay: float = DELAY_S,
    xml_dir: Path | None = PUBMED_XML_DIR,
) -> list[dict]:
    """Fetch any number of PMIDs politely, in batches."""
    session = session or create_session()
    pmids = list(pmids)
    out: list[dict] = []
    n_batches = (len(pmids) + batch_size - 1) // batch_size
    for i in range(0, len(pmids), batch_size):
        batch_number = i // batch_size + 1
        out.extend(load_abstracts(pmids[i : i + batch_size], session, batch_number, xml_dir))
        if batch_number % 10 == 0 or batch_number == n_batches:
            print(f"Fetched batch {batch_number}/{n_batches}, {len(out)} records so far")
        if i + batch_size < len(pmids):
            time.sleep(delay)
    return out


def all_abstracts_in_csv(in_path=CLINVAR_CSV, out_path=PUBMED_ABSTRACTS_CSV) -> None:
    """Fetch abstracts for every PMID cited in the ClinVar CSV and write them to one CSV."""
    unique_pmids = collect_unique_pmids(in_path)
    if not unique_pmids:
        print("No PMIDs found to process")
        return

    records = fetch_abstracts(unique_pmids)
    if not records:
        print("No data to write")
        return

    df = pd.DataFrame(records).rename(
        columns={"pmid": "PMID", "doi": "DOI", "title": "Title", "abstract": "Abstract"}
    )
    df.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL, escapechar="\\")
    print(f"Wrote {len(df)} records to {out_path} ({(df['Abstract'] == '').sum()} without abstract)")


if __name__ == "__main__":
    all_abstracts_in_csv()
