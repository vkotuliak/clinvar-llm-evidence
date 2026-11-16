"""Build the ClinVar literature-classification benchmark.

Task: given a variant (gene + HGVS name) and the abstracts of the PubMed papers ClinVar
submitters cited for it, predict its aggregate germline classification in 3 classes:
``P/LP`` (pathogenic / likely pathogenic), ``B/LB`` (benign / likely benign), ``VUS``.

Design choices (see README "Method"):
* High-confidence labels only (>=2 concordant submitters, expert panel or practice guideline).
* Generic citations (ACMG guidelines, population databases, ...) cited by more than
  ``GENERIC_PMID_THRESHOLD`` variants are removed - they say nothing about a specific variant.
* Gene-disjoint train/test split, so neither a baseline nor anything tuned on train can
  exploit a "gene -> label" shortcut.
* At most ``MAX_PER_GENE_TEST`` test variants per gene and class, so BRCA1/2 do not dominate.
* ClinVar free-text comments are NOT used as input: they frequently state the classification.
"""

import argparse
import json
import random
import subprocess
from collections import Counter

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import BENCHMARK_DIR, CLINVAR_FTP, RAW_DIR, VAR_CITATIONS, VARIANT_SUMMARY
from src.s03_fetch_pubmed_abstracts import create_session, fetch_abstracts

LABELS = ["P/LP", "VUS", "B/LB"]
LABEL_MAP = {
    "Pathogenic": "P/LP",
    "Likely pathogenic": "P/LP",
    "Pathogenic/Likely pathogenic": "P/LP",
    "Uncertain significance": "VUS",
    "Benign": "B/LB",
    "Likely benign": "B/LB",
    "Benign/Likely benign": "B/LB",
}
HIGH_CONFIDENCE_REVIEW = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}
GENERIC_PMID_THRESHOLD = 200
MAX_PMIDS = 8
MAX_PER_GENE_TEST = 3
N_TEST_PER_CLASS = 100
N_TRAIN_PER_CLASS = 1000
OVERSAMPLE = 1.3  # sample extra so we can drop variants whose papers have no abstract
SEED = 42

ABSTRACT_CACHE = BENCHMARK_DIR / "abstracts_cache.jsonl"


def download_raw() -> None:
    """Download the two ClinVar tab-delimited exports (~720 MB) if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in (VARIANT_SUMMARY, VAR_CITATIONS):
        if not path.exists():
            url = f"{CLINVAR_FTP}/{path.name}"
            print(f"Downloading {url}")
            subprocess.run(["curl", "-sSfL", "-o", str(path), url], check=True)


def map_label(clinical_significance: str) -> str | None:
    return LABEL_MAP.get(str(clinical_significance).strip())


def load_variants() -> pd.DataFrame:
    """High-confidence, single-gene germline variants with a 3-class label."""
    cols = ["VariationID", "Name", "Type", "GeneSymbol", "ClinicalSignificance",
            "ReviewStatus", "Assembly", "OriginSimple"]  # fmt: skip
    vs = pd.read_csv(VARIANT_SUMMARY, sep="\t", usecols=cols, low_memory=False)
    vs = vs[vs["Assembly"] == "GRCh38"].drop_duplicates("VariationID")
    vs = vs[vs["ReviewStatus"].isin(HIGH_CONFIDENCE_REVIEW)]
    vs = vs[vs["OriginSimple"] == "germline"]
    vs = vs[vs["GeneSymbol"].notna() & ~vs["GeneSymbol"].str.contains(r"[;,]|^-$", regex=True)]
    vs = vs.assign(label=vs["ClinicalSignificance"].map(map_label)).dropna(subset=["label"])
    return vs.drop(columns=["Assembly", "OriginSimple"])


def load_citations(threshold: int = GENERIC_PMID_THRESHOLD) -> pd.Series:
    """VariationID -> sorted list of variant-specific PubMed IDs."""
    cit = pd.read_csv(
        VAR_CITATIONS, sep="\t", usecols=["VariationID", "citation_source", "citation_id"], dtype=str
    )
    cit = cit[cit["citation_source"] == "PubMed"].drop_duplicates(["VariationID", "citation_id"])
    freq = cit["citation_id"].value_counts()
    generic = set(freq[freq > threshold].index)
    print(f"Dropping {len(generic)} generic PMIDs cited by >{threshold} variants")
    cit = cit[~cit["citation_id"].isin(generic) & cit["citation_id"].str.isdigit()]
    cit["VariationID"] = cit["VariationID"].astype(int)
    return cit.groupby("VariationID")["citation_id"].agg(lambda s: sorted(s, key=int))


def gene_disjoint_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = SEED):
    """Split rows so that no GeneSymbol appears on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["GeneSymbol"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def sample_per_class(df: pd.DataFrame, n: int, max_per_gene: int | None, seed: int) -> pd.DataFrame:
    """Up to ``n`` rows per label, optionally capping rows per (gene, label)."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    if max_per_gene is not None:
        shuffled = shuffled.groupby(["GeneSymbol", "label"], sort=False).head(max_per_gene)
    return shuffled.groupby("label", sort=False).head(n)


def load_abstract_cache() -> dict[str, dict]:
    if not ABSTRACT_CACHE.exists():
        return {}
    with ABSTRACT_CACHE.open() as f:
        return {r["pmid"]: r for r in map(json.loads, f)}


def get_abstracts(pmids: set[str]) -> dict[str, dict]:
    cache = load_abstract_cache()
    missing = sorted(pmids - cache.keys(), key=int)
    if missing:
        print(f"Fetching {len(missing)} abstracts from PubMed ({len(pmids) - len(missing)} cached)")
        records = fetch_abstracts(missing, create_session(), xml_dir=None)
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with ABSTRACT_CACHE.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                cache[r["pmid"]] = r
    return cache


def to_records(df: pd.DataFrame, abstracts: dict[str, dict], n_per_class: int) -> list[dict]:
    """Attach abstracts; drop variants with no usable abstract; keep ``n_per_class`` per label."""
    out, per_label = [], Counter()
    for row in df.itertuples(index=False):
        if per_label[row.label] >= n_per_class:
            continue
        papers = [
            {"pmid": p, "title": abstracts[p]["title"], "abstract": abstracts[p]["abstract"]}
            for p in row.pmids
            if p in abstracts and abstracts[p]["abstract"]
        ]
        if not papers:
            continue
        per_label[row.label] += 1
        out.append({
            "variation_id": int(row.VariationID),
            "gene": row.GeneSymbol,
            "name": row.Name,
            "type": row.Type,
            "clinical_significance": row.ClinicalSignificance,
            "review_status": row.ReviewStatus,
            "label": row.label,
            "papers": papers,
        })  # fmt: skip
    return out


def write_jsonl(records: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(n_test: int = N_TEST_PER_CLASS, n_train: int = N_TRAIN_PER_CLASS, seed: int = SEED) -> None:
    download_raw()
    variants = load_variants()
    citations = load_citations()
    variants = variants.assign(pmids=variants["VariationID"].map(citations)).dropna(subset=["pmids"])
    variants = variants[variants["pmids"].map(len).between(1, MAX_PMIDS)]
    print(f"{len(variants)} candidate variants:", variants["label"].value_counts().to_dict())

    train_pool, test_pool = gene_disjoint_split(variants, seed=seed)
    test_df = sample_per_class(test_pool, int(n_test * OVERSAMPLE), MAX_PER_GENE_TEST, seed)
    train_df = sample_per_class(train_pool, int(n_train * OVERSAMPLE), None, seed)

    pmids = {p for lst in pd.concat([test_df, train_df])["pmids"] for p in lst}
    abstracts = get_abstracts(pmids)

    test = to_records(test_df, abstracts, n_test)
    train = to_records(train_df, abstracts, n_train)
    random.Random(seed).shuffle(test)
    random.Random(seed).shuffle(train)

    overlap = {r["gene"] for r in test} & {r["gene"] for r in train}
    assert not overlap, f"gene leakage between splits: {sorted(overlap)[:10]}"

    write_jsonl(test, BENCHMARK_DIR / "test.jsonl")
    write_jsonl(train, BENCHMARK_DIR / "train.jsonl")
    for name, recs in (("train", train), ("test", test)):
        n_papers = [len(r["papers"]) for r in recs]
        print(
            f"{name}: {len(recs)} variants, {len({r['gene'] for r in recs})} genes, "
            f"labels {dict(Counter(r['label'] for r in recs))}, "
            f"papers/variant mean {sum(n_papers) / len(n_papers):.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ClinVar literature benchmark.")
    parser.add_argument("--n-test", type=int, default=N_TEST_PER_CLASS)
    parser.add_argument("--n-train", type=int, default=N_TRAIN_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(args.n_test, args.n_train, args.seed)
