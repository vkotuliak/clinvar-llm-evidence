"""Shared helpers: loading the benchmark, prompt construction, prediction files."""

import json
import re
from pathlib import Path

from benchmark.build_benchmark import LABELS
from src.config import BENCHMARK_DIR, PREDICTIONS_DIR

INVALID = "INVALID"


def load_split(split: str) -> list[dict]:
    with (BENCHMARK_DIR / f"{split}.jsonl").open() as f:
        return [json.loads(line) for line in f]


def variant_header(rec: dict) -> str:
    return f"Variant: {rec['name']}\nGene: {rec['gene']}\nVariant type: {rec['type']}"


def evidence_text(rec: dict, max_chars: int | None = None) -> str:
    """Titles + abstracts of the cited papers; each paper gets an equal share of ``max_chars``."""
    per_paper = max_chars // len(rec["papers"]) if max_chars else None
    blocks = []
    for i, p in enumerate(rec["papers"], start=1):
        abstract = p["abstract"]
        if per_paper and len(abstract) > per_paper:
            abstract = abstract[:per_paper].rsplit(" ", 1)[0] + " [...]"
        blocks.append(f"[{i}] PMID {p['pmid']}: {p['title']}\n{abstract}")
    return "\n\n".join(blocks)


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")


def predictions_path(name: str) -> Path:
    return PREDICTIONS_DIR / f"{model_slug(name)}.jsonl"


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_predictions(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_label(value) -> str:
    """Map a free-form model answer onto one of ``LABELS`` (or ``INVALID``)."""
    if not isinstance(value, str):
        return INVALID
    v = value.strip().upper().replace(" ", "")
    if v in {l.upper() for l in LABELS}:
        return next(l for l in LABELS if l.upper() == v)
    if "UNCERTAIN" in v or v.startswith("VUS"):
        return "VUS"
    if "PATHOGENIC" in v or v in {"P", "LP"}:
        return "P/LP"
    if "BENIGN" in v or v in {"B", "LB"}:
        return "B/LB"
    return INVALID
