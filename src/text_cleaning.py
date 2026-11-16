"""Light-weight cleaning of text extracted from PDFs (replaces the heavy ``unstructured`` dependency)."""

import re

_UNICODE_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
}  # fmt: skip
_DASHES = re.compile(r"[‐-―−]")
_HYPHENATED_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def group_broken_paragraphs(text: str) -> str:
    """Join lines that a PDF layout broke mid-paragraph; keep blank-line paragraph boundaries."""
    text = _HYPHENATED_LINEBREAK.sub(r"\1\2", text)
    paragraphs = (" ".join(p.split()) for p in _PARAGRAPH_SPLIT.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def clean_raw_text(text: str) -> str:
    """Normalise quotes, dashes and whitespace while preserving paragraph breaks."""
    text = text.translate(str.maketrans(_UNICODE_QUOTES))
    text = _DASHES.sub("-", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
