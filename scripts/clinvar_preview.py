"""Print the first bytes of the ClinVar XML release to inspect its structure."""

import gzip
import sys

from src.config import CLINVAR_XML_GZ


def quick_preview(filename, n_bytes: int = 30_000) -> None:
    opener = gzip.open if str(filename).endswith(".gz") else open
    with opener(filename, "rt", encoding="utf-8") as f:
        print(f.read(n_bytes))


if __name__ == "__main__":
    quick_preview(sys.argv[1] if len(sys.argv) > 1 else CLINVAR_XML_GZ)
