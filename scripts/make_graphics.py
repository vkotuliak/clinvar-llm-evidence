"""Exploratory plots of the ClinVar CSV produced by stage 1 (saved to docs/figures/)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CLINVAR_CSV, DOCS_FIGURES_DIR  # noqa: E402


def barh(counts: pd.Series, title: str, filename: str, xlabel="Number of variants") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.sort_values().plot(kind="barh", ax=ax, color="#4269d0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLINVAR_CSV, usecols=[
        "ClinicalSignificance", "Type", "GeneSymbol", "ReviewStatus", "Condition", "NumberOfSubmissions"
    ])  # fmt: skip

    # A cell can hold several semicolon-separated significance values.
    significance = df["ClinicalSignificance"].dropna().str.split(";").explode().str.strip()
    barh(
        significance.value_counts().head(10),
        "Top 10 clinical significance values",
        "clinical_significance.png",
    )
    barh(df["Type"].value_counts(), "Variant types", "value_count_type.png")
    barh(df["GeneSymbol"].value_counts().head(20), "Top 20 genes", "gene_symbols.png")
    barh(df["ReviewStatus"].value_counts(), "Review status", "review_status.png")
    barh(df["Condition"].value_counts().head(30), "Top 30 conditions", "conditions_counts.png")

    submissions = pd.to_numeric(df["NumberOfSubmissions"], errors="coerce").dropna().astype(int)
    bins = [1, 2, 6, 11, 51, 101, 501, 1001, max(submissions.max() + 1, 1002)]
    labels = ["1", "2-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001+"]
    binned = pd.cut(submissions, bins=bins, labels=labels, right=False).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    binned.plot(kind="bar", ax=ax, color="#4269d0")
    ax.set_title("Number of submissions per variant")
    ax.set_xlabel("Submissions")
    ax.set_ylabel("Number of variants")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES_DIR / "number_of_submissions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
