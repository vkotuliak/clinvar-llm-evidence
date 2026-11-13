"""Non-LLM baselines trained on the (gene-disjoint) train split.

* ``majority``        - always predicts the most frequent train label (chance level here).
* ``tfidf-name-only`` - char n-grams of the HGVS variant name only (e.g. "Ter", "fs", "p.=").
* ``tfidf-evidence``  - word 1-2-grams of variant name + titles + abstracts (same input as the LLM).
"""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from benchmark.common import evidence_text, load_split, predictions_path, write_predictions

SEED = 42


def name_text(rec: dict) -> str:
    return rec["name"]


def full_text(rec: dict) -> str:
    return f"{rec['name']}\n{evidence_text(rec)}"


def tfidf_model(analyzer: str, ngram_range: tuple[int, int]):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def run() -> None:
    train, test = load_split("train"), load_split("test")
    y_train = [r["label"] for r in train]

    def save(name, preds):
        write_predictions(
            [{"variation_id": r["variation_id"], "label": r["label"], "pred": p}
             for r, p in zip(test, preds, strict=True)],
            predictions_path(name),
        )  # fmt: skip
        acc = sum(p == r["label"] for r, p in zip(test, preds, strict=True)) / len(test)
        print(f"{name:>16}: accuracy {acc:.3f}")

    majority = Counter(y_train).most_common(1)[0][0]
    save("majority", [majority] * len(test))

    for name, featurize, model in [
        ("tfidf-name-only", name_text, tfidf_model("char_wb", (2, 5))),
        ("tfidf-evidence", full_text, tfidf_model("word", (1, 2))),
    ]:
        model.fit([featurize(r) for r in train], y_train)
        save(name, model.predict([featurize(r) for r in test]).tolist())


if __name__ == "__main__":
    run()
