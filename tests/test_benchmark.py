import numpy as np
import pandas as pd
import pytest

from benchmark.build_benchmark import gene_disjoint_split, map_label, sample_per_class
from benchmark.common import INVALID, evidence_text, normalize_label
from benchmark.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar, paired_bootstrap, summarize
from experiments.pilot_participant_count import filter_answer
from src.s06_LLM_analysis import parse_json_response, split_papers


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("Pathogenic", "P/LP"),
        ("Pathogenic/Likely pathogenic", "P/LP"),
        ("Likely benign", "B/LB"),
        ("Uncertain significance", "VUS"),
        ("Conflicting classifications of pathogenicity", None),
        ("Pathogenic; drug response", None),
    ],
)
def test_map_label(raw, label):
    assert map_label(raw) == label


def _toy_variants(n_genes=40, per_gene=6):
    rng = np.random.default_rng(0)
    rows = [
        {"VariationID": g * 100 + i, "GeneSymbol": f"G{g}", "label": rng.choice(["P/LP", "VUS", "B/LB"])}
        for g in range(n_genes)
        for i in range(per_gene)
    ]
    return pd.DataFrame(rows)


def test_gene_disjoint_split_has_no_gene_overlap():
    df = _toy_variants()
    train, test = gene_disjoint_split(df, test_size=0.25, seed=1)
    assert set(train["GeneSymbol"]).isdisjoint(test["GeneSymbol"])
    assert len(train) + len(test) == len(df)


def test_sample_per_class_respects_caps():
    df = _toy_variants()
    sample = sample_per_class(df, n=10, max_per_gene=1, seed=0)
    assert sample["label"].value_counts().max() <= 10
    assert sample.groupby(["GeneSymbol", "label"]).size().max() == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("P/LP", "P/LP"), ("vus", "VUS"), ("Likely Benign", "B/LB"), ("pathogenic", "P/LP"), ("??", INVALID),
     (None, INVALID)],
)  # fmt: skip
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_evidence_text_truncates_each_paper():
    rec = {"papers": [{"pmid": "1", "title": "T1", "abstract": "word " * 500},
                      {"pmid": "2", "title": "T2", "abstract": "short"}]}  # fmt: skip
    text = evidence_text(rec, max_chars=200)
    assert "[1] PMID 1: T1" in text and "[2] PMID 2: T2\nshort" in text
    assert "[...]" in text and len(text) < 400


def test_metrics_on_toy_data():
    y = np.array(["P/LP", "VUS", "B/LB", "P/LP"])
    assert accuracy(y, y) == 1.0 and macro_f1(y, y) == 1.0
    pred = np.array(["P/LP", INVALID, "B/LB", "VUS"])
    assert accuracy(y, pred) == 0.5
    s = summarize(y, pred, n_boot=50)
    assert s["invalid_rate"] == 0.25
    assert s["macro_f1_ci"][0] <= s["macro_f1"] <= s["macro_f1_ci"][1]


def test_bootstrap_ci_contains_point_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)
    labels = np.array(["P/LP", "VUS", "B/LB"])

    def width(n):
        y = rng.choice(labels, n)
        pred = np.where(rng.random(n) < 0.7, y, rng.choice(labels, n))
        lo, hi = bootstrap_ci(y, pred, accuracy, n_boot=300)
        assert lo <= accuracy(y, pred) <= hi
        return hi - lo

    assert width(2000) < width(100)


def test_paired_tests_detect_a_clear_difference():
    rng = np.random.default_rng(0)
    y = rng.choice(["P/LP", "VUS", "B/LB"], 300)
    good = y.copy()
    bad = np.where(rng.random(300) < 0.5, y, "VUS")
    pb = paired_bootstrap(y, good, bad, accuracy, n_boot=300)
    assert pb["diff"] > 0 and pb["ci_low"] > 0 and pb["p_value"] < 0.01
    assert mcnemar(y, good, bad)["p_value"] < 0.01
    assert mcnemar(y, good, good)["p_value"] == 1.0


def test_parse_json_response():
    assert parse_json_response('```json\n{"label": "VUS"}\n```') == {"label": "VUS"}
    assert parse_json_response('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_json_response("no json") is None
    assert parse_json_response("{broken") is None


def test_split_papers_and_pilot_answer_filter():
    text = "COMMENT:\nx\n---\n#####\nPAPER 1\nA\n#####\nPAPER 2\nB\n"
    assert split_papers(text) == ["PAPER 1\nA", "PAPER 2\nB"]
    assert filter_answer('{"number_of_participants": "1,234"}') == 1234
    assert filter_answer('{"number_of_participants": 81}') == 81
    assert filter_answer("none") is None
