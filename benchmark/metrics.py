"""Classification metrics with bootstrap confidence intervals and paired significance tests.

Invalid / unparseable model outputs are kept as their own prediction value, so they count
as errors for every class (they are never silently dropped).
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from benchmark.build_benchmark import LABELS

Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, labels: Sequence[str] = LABELS) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, labels: Sequence[str] = LABELS) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {label: float(s) for label, s in zip(labels, scores, strict=True)}


def confusion(y_true, y_pred, labels: Sequence[str] = LABELS) -> list[list[int]]:
    """Rows = true label, columns = predicted label (invalid predictions are not shown)."""
    return confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    y_true, y_pred, metric: Metric, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI over test examples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = [metric(y_true[i], y_pred[i]) for i in _bootstrap_indices(len(y_true), n_boot, seed)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(
    y_true, pred_a, pred_b, metric: Metric, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Difference ``metric(a) - metric(b)`` with a 95% CI and a one-sided bootstrap p-value.

    The same resampled indices are used for both systems, so the test accounts for the
    fact that they are evaluated on identical examples.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    diffs = np.array([
        metric(y_true[i], pred_a[i]) - metric(y_true[i], pred_b[i])
        for i in _bootstrap_indices(len(y_true), n_boot, seed)
    ])  # fmt: skip
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": metric(y_true, pred_a) - metric(y_true, pred_b),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(np.mean(diffs <= 0)),
    }


def mcnemar(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test on per-example correctness of two systems."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def summarize(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci": bootstrap_ci(y_true, y_pred, accuracy, n_boot, seed=seed),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci": bootstrap_ci(y_true, y_pred, macro_f1, n_boot, seed=seed),
        "per_class_f1": per_class_f1(y_true, y_pred),
        "invalid_rate": float(np.mean(~np.isin(y_pred, LABELS))),
        "confusion": confusion(y_true, y_pred),
    }
