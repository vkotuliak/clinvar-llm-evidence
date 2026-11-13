"""Aggregate every prediction file into metrics, significance tests, a Markdown table and figures.

Outputs: ``results/metrics.json``, ``results/results.md``, ``results/figures/*.png``.
Only runs that cover the whole test split are reported (partial runs are listed and skipped).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.build_benchmark import LABELS
from benchmark.common import load_split, read_predictions
from benchmark.metrics import accuracy, macro_f1, mcnemar, paired_bootstrap, summarize
from src.config import FIGURES_DIR, PREDICTIONS_DIR, RESULTS_DIR

N_BOOT = 2000

# Palette (light surface): categorical slots 1-2, muted ink for baselines, blue sequential ramp.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

BASELINE_NAMES = {
    "majority": "Majority class",
    "tfidf-name-only": "TF-IDF + LR, variant name only",
    "tfidf-evidence": "TF-IDF + LR, name + abstracts",
}
MODEL_NAMES = {
    "qwen2.5-7b": "Qwen2.5 7B",
    "qwen2.5-3b": "Qwen2.5 3B",
    "llama3.1-8b": "Llama 3.1 8B",
    "gemma4-e4b": "Gemma 4 E4B",
}


def display_name(run: str) -> str:
    if run in BASELINE_NAMES:
        return BASELINE_NAMES[run]
    model = run.removeprefix("llm-").removesuffix("-no-evidence")
    return MODEL_NAMES.get(model, model) + (" (no evidence)" if run.endswith("-no-evidence") else "")


def load_runs(test_ids: list[int]) -> dict[str, np.ndarray]:
    runs = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.jsonl")):
        preds = {p["variation_id"]: p["pred"] for p in read_predictions(path)}
        if set(test_ids) - preds.keys():
            print(f"skipping partial run {path.stem}: {len(preds)}/{len(test_ids)}")
            continue
        runs[path.stem] = np.array([preds[i] for i in test_ids])
    return runs


def latency(run: str) -> float | None:
    vals = [p.get("latency_s") for p in read_predictions(PREDICTIONS_DIR / f"{run}.jsonl")]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_macro_f1(metrics: dict, path) -> None:
    runs = [r for r in metrics if not r.endswith("-no-evidence")]
    runs.sort(key=lambda r: metrics[r]["macro_f1"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(runs) + 1.4), facecolor=SURFACE)
    style(ax)
    for y, run in enumerate(runs):
        m = metrics[run]
        lo, hi = m["macro_f1_ci"]
        color = MUTED if run in BASELINE_NAMES else BLUE
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter(m["macro_f1"], y, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.012, y, f"{m['macro_f1']:.2f}", va="center", color=INK, fontsize=9)
    ax.axvline(1 / 3, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.text(1 / 3, len(runs) - 0.4, " uniform random guess", color=INK_2, fontsize=8, va="bottom")
    ax.set_yticks(range(len(runs)), [display_name(r) for r in runs], color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(runs) - 0.2)
    ax.set_xlabel("Macro-F1 on 300 test variants (dot) with 95% bootstrap CI (line)", color=INK_2)
    ax.set_title("LLMs vs. baselines", loc="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_ablation(metrics: dict, path) -> None:
    models = [r for r in metrics if r.startswith("llm-") and f"{r}-no-evidence" in metrics]
    if not models:
        return
    models.sort(key=lambda r: metrics[r]["macro_f1"])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.6), facecolor=SURFACE)
    style(ax)
    for y, run in enumerate(models):
        with_ev, without = metrics[run]["macro_f1"], metrics[f"{run}-no-evidence"]["macro_f1"]
        ax.plot([without, with_ev], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter(without, y, s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(with_ev, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        delta = with_ev - without
        ax.text(max(with_ev, without) + 0.015, y, f"{delta:+.2f}", va="center", color=INK, fontsize=9)
    ax.scatter([], [], s=50, color=ORANGE, label="Variant name only")
    ax.scatter([], [], s=50, color=BLUE, label="Name + cited abstracts")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2, fontsize=9)
    ax.set_yticks(range(len(models)), [display_name(r) for r in models], color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Macro-F1", color=INK_2)
    ax.set_title("How much does the literature help?", loc="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(m: dict, title: str, path) -> None:
    cm = np.array(m["confusion"])
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(4.6, 4), facecolor=SURFACE)
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ink = SURFACE if row_share[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=ink, fontsize=11)
    ax.set_xticks(range(len(LABELS)), LABELS, color=INK)
    ax.set_yticks(range(len(LABELS)), LABELS, color=INK)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("ClinVar label", color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fmt_ci(m: dict, key: str) -> str:
    lo, hi = m[f"{key}_ci"]
    return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def markdown_table(metrics: dict) -> str:
    rows = [
        "| System | Macro-F1 [95% CI] | Accuracy [95% CI] | F1 P/LP | F1 VUS | F1 B/LB | Invalid "
        "| Median s/variant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, m in sorted(metrics.items(), key=lambda kv: -kv[1]["macro_f1"]):
        f1 = m["per_class_f1"]
        lat = f"{m['median_latency_s']:.1f}" if m.get("median_latency_s") else "-"
        rows.append(
            f"| {display_name(run)} | {fmt_ci(m, 'macro_f1')} | {fmt_ci(m, 'accuracy')} | "
            f"{f1['P/LP']:.2f} | {f1['VUS']:.2f} | {f1['B/LB']:.2f} | {m['invalid_rate']:.1%} | {lat} |"
        )
    return "\n".join(rows)


def comparisons(runs: dict, y: np.ndarray, metrics: dict) -> list[dict]:
    llms = [r for r in runs if r.startswith("llm-") and not r.endswith("-no-evidence")]
    out = []
    pairs = []
    if llms:
        best = max(llms, key=lambda r: metrics[r]["macro_f1"])
        pairs += [(best, b) for b in ("tfidf-evidence", "tfidf-name-only") if b in runs]
    pairs += [(r, f"{r}-no-evidence") for r in llms if f"{r}-no-evidence" in runs]
    for a, b in pairs:
        out.append({
            "a": a,
            "b": b,
            "macro_f1": paired_bootstrap(y, runs[a], runs[b], macro_f1, N_BOOT),
            "accuracy": paired_bootstrap(y, runs[a], runs[b], accuracy, N_BOOT),
            "mcnemar": mcnemar(y, runs[a], runs[b]),
        })  # fmt: skip
    return out


def main() -> None:
    test = load_split("test")
    ids = [r["variation_id"] for r in test]
    y = np.array([r["label"] for r in test])
    runs = load_runs(ids)

    metrics = {}
    for run, pred in runs.items():
        metrics[run] = summarize(y, pred, n_boot=N_BOOT)
        metrics[run]["median_latency_s"] = latency(run)
    comps = comparisons(runs, y, metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps({"systems": metrics, "comparisons": comps}, indent=2)
    )

    lines = ["# Results", "", f"Test split: {len(y)} variants ({', '.join(LABELS)}: 100 each).", ""]
    lines += [markdown_table(metrics), "", "## Paired comparisons (A - B)", ""]
    lines += [
        "| A | B | Δ macro-F1 [95% CI] | p (bootstrap) | Δ accuracy | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for c in comps:
        f, a = c["macro_f1"], c["accuracy"]
        lines.append(
            f"| {display_name(c['a'])} | {display_name(c['b'])} | {f['diff']:+.3f} "
            f"[{f['ci_low']:+.3f}, {f['ci_high']:+.3f}] | {f['p_value']:.3f} | {a['diff']:+.3f} | "
            f"{c['mcnemar']['p_value']:.3g} |"
        )
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_macro_f1(metrics, FIGURES_DIR / "macro_f1.png")
    plot_ablation(metrics, FIGURES_DIR / "evidence_ablation.png")
    llms = [r for r in metrics if r.startswith("llm-") and not r.endswith("-no-evidence")]
    if llms:
        best = max(llms, key=lambda r: metrics[r]["macro_f1"])
        plot_confusion(
            metrics[best], f"{display_name(best)} - confusion matrix", FIGURES_DIR / "confusion_best_llm.png"
        )
    if "tfidf-evidence" in metrics:
        plot_confusion(
            metrics["tfidf-evidence"], "TF-IDF + LR - confusion matrix", FIGURES_DIR / "confusion_tfidf.png"
        )


if __name__ == "__main__":
    main()
