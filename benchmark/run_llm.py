"""Zero-shot LLM classification of benchmark variants (local models via Ollama by default).

Resumable: predictions are appended to ``results/predictions/<model>.jsonl`` one at a time,
and variants that already have a prediction are skipped on restart.

    python -m benchmark.run_llm --model qwen2.5:7b
    python -m benchmark.run_llm --model qwen2.5:7b --no-evidence   # ablation: variant name only
"""

import argparse
import json

from benchmark.build_benchmark import LABELS
from benchmark.common import (
    INVALID,
    evidence_text,
    load_split,
    normalize_label,
    predictions_path,
    read_predictions,
    variant_header,
)
from src.s06_LLM_analysis import LLMClient, parse_json_response

SYSTEM_PROMPT = """You are an expert clinical geneticist curating germline variants for ClinVar \
using the ACMG/AMP guidelines. Weigh the evidence that is available: predicted molecular \
consequence (e.g. nonsense, frameshift, canonical splice, missense, synonymous), segregation and \
case reports, functional studies, population frequency and computational evidence."""

LABEL_DEFINITIONS = """Classify the variant into exactly one class:
- "P/LP": pathogenic or likely pathogenic
- "VUS": uncertain significance - evidence is insufficient or conflicting
- "B/LB": benign or likely benign"""

ANSWER_FORMAT = (
    """Answer with JSON: {"rationale": "<at most 3 sentences>", "label": "P/LP" | "VUS" | "B/LB"}"""
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "label": {"type": "string", "enum": LABELS},
    },
    "required": ["rationale", "label"],
}

DEFAULT_MAX_EVIDENCE_CHARS = 16_000  # ~4k tokens; keeps prompt + answer within an 8k context


def build_messages(rec: dict, with_evidence: bool = True, max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS):
    parts = [variant_header(rec)]
    if with_evidence:
        parts.append(
            "Publications cited as evidence for this variant (titles and abstracts):\n\n"
            + evidence_text(rec, max_chars)
        )
    parts += [LABEL_DEFINITIONS, ANSWER_FORMAT]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def run_name(model: str, with_evidence: bool) -> str:
    return f"llm-{model}" + ("" if with_evidence else "-no-evidence")


def run(model: str, backend: str, with_evidence: bool, limit: int | None, split: str) -> None:
    records = load_split(split)[:limit]
    out_path = predictions_path(run_name(model, with_evidence))
    done = {p["variation_id"] for p in read_predictions(out_path)}
    todo = [r for r in records if r["variation_id"] not in done]
    print(f"{model} ({'evidence' if with_evidence else 'no evidence'}): {len(done)} done, {len(todo)} to go")

    client = LLMClient(model, backend=backend, max_new_tokens=300)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    with out_path.open("a") as f:
        for i, rec in enumerate(todo, start=1):
            resp = client.chat(build_messages(rec, with_evidence), json_schema=RESPONSE_SCHEMA)
            parsed = parse_json_response(resp.text) or {}
            pred = normalize_label(parsed.get("label"))
            n_correct += pred == rec["label"]
            f.write(json.dumps({
                "variation_id": rec["variation_id"],
                "label": rec["label"],
                "pred": pred,
                "rationale": parsed.get("rationale"),
                "raw": resp.text if pred == INVALID else None,
                "latency_s": round(resp.latency_s, 3),
                "prompt_tokens": resp.prompt_tokens,
                "output_tokens": resp.output_tokens,
            }, ensure_ascii=False) + "\n")  # fmt: skip
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  running acc {n_correct / i:.3f}  last {resp.latency_s:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM on the benchmark.")
    parser.add_argument("--model", required=True, help="Ollama tag or Hugging Face model id")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    parser.add_argument("--no-evidence", action="store_true", help="ablation: variant description only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run(args.model, args.backend, not args.no_evidence, args.limit, args.split)
