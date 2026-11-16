"""Pilot experiment (Aug 2025): consistency of participant-count extraction from one paper.

The question "how many participants were studied?" was asked N times per model about a
single paper with a known answer, and we counted how often each model got it right
(see ``pilot_participant_count_log.txt``). It motivated the larger benchmark in ``benchmark/``:
a single paper is far too small a sample to compare models.

    python -m experiments.pilot_participant_count --variation 3254 --paper 1 --answer 81
"""

import argparse
import re
from collections import Counter

from src.s05_data_preparation import aggregate_data
from src.s06_LLM_analysis import LLMClient, extract_participants, split_papers

_NUMBER = re.compile(r'"number_of_participants"\s*:\s*"?([\d,]+)"?')


def filter_answer(answer: str) -> int | None:
    """Extract the participant count from a raw model answer (accepts ``"1,234"``)."""
    match = _NUMBER.search(answer or "")
    return int(match.group(1).replace(",", "")) if match else None


def evaluate(models, variation_id: int, paper: int, answer: int, runs: int, temperature: float):
    papers = split_papers(aggregate_data(variation_id))
    text = papers[paper - 1]
    results = {}
    for model in models:
        client = LLMClient(model, temperature=temperature)
        answers = Counter()
        for run in range(runs):
            client.seed = run
            answers[extract_participants(text, client)] += 1
        results[model] = answers
        print(f"{model}: correct {answers[answer]}/{runs}, answers {dict(answers)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b"])
    parser.add_argument("--variation", type=int, default=3254)
    parser.add_argument("--paper", type=int, default=1, help="1-based index of the cited paper")
    parser.add_argument("--answer", type=int, default=81, help="ground-truth participant count")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    evaluate(args.models, args.variation, args.paper, args.answer, args.runs, args.temperature)
