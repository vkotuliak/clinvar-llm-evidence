import gc
import torch
from s06_LLM_analysis import analyze_text_with_llm
import re
from datetime import datetime

# find a question, which will have one answer per paper, something like number of participants
# then run it 10 times with all the models
# which will repeat same answer most times is the best


def filter_answer(answer: str) -> int:
    """Filters the answer to extract the number of participants using a regular expression."""
    try:
        match = re.search(r'"number_of_participants"\s*:\s*"?(\d+)"?', answer)
        if match:
            return int(match.group(1))
    except (ValueError, IndexError):
        pass
    return -1


def evaluate_models():
    """
    Evaluates LLM consistency by running the same question multiple times
    and counting the frequency of each answer.
    """
    models_to_test = [
        "google/medgemma-4b-it",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-72B-Instruct",
        "BioMistral/BioMistral-7B"
    ]
    num_runs = 10
    results = {}
    with open("aggregated_data_3254.txt", "r") as file:
        text = file.read()

    for model in models_to_test:
        print(f"--- Evaluating model: {model} ---")
        answers = []
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs} for {model}...")
            answer = analyze_text_with_llm(text, model)
            filtered = filter_answer(answer)
            answers.append(filtered)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # results[model] = collections.Counter(answers)
        count_answer = answers.count(81)
        results[model] = count_answer


    # Print the results
    print("\n--- Evaluation Summary ---")
    with open("model_evaluation_results.txt", "a") as f:
        f.write(f"Evaluation run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(results.__str__() + "\n")
        f.write("=" * 40 + "\n")
    for model, counts in results.items():
        print(f"\nModel: {model}")
        print(f"Count of '81' answers: {counts} out of {num_runs} runs")
    #     with open("model_evaluation_results.txt", "a") as f:
    #         f.write(f"Model: {model}\n")
    #         f.write(f"Count of '2' answers: {counts} out of {num_runs} runs\n\n")
        # for answer, count in counts.most_common():
        #     print(
        #         f"  - Answer: '{answer}' was given {count}/{num_runs} times."
        #     )


if __name__ == "__main__":
    evaluate_models()
