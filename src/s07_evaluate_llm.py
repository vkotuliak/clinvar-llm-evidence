import gc
from unittest import result
import torch
from s06_LLM_analysis import analyze_text_with_llm
import collections
import json
import re
from datetime import datetime

# find a question, which will have one answer per paper, something like number of participants
# then run it 10 times with all the models
# which will repeat same answer most times is the best


def filter_answer(answer: str) -> int:
    """Filters the answer to extract the number of participants from a JSON-like string."""
    try:
        # Find the JSON object within the answer string
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            # Extract the number of participants and return it as a string
            if "number_of_participants" in data:
                try:
                    return int(data["number_of_participants"])
                except ValueError:
                    return -1
                # return int(data["number_of_participants"])
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the string is not valid JSON or other errors
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
        "BioMistral/BioMistral-7B",
        "meta-llama/Llama-3.3-70B-Instruct"
    ]
    num_runs = 10
    results = {}
    with open("aggregated_data_306.txt", "r") as file:
        text = file.read()

    for model in models_to_test:
        print(f"--- Evaluating model: {model} ---")
        answers = []
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs} for {model}...")
            answer = analyze_text_with_llm(text, model)
            filtered = filter_answer(answer)
            answers.append(filtered)

        # results[model] = collections.Counter(answers)
        count_of_2 = answers.count(2)
        results[model] = count_of_2

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print the results
    print("\n--- Evaluation Summary ---")
    with open("model_evaluation_results.txt", "a") as f:
        f.write(f"Evaluation run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(results.__str__() + "\n")
        f.write("=" * 40 + "\n")
    for model, counts in results.items():
        print(f"\nModel: {model}")
        print(f"Count of '2' answers: {counts} out of {num_runs} runs")
    #     with open("model_evaluation_results.txt", "a") as f:
    #         f.write(f"Model: {model}\n")
    #         f.write(f"Count of '2' answers: {counts} out of {num_runs} runs\n\n")
        # for answer, count in counts.most_common():
        #     print(
        #         f"  - Answer: '{answer}' was given {count}/{num_runs} times."
        #     )


if __name__ == "__main__":
    evaluate_models()
