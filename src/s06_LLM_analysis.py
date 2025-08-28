import transformers
import torch
from transformers import BitsAndBytesConfig

from s05_data_preparation import aggregate_data
import pandas as pd


def analyze_text_with_llm(text, model_id) -> str:
    """Analyze text using a specified LLM model."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    if model_id == "google/medgemma-4b-it":
        pipeline = transformers.pipeline(
            "image-text-to-text",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization_config,
            },
            device_map="auto",
        )
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization_config,
            },
            device_map="auto",
        )

    text_chunks = text.split("#####")
    # for i, chunk in enumerate(text_chunks):
    #     char_count = len(chunk)
    #     print(f"Chunk {i+1}: {char_count} characters")


    # 2. Map Step: Process all chunks in a single batch for efficiency
    print(f"Processing {len(text_chunks)} chunks in a batch...")
    summary_prompts = [
        f"""
    ### Input Text:
    {text_chunks[1]}

    From the text above find the information about number of participants in the study.
    List ONLY the number of participants in specified JSON format:

    ```
    {{
      "number_of_participants": "..."
    }}
    ```
    """
        # for chunk in text_chunks
    ]

    responses = pipeline(
        summary_prompts, max_new_tokens=512, num_return_sequences=1
    )

    summaries = []
    for i, response in enumerate(responses):
        prompt = summary_prompts[i]
        summary = response[0]["generated_text"][len(prompt) :].strip()
        print(f"Summary for chunk {i+1}: {summary}")
        return summary
        summaries.append(summary)
    print(f"All summaries: {summaries}")

    # 3. Reduce Step: Synthesize the summaries
    print("FINITO")
    # print("Synthesizing final themes...")
    # combined_summaries = "\n\n---\n\n".join(summaries)

    # final_prompt = f"""
    # ### Instruction:
    # Read the text provided below. List only the number of participants in each paper. If it isn't study with participants list NA.

    # ### Input Text:
    # {combined_summaries}
    # """

    # response = pipeline(
    #     final_prompt, max_new_tokens=512, num_return_sequences=1
    # )
    # analysis = response[0]["generated_text"][len(final_prompt) :]

    # if model_id == "openai/gpt-oss-20b":
    #     keyword = ".assistantfinal"
    #     analysis = analysis[analysis.find(keyword) + len(keyword) :].strip()

    return summaries


def analyze_random_10():
    """Randomly pick 10 variations from data/clinvar_csv_dataset.csv and analyze them using llm."""
    df = pd.read_csv('data/clinvar_csv_dataset.csv')
    random_10 = df.sample(n=min(10, len(df)), random_state=42)

    for _, row in random_10.iterrows():
        variation_id = row['VariationID']
        print(f"Processing variation {variation_id}...")
        data = aggregate_data(variation_id)
        analysis_results = analyze_text_with_llm(data)
        print(f"Analysis for variation {variation_id}:")
        print(analysis_results)
        print("-" * 50)
        # Write analysis to file (append to same file for all variations)
        with open("all_variations_analysis.txt", 'a') as f:
            f.write(f"Analysis for variation {variation_id}:\n")
            f.write(analysis_results)
            f.write("\n" + "-" * 50 + "\n")


def main():
    # model_id = "Qwen/Qwen3-4B"
    # model_id = "google/medgemma-4b-it"
    # model_id = "Qwen/Qwen2.5-7B"
    # model_id = "BioMistral/BioMistral-7B" # context window is too small :(
    # model_id = "openai/gpt-oss-20b"
    # model_id = "meta-llama/Llama-3.3-70B-Instruct" # Too big, requires more resources
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    variation_id = 306
    data = aggregate_data(variation_id)
    analysis_results = analyze_text_with_llm(data, model_id)
    print("Analysis Results:")
    print(analysis_results)
    # analyze_random_10()


if __name__ == "__main__":
    main()
