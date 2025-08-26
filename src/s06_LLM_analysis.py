import transformers
import torch
from transformers import BitsAndBytesConfig

from s05_data_preparation import aggregate_data
import pandas as pd


def analyze_text_with_llm(text):
    model_id = "Qwen/Qwen3-4B"
    # model_id = "Qwen/Qwen2.5-7B"
    # model_id = "openai/gpt-oss-20b"
    # model_id = "meta-llama/Llama-3.3-70B-Instruct" # Too big, requires more resources
    # model_id = "Qwen/Qwen2.5-72B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization_config,
        },
        device_map="auto",
    )

    # 1. Split the text into manageable chunks
    # chunk_size = 500000
    # text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    text_chunks = text.split("#####")
    # text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]

    # 2. Map Step: Process all chunks in a single batch for efficiency
    print(f"Processing {len(text_chunks)} chunks in a batch...")

    summary_prompts = [
        f"""
    ### Instruction:
    Read the text bellow. State only the number of participants in the scientific paper and nothing more. If it is paper without participants list NA.

    ### Input Text:
    {chunk}

    ### Response:
    """
        for chunk in text_chunks
    ]

    responses = pipeline(
        summary_prompts, max_new_tokens=256, num_return_sequences=1
    )

    summaries = []
    for i, response in enumerate(responses):
        prompt = summary_prompts[i]
        summary = response[0]["generated_text"][len(prompt) :].strip()
        print(f"Summary for chunk {i+1}: {summary}")
        summaries.append(summary)

    # 3. Reduce Step: Synthesize the summaries
    print("Synthesizing final themes...")
    combined_summaries = "\n\n---\n\n".join(summaries)

    final_prompt = f"""
    ### Instruction:
    Read the text provided below. List only the number of participants in each paper. If it isn't study with participants list NA.

    ### Input Text:
    {combined_summaries}
    """

    response = pipeline(
        final_prompt, max_new_tokens=512, num_return_sequences=1
    )
    analysis = response[0]["generated_text"][len(final_prompt) :]

    if model_id == "openai/gpt-oss-20b":
        keyword = ".assistantfinal"
        analysis = analysis[analysis.find(keyword) + len(keyword) :].strip()

    return analysis


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
    # variation_id = 3254
    # data = aggregate_data(variation_id)
    # analysis_results = analyze_text_with_llm(data)
    # print("Analysis Results:")
    # print(analysis_results)
    analyze_random_10()


if __name__ == "__main__":
    main()
