#!/usr/bin/env bash
# Run every LLM on the test split, with and without literature evidence.
# Resumable: re-running only computes missing predictions.
set -euo pipefail
PY=${PYTHON:-python}
MODELS=${MODELS:-"qwen2.5:7b gemma4:e4b llama3.1:8b qwen2.5:3b"}

for m in $MODELS; do
  ollama show "$m" >/dev/null 2>&1 || ollama pull "$m"
  $PY -m benchmark.run_llm --model "$m"
done
for m in $MODELS; do
  $PY -m benchmark.run_llm --model "$m" --no-evidence
done
