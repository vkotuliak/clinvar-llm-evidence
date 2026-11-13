PYTHON ?= .venv/bin/python
MODELS ?= qwen2.5:7b gemma4:e4b llama3.1:8b qwen2.5:3b

.PHONY: install benchmark baselines eval report all test lint

install:            ## create .venv and install dependencies
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt

benchmark:          ## download ClinVar exports, fetch abstracts, build gene-disjoint splits
	$(PYTHON) -m benchmark.build_benchmark

baselines:          ## majority + TF-IDF/logistic-regression baselines
	$(PYTHON) -m benchmark.baselines

eval:               ## all LLMs with and without evidence (needs `ollama serve`)
	PYTHON=$(PYTHON) MODELS="$(MODELS)" ./scripts/run_llm_eval.sh

report:             ## metrics, bootstrap CIs, significance tests, figures
	$(PYTHON) -m benchmark.report

all: benchmark baselines eval report

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
