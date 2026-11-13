"""Stage 6: LLM client + per-paper information extraction.

``LLMClient`` hides two backends behind one ``chat()`` call:

* ``ollama`` (default) - local quantised models via the Ollama HTTP API; works on a laptop.
* ``hf``              - Hugging Face transformers, 4-bit on CUDA when available; for GPU servers.

Models are loaded once per client, prompts go through the model's chat template, and
generation is deterministic (temperature 0, fixed seed) so repeated runs are comparable.
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests

from src.s05_data_preparation import PAPER_SEPARATOR, aggregate_data

OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class LLMClient:
    model: str
    backend: str = "ollama"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    max_new_tokens: int = 512
    host: str = OLLAMA_URL
    _hf: tuple | None = field(default=None, init=False, repr=False)

    def chat(self, messages: list[dict], json_schema: dict | None = None) -> LLMResponse:
        if self.backend == "ollama":
            return self._chat_ollama(messages, json_schema)
        if self.backend == "hf":
            return self._chat_hf(messages)
        raise ValueError(f"Unknown backend {self.backend!r}")

    # -- Ollama ---------------------------------------------------------------------------
    def _chat_ollama(self, messages, json_schema) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                # Ollama silently truncates prompts longer than num_ctx, so set it explicitly.
                "num_ctx": self.num_ctx,
                "num_predict": self.max_new_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        start = time.perf_counter()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=900)
        r.raise_for_status()
        body = r.json()
        return LLMResponse(
            text=body["message"]["content"],
            latency_s=time.perf_counter() - start,
            prompt_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )

    # -- Hugging Face ---------------------------------------------------------------------
    def _load_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        torch.manual_seed(self.seed)
        return tokenizer, model

    def _chat_hf(self, messages) -> LLMResponse:
        if self._hf is None:
            self._hf = self._load_hf()
        tokenizer, model = self._hf
        start = time.perf_counter()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device
        )
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        output = model.generate(inputs, **gen_kwargs)
        new_tokens = output[0, inputs.shape[-1] :]
        return LLMResponse(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True),
            latency_s=time.perf_counter() - start,
            prompt_tokens=int(inputs.shape[-1]),
            output_tokens=int(new_tokens.shape[-1]),
        )


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str) -> dict | None:
    """Extract the first JSON object from a model answer (tolerates code fences and chatter)."""
    if not text:
        return None
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# -- Per-paper extraction: number of study participants ------------------------------------

PARTICIPANTS_SCHEMA = {
    "type": "object",
    "properties": {"number_of_participants": {"type": ["integer", "null"]}},
    "required": ["number_of_participants"],
}

PARTICIPANTS_PROMPT = """Read the scientific text below and report how many human participants \
(patients, probands, family members or controls) were studied. Answer null if the paper is not \
a study with participants.

Answer ONLY with JSON: {{"number_of_participants": <integer or null>}}

### Text
{text}"""


def split_papers(aggregated_text: str) -> list[str]:
    """Split the output of ``aggregate_data`` into one chunk per cited paper (drops the comment header)."""
    return [c.strip() for c in aggregated_text.split(PAPER_SEPARATOR)[1:] if c.strip()]


def extract_participants(paper_text: str, client: LLMClient) -> int | None:
    messages = [{"role": "user", "content": PARTICIPANTS_PROMPT.format(text=paper_text)}]
    answer = parse_json_response(client.chat(messages, json_schema=PARTICIPANTS_SCHEMA).text)
    value = (answer or {}).get("number_of_participants")
    if isinstance(value, str):
        value = int(value.replace(",", "")) if value.replace(",", "").isdigit() else None
    return value if isinstance(value, int) else None


def analyze_variation(variation_id: int, client: LLMClient) -> list[int | None]:
    """Participant count for every paper cited by a ClinVar variation."""
    papers = split_papers(aggregate_data(variation_id))
    return [extract_participants(p, client) for p in papers]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variation_id", type=int)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    args = parser.parse_args()

    counts = analyze_variation(args.variation_id, LLMClient(args.model, backend=args.backend))
    for i, n in enumerate(counts, start=1):
        print(f"Paper {i}: {n}")
