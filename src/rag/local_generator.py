import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag.generation_prompt import build_generation_prompt
from rag.generator import GenerationResult


class LocalGenerator:
    """Generates answers using a local Hugging Face seq2seq model."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> GenerationResult:
        prompt = build_generation_prompt(
            query=query,
            context=context,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise RuntimeError("Local model returned an empty answer.")

        return GenerationResult(
            answer=answer,
            model=self.model_name,
        )
