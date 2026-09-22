import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.generator import GenerationResult
from rag.generation_prompt import build_generation_prompt


class QwenLocalGenerator:
    """Generates RAG answers using Qwen2.5-1.5B-Instruct locally."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer only using the provided context. "
                    "Do not use outside knowledge. "
                    "Write a complete, direct answer."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[0][prompt_length:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise RuntimeError(
                "Qwen model returned an empty answer."
            )

        return GenerationResult(
            answer=answer,
            model=self.model_name,
        )
