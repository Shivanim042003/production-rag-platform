import re
from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class FaithfulnessScorer(Protocol):
    """Protocol for scoring answer faithfulness."""

    def score(
        self,
        answer: str,
        context: list[str],
    ) -> float:
        """Return a faithfulness score between 0 and 1."""
        ...


class QwenFaithfulnessScorer:
    """Evaluate whether an answer is supported by retrieved context."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

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

    def score(
        self,
        answer: str,
        context: list[str],
    ) -> float:
        if not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not context:
            raise ValueError("Context cannot be empty.")

        if any(not item.strip() for item in context):
            raise ValueError(
                "Context cannot contain empty strings."
            )

        formatted_context = "\n\n".join(
            f"[Context {index}]\n{item}"
            for index, item in enumerate(context, start=1)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict faithfulness evaluator. "
                    "Determine whether the answer is fully "
                    "supported by the provided context. "
                    "Answer YES if the answer is supported by "
                    "the context. Answer NO if the answer "
                    "contains unsupported claims. "
                    "Return only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{formatted_context}\n\n"
                    f"Answer:\n{answer}\n\n"
                    "Is the answer fully supported by the context?"
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
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
                max_new_tokens=8,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[0][prompt_length:]

        decision = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        normalized = decision.upper()

        match = re.search(
            r"\b(YES|NO)\b",
            normalized,
        )

        if match is None:
            raise RuntimeError(
                "Faithfulness evaluator returned an "
                f"unrecognized response: {decision!r}"
            )

        return 1.0 if match.group(1) == "YES" else 0.0