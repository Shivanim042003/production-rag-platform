import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.grounding import GroundingResult


class QwenGroundingScorer:
    """Checks whether an answer is supported by context using Qwen."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        self.model_name = model_name
        self.threshold = threshold

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
    ) -> GroundingResult:
        if not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not context:
            raise ValueError("Context cannot be empty.")

        formatted_context = "\n\n".join(
            f"[Context {index}]\n{text}"
            for index, text in enumerate(context, start=1)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer-grounding evaluator.\n"
                    "Determine whether the answer is fully supported "
                    "by the provided context.\n"
                    "Do not use outside knowledge.\n\n"
                    "Reply with exactly one word:\n"
                    "YES\n"
                    "or\n"
                    "NO"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<CONTEXT>\n"
                    f"{formatted_context}\n"
                    f"</CONTEXT>\n\n"
                    f"<ANSWER>\n"
                    f"{answer}\n"
                    f"</ANSWER>"
                ),
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
                max_new_tokens=4,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[0][prompt_length:]

        decision = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        normalized = decision.upper()

        # Accept a few safe variants produced by small local models.
        match = re.search(
            r"\b(YES|NO)\b",
            normalized,
        )

        if match is None:
            raise RuntimeError(
                "Qwen returned an invalid grounding decision: "
                f"{decision!r}"
            )

        grounded = match.group(1) == "YES"

        return GroundingResult(
            grounded=grounded,
            score=1.0 if grounded else 0.0,
            reason=(
                "Qwen judged the answer to be supported "
                "by the context."
                if grounded
                else
                "Qwen judged the answer to be unsupported "
                "by the context."
            ),
        )
