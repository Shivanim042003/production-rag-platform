from typing import Protocol

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class MultiQueryGenerator(Protocol):
    def generate(self, query: str) -> list[str]:
        """Generate multiple retrieval queries from a user query."""
        ...


class IdentityMultiQueryGenerator:
    """
    Baseline multi-query generator.

    Returns the original query as a single-item list.
    Useful as a baseline before introducing LLM-generated
    query variations.
    """

    def generate(self, query: str) -> list[str]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return [query.strip()]


class QwenMultiQueryGenerator:
    """
    Generates multiple retrieval-oriented query variations
    using an already-loaded Qwen causal language model.

    The model and tokenizer are injected so that Qwen is not
    loaded a second time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: str,
        model_name: str,
        num_queries: int = 3,
        max_new_tokens: int = 128,
    ) -> None:
        if num_queries <= 0:
            raise ValueError(
                "num_queries must be greater than zero."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model_name = model_name
        self.num_queries = num_queries
        self.max_new_tokens = max_new_tokens

    def generate(self, query: str) -> list[str]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate alternative search queries for "
                    "document retrieval. "
                    f"Generate exactly {self.num_queries} different queries. "
                    "Preserve the original meaning. "
                    "Use different wording or perspectives. "
                    "Include important technical terms. "
                    "Do not answer the question. "
                    "Return only the queries, one per line, "
                    "optionally numbered."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate alternative retrieval queries for:\n\n"
                    f"{query.strip()}\n\n"
                    "Queries:"
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
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[0][prompt_length:]

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not generated_text:
            raise RuntimeError(
                "Qwen multi-query generator returned empty output."
            )

        queries = self._parse_queries(generated_text)

        if not queries:
            raise RuntimeError(
                "Qwen multi-query generator produced no valid queries."
            )

        return queries[: self.num_queries]

    @staticmethod
    def _parse_queries(text: str) -> list[str]:
        queries = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Remove common numbering formats:
            # 1. query
            # 1) query
            # - query
            # * query
            if len(line) >= 2 and line[0] in "-*":
                line = line[1:].strip()

            if ". " in line:
                prefix, remainder = line.split(". ", 1)

                if prefix.isdigit():
                    line = remainder.strip()

            if ") " in line:
                prefix, remainder = line.split(") ", 1)

                if prefix.isdigit():
                    line = remainder.strip()

            if line:
                queries.append(line)

        # Remove duplicates while preserving order.
        unique_queries = []

        for query in queries:
            normalized = query.casefold()

            if normalized not in {
                existing.casefold()
                for existing in unique_queries
            }:
                unique_queries.append(query)

        return unique_queries
