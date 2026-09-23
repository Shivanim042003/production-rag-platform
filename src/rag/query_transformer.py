from typing import Protocol

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class QueryTransformer(Protocol):
    def transform(self, query: str) -> str:
        """Transform a user query into a retrieval-oriented query."""
        ...


class IdentityQueryTransformer:
    """
    Baseline transformer.

    Returns the original query unchanged.
    Useful as a control when evaluating whether
    query transformation improves retrieval.
    """

    def transform(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return query.strip()


class QwenQueryTransformer:
    """
    Transforms a user query into a concise retrieval-oriented
    query using an already-loaded Qwen causal language model.

    The model and tokenizer are injected so that the Qwen model
    does not need to be loaded a second time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: str,
        model_name: str,
        max_new_tokens: int = 64,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def transform(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite user questions for document retrieval. "
                    "Return exactly one concise search query. "
                    "Preserve the original meaning. "
                    "Include important technical terms. "
                    "Do not answer the question. "
                    "Do not add facts that are not implied by the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite this question as a retrieval query.\n\n"
                    f"Question: {query.strip()}\n\n"
                    "Retrieval query:"
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

        transformed_query = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not transformed_query:
            raise RuntimeError(
                "Qwen query transformer returned an empty query."
            )

        return transformed_query
