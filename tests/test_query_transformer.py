from types import SimpleNamespace

import torch

from rag.graph_nodes import QueryTransformNode
from rag.graph_state import RAGState
from rag.query_transformer import (
    IdentityQueryTransformer,
    QwenQueryTransformer,
)


def test_identity_transformer():
    transformer = IdentityQueryTransformer()

    assert transformer.transform(
        "  What are PostgreSQL indexes used for?  "
    ) == "What are PostgreSQL indexes used for?"


def test_identity_transformer_rejects_empty_query():
    transformer = IdentityQueryTransformer()

    try:
        transformer.transform("   ")
        assert False
    except ValueError:
        pass


def test_query_transform_node():
    transformer = IdentityQueryTransformer()
    node = QueryTransformNode(transformer)

    state = RAGState(
        query="What are PostgreSQL indexes used for?"
    )

    result = node(state)

    assert result["transformed_queries"] == [
        "What are PostgreSQL indexes used for?"
    ]


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, generated_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "PostgreSQL query performance and slow query execution"


class FakeModel:
    def generate(self, **inputs):
        return torch.tensor([
            [1, 2, 3, 4, 5, 6]
        ])


def test_qwen_query_transformer():
    transformer = QwenQueryTransformer(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_name="fake-qwen",
    )

    result = transformer.transform(
        "Why is my PostgreSQL database slow?"
    )

    assert result == (
        "PostgreSQL query performance and slow query execution"
    )


def test_qwen_query_transformer_rejects_empty_query():
    transformer = QwenQueryTransformer(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_name="fake-qwen",
    )

    try:
        transformer.transform("   ")
        assert False
    except ValueError:
        pass
