import torch

from rag.multi_query import (
    IdentityMultiQueryGenerator,
    QwenMultiQueryGenerator,
)


def test_identity_multi_query_generator():
    generator = IdentityMultiQueryGenerator()

    result = generator.generate(
        "What are PostgreSQL indexes used for?"
    )

    assert result == [
        "What are PostgreSQL indexes used for?"
    ]


def test_identity_multi_query_generator_strips_query():
    generator = IdentityMultiQueryGenerator()

    result = generator.generate(
        "   What are PostgreSQL indexes used for?   "
    )

    assert result == [
        "What are PostgreSQL indexes used for?"
    ]


def test_identity_multi_query_generator_rejects_empty_query():
    generator = IdentityMultiQueryGenerator()

    try:
        generator.generate("   ")
        assert False
    except ValueError:
        pass


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

    def decode(
        self,
        generated_ids,
        skip_special_tokens=True,
    ):
        assert skip_special_tokens is True

        return (
            "1. PostgreSQL query performance problems\n"
            "2. PostgreSQL slow query execution causes\n"
            "3. PostgreSQL performance optimization factors"
        )


class FakeModel:
    def generate(self, **inputs):
        return torch.tensor([
            [1, 2, 3, 4, 5, 6]
        ])


def test_qwen_multi_query_generator():
    generator = QwenMultiQueryGenerator(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_name="fake-qwen",
        num_queries=3,
    )

    result = generator.generate(
        "Why is my PostgreSQL database slow?"
    )

    assert result == [
        "PostgreSQL query performance problems",
        "PostgreSQL slow query execution causes",
        "PostgreSQL performance optimization factors",
    ]


def test_qwen_multi_query_parser_removes_duplicates():
    result = QwenMultiQueryGenerator._parse_queries(
        "1. PostgreSQL performance\n"
        "2. PostgreSQL performance\n"
        "3. PostgreSQL query optimization"
    )

    assert result == [
        "PostgreSQL performance",
        "PostgreSQL query optimization",
    ]


def test_qwen_multi_query_parser_supports_bullets():
    result = QwenMultiQueryGenerator._parse_queries(
        "- PostgreSQL performance\n"
        "* PostgreSQL query optimization"
    )

    assert result == [
        "PostgreSQL performance",
        "PostgreSQL query optimization",
    ]


def test_qwen_multi_query_generator_rejects_empty_query():
    generator = QwenMultiQueryGenerator(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_name="fake-qwen",
    )

    try:
        generator.generate("   ")
        assert False
    except ValueError:
        pass


def test_qwen_multi_query_generator_rejects_invalid_num_queries():
    try:
        QwenMultiQueryGenerator(
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            device="cpu",
            model_name="fake-qwen",
            num_queries=0,
        )
        assert False
    except ValueError:
        pass
