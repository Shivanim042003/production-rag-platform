import pytest
import torch

from rag.local_generator import LocalGenerator


class FakeTokenizer:
    def __init__(self) -> None:
        self.received_prompt = None

    def __call__(self, prompt, return_tensors, truncation):
        self.received_prompt = prompt

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, output_ids, skip_special_tokens):
        return "PostgreSQL indexes improve query performance."


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs = None
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[4, 5, 6]])


def make_generator(monkeypatch):
    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeTokenizerFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return tokenizer

    class FakeModelFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return model

    import rag.local_generator as local_generator

    monkeypatch.setattr(
        local_generator,
        "AutoTokenizer",
        FakeTokenizerFactory,
    )

    monkeypatch.setattr(
        local_generator,
        "AutoModelForSeq2SeqLM",
        FakeModelFactory,
    )

    generator = LocalGenerator(
        model_name="test-model",
        device="cpu",
        max_new_tokens=64,
    )

    return generator, tokenizer, model


def test_local_generator_initializes_model(monkeypatch) -> None:
    generator, tokenizer, model = make_generator(monkeypatch)

    assert generator.model_name == "test-model"
    assert generator.device == "cpu"
    assert generator.max_new_tokens == 64
    assert tokenizer is generator.tokenizer
    assert model is generator.model
    assert model.device == "cpu"
    assert model.eval_called is True


def test_local_generator_returns_generation_result(monkeypatch) -> None:
    generator, _, _ = make_generator(monkeypatch)

    result = generator.generate(
        query="What are PostgreSQL indexes used for?",
        context=[
            "PostgreSQL indexes improve query performance."
        ],
    )

    assert result.answer == (
        "PostgreSQL indexes improve query performance."
    )
    assert result.model == "test-model"


def test_local_generator_builds_prompt_from_query_and_context(
    monkeypatch,
) -> None:
    generator, tokenizer, _ = make_generator(monkeypatch)

    generator.generate(
        query="What are PostgreSQL indexes used for?",
        context=[
            "Indexes help PostgreSQL locate rows efficiently."
        ],
    )

    assert "What are PostgreSQL indexes used for?" in (
        tokenizer.received_prompt
    )

    assert "Indexes help PostgreSQL locate rows efficiently." in (
        tokenizer.received_prompt
    )


def test_local_generator_passes_inputs_to_model(monkeypatch) -> None:
    generator, _, model = make_generator(monkeypatch)

    generator.generate(
        query="test query",
        context=["test context"],
    )

    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["do_sample"] is False
    assert "input_ids" in model.generate_kwargs
    assert "attention_mask" in model.generate_kwargs


def test_local_generator_rejects_invalid_max_new_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        LocalGenerator(
            model_name="test-model",
            device="cpu",
            max_new_tokens=0,
        )


def test_local_generator_raises_on_empty_generated_answer(
    monkeypatch,
) -> None:
    generator, tokenizer, _ = make_generator(monkeypatch)

    tokenizer.decode = lambda output_ids, skip_special_tokens: "   "

    with pytest.raises(
        RuntimeError,
        match="Local model returned an empty answer",
    ):
        generator.generate(
            query="test query",
            context=["test context"],
        )
