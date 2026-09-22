import pytest
import torch

from rag.qwen_generator import QwenLocalGenerator


class FakeTokenizer:
    def __init__(self) -> None:
        self.received_messages = None

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        self.received_messages = messages

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
        skip_special_tokens,
    ):
        return "PostgreSQL indexes improve query performance."


class FakeModel:
    def __init__(self) -> None:
        self.device = None
        self.eval_called = False
        self.generate_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs

        return torch.tensor([[1, 2, 3, 4, 5]])


def make_generator(monkeypatch):
    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeTokenizerFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return tokenizer

    class FakeModelFactory:
        @classmethod
        def from_pretrained(
            cls,
            model_name,
            **kwargs,
        ):
            assert kwargs["dtype"] == torch.float32
            return model

    import rag.qwen_generator as qwen_generator

    monkeypatch.setattr(
        qwen_generator,
        "AutoTokenizer",
        FakeTokenizerFactory,
    )

    monkeypatch.setattr(
        qwen_generator,
        "AutoModelForCausalLM",
        FakeModelFactory,
    )

    generator = QwenLocalGenerator(
        model_name="test-model",
        device="cpu",
        max_new_tokens=64,
    )

    return generator, tokenizer, model


def test_qwen_generator_initializes_model(monkeypatch) -> None:
    generator, tokenizer, model = make_generator(monkeypatch)

    assert generator.model_name == "test-model"
    assert generator.device == "cpu"
    assert generator.max_new_tokens == 64
    assert generator.tokenizer is tokenizer
    assert generator.model is model
    assert model.device == "cpu"
    assert model.eval_called is True


def test_qwen_generator_returns_generation_result(
    monkeypatch,
) -> None:
    generator, _, _ = make_generator(monkeypatch)

    result = generator.generate(
        query="What are PostgreSQL indexes used for?",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    assert result.answer == (
        "PostgreSQL indexes improve query performance."
    )
    assert result.model == "test-model"


def test_qwen_generator_builds_expected_messages(
    monkeypatch,
) -> None:
    generator, tokenizer, _ = make_generator(monkeypatch)

    generator.generate(
        query="What are PostgreSQL indexes used for?",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    messages = tokenizer.received_messages

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert "document question-answering assistant" in (
        messages[0]["content"]
    )

    assert "What are PostgreSQL indexes used for?" in (
        messages[1]["content"]
    )

    assert "Indexes improve PostgreSQL query performance." in (
        messages[1]["content"]
    )


def test_qwen_generator_passes_generation_settings_to_model(
    monkeypatch,
) -> None:
    generator, _, model = make_generator(monkeypatch)

    generator.generate(
        query="test query",
        context=["test context"],
    )

    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["do_sample"] is False
    assert "input_ids" in model.generate_kwargs
    assert "attention_mask" in model.generate_kwargs


def test_qwen_generator_rejects_invalid_max_new_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        QwenLocalGenerator(
            model_name="test-model",
            device="cpu",
            max_new_tokens=0,
        )


def test_qwen_generator_raises_on_empty_answer(
    monkeypatch,
) -> None:
    generator, tokenizer, _ = make_generator(monkeypatch)

    tokenizer.decode = (
        lambda generated_ids, skip_special_tokens: "   "
    )

    with pytest.raises(
        RuntimeError,
        match="Qwen model returned an empty answer",
    ):
        generator.generate(
            query="test query",
            context=["test context"],
        )


def test_qwen_generator_uses_only_generated_tokens(
    monkeypatch,
) -> None:
    generator, tokenizer, model = make_generator(monkeypatch)

    generator.generate(
        query="test query",
        context=["test context"],
    )

    assert model.generate_kwargs["input_ids"].shape[-1] == 3
    assert tokenizer.received_messages is not None
