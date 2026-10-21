import pytest
import torch

import rag.qwen_model as qwen_model
from rag.qwen_model import QwenModelBundle

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, generated_ids, skip_special_tokens):
        return "  full answer  "


class FakeModel:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.generate_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs

        if self.error is not None:
            raise self.error

        return torch.tensor([[1, 2, 3, 4, 5]])


class FakeStreamer:
    def __init__(self, tokenizer, skip_prompt, skip_special_tokens):
        self.ended = False

    def __iter__(self):
        return iter(["Hello", " ", "world"])

    def end(self):
        self.ended = True


def make_bundle(monkeypatch, model: FakeModel) -> QwenModelBundle:
    class TokenizerFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return FakeTokenizer()

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return model

    monkeypatch.setattr(qwen_model, "AutoTokenizer", TokenizerFactory)
    monkeypatch.setattr(
        qwen_model,
        "AutoModelForCausalLM",
        ModelFactory,
    )
    monkeypatch.setattr(
        qwen_model,
        "TextIteratorStreamer",
        FakeStreamer,
    )

    return QwenModelBundle(model_name="test-model", device="cpu")


def test_generate_tokens_without_callback_does_not_stream(
    monkeypatch,
) -> None:
    model = FakeModel()
    bundle = make_bundle(monkeypatch, model)

    answer = bundle.generate_tokens(MESSAGES, max_new_tokens=8)

    assert answer == "full answer"
    assert "streamer" not in model.generate_kwargs


def test_generate_tokens_streams_pieces_in_order(monkeypatch) -> None:
    model = FakeModel()
    bundle = make_bundle(monkeypatch, model)
    pieces: list[str] = []

    answer = bundle.generate_tokens(
        MESSAGES,
        max_new_tokens=8,
        on_token=pieces.append,
    )

    assert pieces == ["Hello", " ", "world"]
    assert answer == "Hello world"
    assert "streamer" in model.generate_kwargs
    assert model.generate_kwargs["max_new_tokens"] == 8
    assert model.generate_kwargs["do_sample"] is False


def test_generate_tokens_streaming_propagates_model_errors(
    monkeypatch,
) -> None:
    bundle = make_bundle(
        monkeypatch,
        FakeModel(error=RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        bundle.generate_tokens(
            MESSAGES,
            max_new_tokens=8,
            on_token=lambda piece: None,
        )


def test_generate_tokens_rejects_non_positive_limit(
    monkeypatch,
) -> None:
    bundle = make_bundle(monkeypatch, FakeModel())

    with pytest.raises(ValueError, match="greater than zero"):
        bundle.generate_tokens(MESSAGES, max_new_tokens=0)
