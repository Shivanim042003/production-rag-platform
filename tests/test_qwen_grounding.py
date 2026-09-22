import pytest
import torch

from rag.qwen_grounding import QwenGroundingScorer


class FakeTokenizer:
    def __init__(self, decision: str) -> None:
        self.decision = decision
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
        return self.decision


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

        return torch.tensor([[1, 2, 3, 4]])


def make_scorer(
    monkeypatch,
    decision: str,
    threshold: float = 0.5,
):
    tokenizer = FakeTokenizer(decision)
    model = FakeModel()

    class FakeTokenizerFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return tokenizer

    class FakeModelFactory:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            assert kwargs["dtype"] == torch.float32
            return model

    import rag.qwen_grounding as qwen_grounding

    monkeypatch.setattr(
        qwen_grounding,
        "AutoTokenizer",
        FakeTokenizerFactory,
    )

    monkeypatch.setattr(
        qwen_grounding,
        "AutoModelForCausalLM",
        FakeModelFactory,
    )

    scorer = QwenGroundingScorer(
        model_name="test-model",
        device="cpu",
        threshold=threshold,
    )

    return scorer, tokenizer, model


def test_qwen_grounding_initializes_model(monkeypatch) -> None:
    scorer, tokenizer, model = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    assert scorer.model_name == "test-model"
    assert scorer.device == "cpu"
    assert scorer.threshold == 0.5
    assert scorer.tokenizer is tokenizer
    assert scorer.model is model
    assert model.device == "cpu"
    assert model.eval_called is True


def test_qwen_grounding_returns_grounded_result(monkeypatch) -> None:
    scorer, _, _ = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    result = scorer.score(
        answer="PostgreSQL indexes improve query performance.",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    assert result.grounded is True
    assert result.score == 1.0
    assert "supported" in result.reason


def test_qwen_grounding_returns_ungrounded_result(monkeypatch) -> None:
    scorer, _, _ = make_scorer(
        monkeypatch,
        "GROUNDED: NO",
    )

    result = scorer.score(
        answer="PostgreSQL indexes reduce memory usage by 90%.",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    assert result.grounded is False
    assert result.score == 0.0
    assert "unsupported" in result.reason


def test_qwen_grounding_builds_expected_messages(monkeypatch) -> None:
    scorer, tokenizer, _ = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    answer = "PostgreSQL indexes improve query performance."
    context = [
        "Indexes improve PostgreSQL query performance.",
        "B-tree is a common PostgreSQL index type.",
    ]

    scorer.score(
        answer=answer,
        context=context,
    )

    messages = tokenizer.received_messages

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert "strict answer-grounding evaluator" in (
        messages[0]["content"]
    )

    assert answer in messages[1]["content"]
    assert context[0] in messages[1]["content"]
    assert context[1] in messages[1]["content"]


def test_qwen_grounding_passes_generation_settings(monkeypatch) -> None:
    scorer, _, model = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    scorer.score(
        answer="test answer",
        context=["test context"],
    )

    assert model.generate_kwargs["max_new_tokens"] == 8
    assert model.generate_kwargs["do_sample"] is False
    assert "input_ids" in model.generate_kwargs
    assert "attention_mask" in model.generate_kwargs


def test_qwen_grounding_rejects_invalid_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be between 0.0 and 1.0",
    ):
        QwenGroundingScorer(
            model_name="test-model",
            device="cpu",
            threshold=1.5,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_qwen_grounding_rejects_out_of_range_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be between 0.0 and 1.0",
    ):
        QwenGroundingScorer(
            model_name="test-model",
            device="cpu",
            threshold=threshold,
        )


def test_qwen_grounding_rejects_empty_answer(
    monkeypatch,
) -> None:
    scorer, _, _ = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        scorer.score(
            answer="   ",
            context=["context"],
        )


def test_qwen_grounding_rejects_empty_context(
    monkeypatch,
) -> None:
    scorer, _, _ = make_scorer(
        monkeypatch,
        "GROUNDED: YES",
    )

    with pytest.raises(
        ValueError,
        match="Context cannot be empty",
    ):
        scorer.score(
            answer="test answer",
            context=[],
        )


def test_qwen_grounding_rejects_invalid_model_decision(
    monkeypatch,
) -> None:
    scorer, _, _ = make_scorer(
        monkeypatch,
        "MAYBE",
    )

    with pytest.raises(
        RuntimeError,
        match="invalid grounding decision",
    ):
        scorer.score(
            answer="test answer",
            context=["test context"],
        )
