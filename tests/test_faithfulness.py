import pytest
import torch

from rag.faithfulness import QwenFaithfulnessScorer


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
    ):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        return "YES"


class FakeModel:
    def __init__(self, output_text: str = "YES"):
        self.output_text = output_text
        self.last_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.last_kwargs = kwargs

        # Three prompt tokens + one generated token.
        return torch.tensor([[1, 2, 3, 4]])


def test_faithfulness_scorer_rejects_empty_answer() -> None:
    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        scorer.score(
            answer="",
            context=["Supporting context."],
        )


def test_faithfulness_scorer_rejects_empty_context() -> None:
    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    with pytest.raises(
        ValueError,
        match="Context cannot be empty",
    ):
        scorer.score(
            answer="Some answer.",
            context=[],
        )


def test_faithfulness_scorer_rejects_empty_context_item() -> None:
    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    with pytest.raises(
        ValueError,
        match="Context cannot contain empty strings",
    ):
        scorer.score(
            answer="Some answer.",
            context=[""],
        )


def test_faithfulness_scorer_returns_one_for_yes() -> None:
    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    scorer.device = "cpu"
    scorer.tokenizer = FakeTokenizer()
    scorer.model = FakeModel()

    score = scorer.score(
        answer="PostgreSQL indexes improve query performance.",
        context=[
            "PostgreSQL indexes improve query performance."
        ],
    )

    assert score == 1.0


def test_faithfulness_scorer_uses_generated_tokens_only() -> None:
    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    scorer.device = "cpu"
    scorer.tokenizer = FakeTokenizer()
    scorer.model = FakeModel()

    score = scorer.score(
        answer="PostgreSQL indexes improve query performance.",
        context=[
            "PostgreSQL indexes improve query performance."
        ],
    )

    assert score == 1.0
    assert scorer.model.last_kwargs is not None
    assert scorer.model.last_kwargs["max_new_tokens"] == 8
    assert scorer.model.last_kwargs["do_sample"] is False


def test_faithfulness_scorer_rejects_unrecognized_model_output() -> None:
    class InvalidTokenizer(FakeTokenizer):
        def decode(
            self,
            token_ids,
            skip_special_tokens=True,
        ):
            return "MAYBE"

    scorer = QwenFaithfulnessScorer.__new__(
        QwenFaithfulnessScorer
    )

    scorer.device = "cpu"
    scorer.tokenizer = InvalidTokenizer()
    scorer.model = FakeModel()

    with pytest.raises(
        RuntimeError,
        match="unrecognized response",
    ):
        scorer.score(
            answer="Some answer.",
            context=["Some context."],
        )