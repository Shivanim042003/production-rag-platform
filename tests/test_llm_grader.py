from types import SimpleNamespace

import pytest

from rag.grader import RelevanceGrade
from rag.llm_grader import (
    LLMRelevanceScorer,
    RelevanceGradeSchema,
)


class FakeResponses:
    def __init__(self, parsed: RelevanceGradeSchema) -> None:
        self.parsed = parsed
        self.received_kwargs = None

    def parse(self, **kwargs):
        self.received_kwargs = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


class FakeOpenAI:
    def __init__(self, parsed: RelevanceGradeSchema) -> None:
        self.responses = FakeResponses(parsed)


def test_llm_scorer_returns_relevance_grade() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=True,
            score=0.95,
            reason="The chunk directly answers the query.",
        )
    )

    scorer = LLMRelevanceScorer(
        model_name="test-model",
        client=fake_client,
    )

    result = scorer.score(
        "How does Redis reduce database load?",
        "Redis can reduce database load by serving frequently requested values.",
    )

    assert isinstance(result, RelevanceGrade)
    assert result.relevant is True
    assert result.score == pytest.approx(0.95)
    assert result.reason == "The chunk directly answers the query."


def test_llm_scorer_sends_expected_model_and_input() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=True,
            score=0.9,
            reason="Relevant.",
        )
    )

    scorer = LLMRelevanceScorer(
        model_name="test-model",
        client=fake_client,
    )

    scorer.score(
        "What are PostgreSQL indexes used for?",
        "Indexes improve PostgreSQL query performance.",
    )

    request = fake_client.responses.received_kwargs

    assert request["model"] == "test-model"
    assert request["text_format"] is RelevanceGradeSchema

    messages = request["input"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What are PostgreSQL indexes used for?" in messages[1]["content"]
    assert "Indexes improve PostgreSQL query performance." in messages[1]["content"]


def test_llm_scorer_clamps_score_above_one() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=True,
            score=1.7,
            reason="Relevant.",
        )
    )

    scorer = LLMRelevanceScorer(client=fake_client)

    result = scorer.score(
        "query",
        "document",
    )

    assert result.score == 1.0


def test_llm_scorer_clamps_score_below_zero() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=False,
            score=-0.5,
            reason="Not relevant.",
        )
    )

    scorer = LLMRelevanceScorer(client=fake_client)

    result = scorer.score(
        "query",
        "document",
    )

    assert result.score == 0.0


def test_llm_scorer_rejects_empty_query() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=True,
            score=0.9,
            reason="Relevant.",
        )
    )

    scorer = LLMRelevanceScorer(client=fake_client)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        scorer.score("   ", "document")


def test_llm_scorer_rejects_empty_document() -> None:
    fake_client = FakeOpenAI(
        RelevanceGradeSchema(
            relevant=True,
            score=0.9,
            reason="Relevant.",
        )
    )

    scorer = LLMRelevanceScorer(client=fake_client)

    with pytest.raises(ValueError, match="Document text cannot be empty"):
        scorer.score("query", "   ")


def test_llm_scorer_raises_when_output_is_not_parsed() -> None:
    class EmptyResponses:
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed=None)

    fake_client = SimpleNamespace(
        responses=EmptyResponses()
    )

    scorer = LLMRelevanceScorer(client=fake_client)

    with pytest.raises(
        RuntimeError,
        match="LLM did not return a parsed relevance grade",
    ):
        scorer.score("query", "document")
