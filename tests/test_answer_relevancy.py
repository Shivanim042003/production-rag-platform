import pytest

from rag.answer_relevancy import (
    AnswerRelevancyResult,
    EmbeddingAnswerRelevancyScorer,
)


def test_valid_result():
    result = AnswerRelevancyResult(
        relevant=True,
        score=1.0,
        reason="Directly answers the question.",
    )

    assert result.relevant is True
    assert result.score == 1.0


def test_partial_result():
    result = AnswerRelevancyResult(
        relevant=True,
        score=0.5,
        reason="Partially answers the question.",
    )

    assert result.relevant is True
    assert result.score == 0.5


def test_invalid_score_above_one():
    with pytest.raises(ValueError):
        AnswerRelevancyResult(
            relevant=True,
            score=1.5,
        )


def test_invalid_negative_score():
    with pytest.raises(ValueError):
        AnswerRelevancyResult(
            relevant=False,
            score=-0.1,
        )


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        EmbeddingAnswerRelevancyScorer(
            relevant_threshold=0.30,
            partially_relevant_threshold=0.50,
        )