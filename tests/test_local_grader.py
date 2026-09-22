import numpy as np
import pytest

from rag.grader import RelevanceGrade
from rag.local_grader import LocalRelevanceScorer


class FakeCrossEncoderScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.received_pairs = None

    def predict(self, pairs) -> np.ndarray:
        self.received_pairs = pairs
        return np.asarray(self.scores, dtype=np.float32)


def test_local_grader_marks_score_above_threshold_relevant() -> None:
    scorer = FakeCrossEncoderScorer([2.5])
    grader = LocalRelevanceScorer(
        scorer=scorer,
        threshold=0.0,
    )

    result = grader.score(
        "How does Redis reduce database load?",
        "Redis serves frequently requested values from memory.",
    )

    assert isinstance(result, RelevanceGrade)
    assert result.relevant is True
    assert result.score == pytest.approx(2.5)
    assert "meets" in result.reason


def test_local_grader_marks_score_below_threshold_irrelevant() -> None:
    scorer = FakeCrossEncoderScorer([-1.5])
    grader = LocalRelevanceScorer(
        scorer=scorer,
        threshold=0.0,
    )

    result = grader.score(
        "How does Redis reduce database load?",
        "PostgreSQL uses MVCC for concurrency control.",
    )

    assert result.relevant is False
    assert result.score == pytest.approx(-1.5)
    assert "does not meet" in result.reason


def test_local_grader_uses_custom_threshold() -> None:
    scorer = FakeCrossEncoderScorer([1.0])
    grader = LocalRelevanceScorer(
        scorer=scorer,
        threshold=2.0,
    )

    result = grader.score(
        "query",
        "document",
    )

    assert result.relevant is False

    scorer = FakeCrossEncoderScorer([2.0])
    grader = LocalRelevanceScorer(
        scorer=scorer,
        threshold=2.0,
    )

    result = grader.score(
        "query",
        "document",
    )

    assert result.relevant is True


def test_local_grader_sends_query_and_document_pair() -> None:
    scorer = FakeCrossEncoderScorer([3.0])
    grader = LocalRelevanceScorer(scorer=scorer)

    grader.score(
        "What are PostgreSQL indexes?",
        "Indexes improve PostgreSQL query performance.",
    )

    assert scorer.received_pairs == [
        [
            "What are PostgreSQL indexes?",
            "Indexes improve PostgreSQL query performance.",
        ]
    ]


def test_local_grader_rejects_empty_query() -> None:
    scorer = FakeCrossEncoderScorer([1.0])
    grader = LocalRelevanceScorer(scorer=scorer)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        grader.score("   ", "document")


def test_local_grader_rejects_empty_document() -> None:
    scorer = FakeCrossEncoderScorer([1.0])
    grader = LocalRelevanceScorer(scorer=scorer)

    with pytest.raises(
        ValueError,
        match="Document text cannot be empty",
    ):
        grader.score("query", "   ")


def test_local_grader_rejects_unexpected_number_of_scores() -> None:
    scorer = FakeCrossEncoderScorer([1.0, 2.0])
    grader = LocalRelevanceScorer(scorer=scorer)

    with pytest.raises(
        RuntimeError,
        match="Expected exactly one relevance score",
    ):
        grader.score("query", "document")
