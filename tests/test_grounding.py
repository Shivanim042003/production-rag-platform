import pytest

from rag.grounding import GroundingChecker, GroundingResult


class FakeGroundingScorer:
    def __init__(self, result: GroundingResult) -> None:
        self.result = result
        self.received_args = None

    def score(self, answer: str, context: list[str]) -> GroundingResult:
        self.received_args = (answer, context)
        return self.result


def test_grounding_checker_returns_scorer_result() -> None:
    expected = GroundingResult(
        grounded=True,
        score=0.95,
        reason="The answer is supported by the context.",
    )

    scorer = FakeGroundingScorer(expected)
    checker = GroundingChecker(scorer)

    result = checker.check(
        answer="PostgreSQL indexes improve query performance.",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    assert result == expected


def test_grounding_checker_passes_answer_and_context_to_scorer() -> None:
    expected = GroundingResult(
        grounded=True,
        score=0.9,
        reason="Supported.",
    )

    scorer = FakeGroundingScorer(expected)
    checker = GroundingChecker(scorer)

    answer = "Redis can reduce database load."
    context = [
        "Redis can reduce database load by serving frequently "
        "requested values directly from memory."
    ]

    checker.check(answer, context)

    assert scorer.received_args == (answer, context)


def test_grounding_checker_handles_ungrounded_answer() -> None:
    expected = GroundingResult(
        grounded=False,
        score=0.15,
        reason="The claim is not supported by the context.",
    )

    scorer = FakeGroundingScorer(expected)
    checker = GroundingChecker(scorer)

    result = checker.check(
        answer="Redis reduces database load by 90%.",
        context=[
            "Redis can reduce database load by serving values from memory."
        ],
    )

    assert result.grounded is False
    assert result.score == pytest.approx(0.15)
    assert result.reason == (
        "The claim is not supported by the context."
    )


def test_grounding_checker_rejects_empty_answer() -> None:
    scorer = FakeGroundingScorer(
        GroundingResult(
            grounded=True,
            score=1.0,
        )
    )

    checker = GroundingChecker(scorer)

    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        checker.check(
            answer="   ",
            context=["context"],
        )


def test_grounding_checker_rejects_empty_context() -> None:
    scorer = FakeGroundingScorer(
        GroundingResult(
            grounded=True,
            score=1.0,
        )
    )

    checker = GroundingChecker(scorer)

    with pytest.raises(
        ValueError,
        match="Context cannot be empty",
    ):
        checker.check(
            answer="test answer",
            context=[],
        )


def test_grounding_checker_preserves_context_order() -> None:
    expected = GroundingResult(
        grounded=True,
        score=0.8,
        reason="Supported.",
    )

    scorer = FakeGroundingScorer(expected)
    checker = GroundingChecker(scorer)

    context = [
        "First context.",
        "Second context.",
        "Third context.",
    ]

    checker.check(
        answer="test answer",
        context=context,
    )

    assert scorer.received_args[1] == context
