from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class GroundingResult:
    grounded: bool
    score: float
    reason: str = ""


class GroundingScorer(Protocol):
    def score(
        self,
        answer: str,
        context: Sequence[str],
    ) -> GroundingResult:
        ...


class GroundingChecker:
    """Checks whether an answer is supported by retrieved context."""

    def __init__(
        self,
        scorer: GroundingScorer,
    ) -> None:
        self.scorer = scorer

    def check(
        self,
        answer: str,
        context: Sequence[str],
    ) -> GroundingResult:
        if not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not context:
            raise ValueError("Context cannot be empty.")

        return self.scorer.score(
            answer,
            context,
        )
