from dataclasses import dataclass
from typing import Protocol

from rag.models import RetrievalResult


@dataclass
class RelevanceGrade:
    relevant: bool
    score: float
    reason: str = ""


class RelevanceScorer(Protocol):
    def score(self, query: str, text: str) -> RelevanceGrade:
        ...


class RelevanceGrader:
    """Grades whether retrieved chunks are relevant to a query."""

    def __init__(self, scorer: RelevanceScorer) -> None:
        self.scorer = scorer

    def grade(
        self,
        query: str,
        result: RetrievalResult,
    ) -> RelevanceGrade:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.scorer.score(
            query,
            result.chunk.text,
        )

    def grade_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[tuple[RetrievalResult, RelevanceGrade]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return [
            (
                result,
                self.scorer.score(
                    query,
                    result.chunk.text,
                ),
            )
            for result in results
        ]

    @staticmethod
    def filter_relevant(
        graded_results: list[tuple[RetrievalResult, RelevanceGrade]],
    ) -> list[RetrievalResult]:
        return [
            result
            for result, grade in graded_results
            if grade.relevant
        ]

    @staticmethod
    def has_sufficient_context(
        graded_results: list[tuple[RetrievalResult, RelevanceGrade]],
        min_relevant: int = 1,
    ) -> bool:
        if min_relevant <= 0:
            raise ValueError("min_relevant must be greater than zero.")

        relevant_count = sum(
            1
            for _, grade in graded_results
            if grade.relevant
        )

        return relevant_count >= min_relevant
