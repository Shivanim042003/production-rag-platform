import pytest

from rag.grader import RelevanceGrade, RelevanceGrader
from rag.models import DocumentChunk, RetrievalResult


class FakeRelevanceScorer:
    def __init__(self, grades: list[RelevanceGrade]) -> None:
        self.grades = grades
        self.calls: list[tuple[str, str]] = []

    def score(self, query: str, text: str) -> RelevanceGrade:
        self.calls.append((query, text))
        return self.grades[len(self.calls) - 1]


def make_result(
    chunk_id: str = "chunk_001",
    text: str = "PostgreSQL supports indexes.",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc_001",
            text=text,
            source="test.txt",
        ),
        score=0.9,
        retriever="reranker",
        metadata={"rank": 1},
    )


def test_grade_returns_scorer_result() -> None:
    expected = RelevanceGrade(
        relevant=True,
        score=0.95,
        reason="Directly answers the query.",
    )
    scorer = FakeRelevanceScorer([expected])
    grader = RelevanceGrader(scorer)

    result = grader.grade(
        "What are PostgreSQL indexes?",
        make_result(),
    )

    assert result == expected


def test_grade_passes_query_and_chunk_text_to_scorer() -> None:
    scorer = FakeRelevanceScorer(
        [
            RelevanceGrade(
                relevant=True,
                score=1.0,
            )
        ]
    )
    grader = RelevanceGrader(scorer)

    result = make_result(
        chunk_id="chunk_007",
        text="Indexes improve PostgreSQL query performance.",
    )

    grader.grade(
        "How do PostgreSQL indexes improve queries?",
        result,
    )

    assert scorer.calls == [
        (
            "How do PostgreSQL indexes improve queries?",
            "Indexes improve PostgreSQL query performance.",
        )
    ]


def test_grade_results_grades_all_results() -> None:
    grades = [
        RelevanceGrade(
            relevant=True,
            score=0.9,
            reason="Relevant.",
        ),
        RelevanceGrade(
            relevant=False,
            score=0.1,
            reason="Unrelated.",
        ),
        RelevanceGrade(
            relevant=True,
            score=0.8,
            reason="Relevant.",
        ),
    ]

    scorer = FakeRelevanceScorer(grades)
    grader = RelevanceGrader(scorer)

    results = [
        make_result(
            chunk_id="chunk_001",
            text="PostgreSQL indexes improve query performance.",
        ),
        make_result(
            chunk_id="chunk_002",
            text="Redis is an in-memory cache.",
        ),
        make_result(
            chunk_id="chunk_003",
            text="PostgreSQL query planning uses statistics.",
        ),
    ]

    graded = grader.grade_results(
        "How does PostgreSQL improve query performance?",
        results,
    )

    assert len(graded) == 3
    assert graded[0] == (results[0], grades[0])
    assert graded[1] == (results[1], grades[1])
    assert graded[2] == (results[2], grades[2])


def test_grade_results_preserves_result_order() -> None:
    grades = [
        RelevanceGrade(relevant=False, score=0.2),
        RelevanceGrade(relevant=True, score=0.9),
        RelevanceGrade(relevant=False, score=0.1),
    ]

    scorer = FakeRelevanceScorer(grades)
    grader = RelevanceGrader(scorer)

    results = [
        make_result(chunk_id="chunk_a"),
        make_result(chunk_id="chunk_b"),
        make_result(chunk_id="chunk_c"),
    ]

    graded = grader.grade_results("database query", results)

    assert [result.chunk.chunk_id for result, _ in graded] == [
        "chunk_a",
        "chunk_b",
        "chunk_c",
    ]


def test_grade_results_returns_empty_for_empty_results() -> None:
    scorer = FakeRelevanceScorer([])
    grader = RelevanceGrader(scorer)

    assert grader.grade_results("database query", []) == []
    assert scorer.calls == []


def test_grade_rejects_empty_query() -> None:
    scorer = FakeRelevanceScorer(
        [RelevanceGrade(relevant=True, score=1.0)]
    )
    grader = RelevanceGrader(scorer)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        grader.grade("   ", make_result())


def test_grade_results_rejects_empty_query() -> None:
    scorer = FakeRelevanceScorer([])
    grader = RelevanceGrader(scorer)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        grader.grade_results("   ", [])


def test_grade_does_not_modify_retrieval_result() -> None:
    scorer = FakeRelevanceScorer(
        [
            RelevanceGrade(
                relevant=True,
                score=0.9,
            )
        ]
    )
    grader = RelevanceGrader(scorer)

    result = make_result()

    original_score = result.score
    original_retriever = result.retriever
    original_metadata = result.metadata.copy()

    grader.grade("PostgreSQL indexes", result)

    assert result.score == original_score
    assert result.retriever == original_retriever
    assert result.metadata == original_metadata
 

def test_filter_relevant_keeps_only_relevant_results() -> None:
    relevant_result = make_result(
        chunk_id="chunk_001",
        text="PostgreSQL indexes improve query performance.",
    )
    irrelevant_result = make_result(
        chunk_id="chunk_002",
        text="Redis is an in-memory cache.",
    )

    graded_results = [
        (
            relevant_result,
            RelevanceGrade(
                relevant=True,
                score=0.95,
            ),
        ),
        (
            irrelevant_result,
            RelevanceGrade(
                relevant=False,
                score=0.10,
            ),
        ),
    ]

    scorer = FakeRelevanceScorer([])
    grader = RelevanceGrader(scorer)

    filtered = grader.filter_relevant(graded_results)

    assert filtered == [relevant_result]


def test_filter_relevant_preserves_original_order() -> None:
    results = [
        make_result(chunk_id="chunk_001"),
        make_result(chunk_id="chunk_002"),
        make_result(chunk_id="chunk_003"),
    ]

    graded_results = [
        (results[0], RelevanceGrade(relevant=True, score=0.9)),
        (results[1], RelevanceGrade(relevant=False, score=0.2)),
        (results[2], RelevanceGrade(relevant=True, score=0.8)),
    ]

    grader = RelevanceGrader(FakeRelevanceScorer([]))

    filtered = grader.filter_relevant(graded_results)

    assert [result.chunk.chunk_id for result in filtered] == [
        "chunk_001",
        "chunk_003",
    ]


def test_has_sufficient_context_when_minimum_is_met() -> None:
    graded_results = [
        (
            make_result(chunk_id="chunk_001"),
            RelevanceGrade(relevant=True, score=0.9),
        ),
        (
            make_result(chunk_id="chunk_002"),
            RelevanceGrade(relevant=True, score=0.8),
        ),
        (
            make_result(chunk_id="chunk_003"),
            RelevanceGrade(relevant=False, score=0.1),
        ),
    ]

    grader = RelevanceGrader(FakeRelevanceScorer([]))

    assert grader.has_sufficient_context(
        graded_results,
        min_relevant=2,
    ) is True


def test_has_sufficient_context_when_minimum_is_not_met() -> None:
    graded_results = [
        (
            make_result(chunk_id="chunk_001"),
            RelevanceGrade(relevant=True, score=0.9),
        ),
        (
            make_result(chunk_id="chunk_002"),
            RelevanceGrade(relevant=False, score=0.1),
        ),
    ]

    grader = RelevanceGrader(FakeRelevanceScorer([]))

    assert grader.has_sufficient_context(
        graded_results,
        min_relevant=2,
    ) is False


def test_has_sufficient_context_returns_false_for_empty_results() -> None:
    grader = RelevanceGrader(FakeRelevanceScorer([]))

    assert grader.has_sufficient_context(
        [],
        min_relevant=1,
    ) is False


def test_has_sufficient_context_rejects_invalid_minimum() -> None:
    grader = RelevanceGrader(FakeRelevanceScorer([]))

    with pytest.raises(
        ValueError,
        match="min_relevant must be greater than zero",
    ):
        grader.has_sufficient_context(
            [],
            min_relevant=0,
        )
