from rag.graph_nodes import GradeNode, RerankNode, RetrieveNode
from rag.grader import RelevanceGrade
from rag.graph_state import RAGState
from rag.models import DocumentChunk, RetrievalResult


def make_result(
    chunk_id: str,
    text: str,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc_001",
            text=text,
            source="test.txt",
        ),
        score=score,
        retriever="test",
        metadata={"rank": 1},
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.received_args = None

    def retrieve(self, **kwargs):
        self.received_args = kwargs
        return self.results


class FakeReranker:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.received_args = None

    def rerank(self, **kwargs):
        self.received_args = kwargs
        return self.results


class FakeGrader:
    def __init__(
        self,
        graded_results: list[tuple[RetrievalResult, RelevanceGrade]],
        sufficient: bool,
    ) -> None:
        self.graded_results = graded_results
        self.sufficient = sufficient
        self.grade_results_args = None
        self.has_sufficient_context_args = None

    def grade_results(self, **kwargs):
        self.grade_results_args = kwargs
        return self.graded_results

    def has_sufficient_context(self, graded_results, min_relevant):
        self.has_sufficient_context_args = (
            graded_results,
            min_relevant,
        )
        return self.sufficient


def test_retrieve_node_updates_candidates() -> None:
    results = [
        make_result("chunk_001", "PostgreSQL supports indexes."),
        make_result("chunk_002", "PostgreSQL supports transactions."),
    ]

    retriever = FakeRetriever(results)
    node = RetrieveNode(
        retriever=retriever,
        top_k=10,
        dense_k=20,
        bm25_k=20,
    )

    state = RAGState(
        query="PostgreSQL indexes"
    )

    update = node(state)

    assert update == {"candidates": results}

    assert retriever.received_args == {
        "query": "PostgreSQL indexes",
        "top_k": 10,
        "dense_k": 20,
        "bm25_k": 20,
    }


def test_retrieve_node_does_not_modify_original_state() -> None:
    results = [
        make_result(
            "chunk_001",
            "PostgreSQL supports indexes.",
        )
    ]

    retriever = FakeRetriever(results)
    node = RetrieveNode(retriever)

    state = RAGState(
        query="PostgreSQL indexes"
    )

    update = node(state)

    assert state.candidates == []
    assert update["candidates"] == results


def test_rerank_node_updates_reranked_results() -> None:
    candidates = [
        make_result(
            "chunk_001",
            "PostgreSQL supports indexes.",
            score=0.8,
        ),
        make_result(
            "chunk_002",
            "PostgreSQL supports transactions.",
            score=0.7,
        ),
    ]

    reranked_results = [
        make_result(
            "chunk_002",
            "PostgreSQL supports transactions.",
            score=5.0,
        ),
        make_result(
            "chunk_001",
            "PostgreSQL supports indexes.",
            score=4.0,
        ),
    ]

    reranker = FakeReranker(reranked_results)

    node = RerankNode(
        reranker=reranker,
        top_k=5,
    )

    state = RAGState(
        query="PostgreSQL",
        candidates=candidates,
    )

    update = node(state)

    assert update == {
        "reranked_results": reranked_results
    }

    assert reranker.received_args == {
        "query": "PostgreSQL",
        "results": candidates,
        "top_k": 5,
    }


def test_grade_node_updates_grades_context_and_sufficiency() -> None:
    relevant = make_result(
        "chunk_001",
        "Indexes improve query performance.",
    )

    irrelevant = make_result(
        "chunk_002",
        "Redis is an in-memory cache.",
    )

    grades = [
        (
            relevant,
            RelevanceGrade(
                relevant=True,
                score=0.95,
                reason="Relevant.",
            ),
        ),
        (
            irrelevant,
            RelevanceGrade(
                relevant=False,
                score=0.10,
                reason="Not relevant.",
            ),
        ),
    ]

    grader = FakeGrader(
        graded_results=grades,
        sufficient=True,
    )

    node = GradeNode(
        grader=grader,
        min_relevant=1,
    )

    state = RAGState(
        query="How do indexes improve queries?",
        reranked_results=[
            relevant,
            irrelevant,
        ],
    )

    update = node(state)

    assert update["graded_results"] == grades
    assert update["context"] == [
        "Indexes improve query performance."
    ]
    assert update["sufficient_context"] is True

    assert grader.grade_results_args == {
        "query": "How do indexes improve queries?",
        "results": [
            relevant,
            irrelevant,
        ],
    }

    assert grader.has_sufficient_context_args == (
        grades,
        1,
    )


def test_grade_node_returns_empty_context_when_nothing_is_relevant() -> None:
    result_1 = make_result(
        "chunk_001",
        "Redis provides caching.",
    )

    result_2 = make_result(
        "chunk_002",
        "Redis supports key-value structures.",
    )

    grades = [
        (
            result_1,
            RelevanceGrade(
                relevant=False,
                score=0.1,
                reason="Not relevant.",
            ),
        ),
        (
            result_2,
            RelevanceGrade(
                relevant=False,
                score=0.2,
                reason="Not relevant.",
            ),
        ),
    ]

    grader = FakeGrader(
        graded_results=grades,
        sufficient=False,
    )

    node = GradeNode(
        grader=grader,
        min_relevant=1,
    )

    state = RAGState(
        query="How does PostgreSQL use MVCC?",
        reranked_results=[
            result_1,
            result_2,
        ],
    )

    update = node(state)

    assert update["graded_results"] == grades
    assert update["context"] == []
    assert update["sufficient_context"] is False


def test_grade_node_preserves_relevant_result_order() -> None:
    first = make_result(
        "chunk_001",
        "First relevant chunk.",
    )

    second = make_result(
        "chunk_002",
        "Second relevant chunk.",
    )

    third = make_result(
        "chunk_003",
        "Irrelevant chunk.",
    )

    grades = [
        (
            first,
            RelevanceGrade(
                relevant=True,
                score=0.9,
            ),
        ),
        (
            second,
            RelevanceGrade(
                relevant=True,
                score=0.8,
            ),
        ),
        (
            third,
            RelevanceGrade(
                relevant=False,
                score=0.1,
            ),
        ),
    ]

    grader = FakeGrader(
        graded_results=grades,
        sufficient=True,
    )

    node = GradeNode(
        grader=grader,
        min_relevant=2,
    )

    state = RAGState(
        query="test query",
        reranked_results=[
            first,
            second,
            third,
        ],
    )

    update = node(state)

    assert update["context"] == [
        "First relevant chunk.",
        "Second relevant chunk.",
    ]
