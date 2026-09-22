from rag.graph import build_rag_graph
from rag.graph_nodes import GradeNode, RerankNode, RetrieveNode
from rag.grader import RelevanceGrade
from rag.graph_state import RAGState
from rag.models import DocumentChunk, RetrievalResult


def make_result(
    chunk_id: str,
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc_001",
            text=text,
            source="test.txt",
        ),
        score=1.0,
        retriever="test",
    )


class FakeRetriever:
    def retrieve(self, **kwargs):
        return [
            make_result(
                "chunk_001",
                "PostgreSQL supports indexes.",
            ),
            make_result(
                "chunk_002",
                "PostgreSQL supports transactions.",
            ),
        ]


class FakeReranker:
    def rerank(self, **kwargs):
        results = kwargs["results"]
        return list(reversed(results))


class FakeGrader:
    def __init__(self, sufficient: bool = True) -> None:
        self.sufficient = sufficient

    def grade_results(self, **kwargs):
        results = kwargs["results"]

        return [
            (
                result,
                RelevanceGrade(
                    relevant=self.sufficient,
                    score=0.9 if self.sufficient else 0.1,
                    reason=(
                        "Relevant."
                        if self.sufficient
                        else "Not relevant."
                    ),
                ),
            )
            for result in results
        ]

    def has_sufficient_context(
        self,
        graded_results,
        min_relevant,
    ):
        return self.sufficient


def make_graph(
    sufficient: bool = True,
    max_retries: int = 2,
):
    retrieve_node = RetrieveNode(
        retriever=FakeRetriever(),
        top_k=2,
        dense_k=2,
        bm25_k=2,
    )

    rerank_node = RerankNode(
        reranker=FakeReranker(),
        top_k=2,
    )

    grade_node = GradeNode(
        grader=FakeGrader(sufficient=sufficient),
        min_relevant=1,
    )

    return build_rag_graph(
        retrieve_node,
        rerank_node,
        grade_node,
        max_retries=max_retries,
    )


def test_graph_builds_successfully() -> None:
    graph = make_graph()

    assert graph is not None


def test_graph_runs_retrieve_rerank_grade_flow() -> None:
    graph = make_graph()

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert len(state["candidates"]) == 2
    assert len(state["reranked_results"]) == 2
    assert len(state["graded_results"]) == 2
    assert state["sufficient_context"] is True

    assert state["context"] == [
        "PostgreSQL supports transactions.",
        "PostgreSQL supports indexes.",
    ]


def test_graph_preserves_query() -> None:
    graph = make_graph()

    state = graph.invoke(
        RAGState(
            query="How do PostgreSQL indexes work?"
        )
    )

    assert state["query"] == (
        "How do PostgreSQL indexes work?"
    )


def test_graph_ends_after_grading() -> None:
    graph = make_graph()

    state = graph.invoke(
        RAGState(
            query="PostgreSQL"
        )
    )

    assert state["answer"] is None
    assert state["grounded"] is None
    assert state["retry_count"] == 0


def test_graph_routes_to_generate_when_context_is_sufficient() -> None:
    graph = make_graph(sufficient=True)

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 0
    assert state["sufficient_context"] is True
    assert state["answer"] is None


def test_graph_retries_when_context_is_insufficient() -> None:
    graph = make_graph(
        sufficient=False,
        max_retries=2,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 2
    assert state["sufficient_context"] is False


def test_graph_routes_to_fallback_after_max_retries() -> None:
    graph = make_graph(
        sufficient=False,
        max_retries=2,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 2
    assert state["answer"] == "I don't have enough information."
    assert state["sufficient_context"] is False


def test_graph_routes_to_fallback_when_retries_are_disabled() -> None:
    graph = make_graph(
        sufficient=False,
        max_retries=0,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 0
    assert state["answer"] == "I don't have enough information."
    assert state["sufficient_context"] is False
