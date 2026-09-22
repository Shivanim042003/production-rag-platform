from rag.generator import GenerationResult
from rag.graph import build_rag_graph
from rag.graph_nodes import (
    GenerateNode,
    GradeNode,
    GroundingNode,
    RerankNode,
    RetrieveNode,
)
from rag.grader import RelevanceGrade
from rag.graph_state import RAGState
from rag.grounding import GroundingResult
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


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> GenerationResult:
        self.calls.append((query, context))

        return GenerationResult(
            answer="PostgreSQL indexes improve query performance.",
            model="fake-model",
        )


class FakeGroundingChecker:
    def __init__(self, grounded: bool = True) -> None:
        self.grounded = grounded
        self.calls = []

    def check(
        self,
        answer: str,
        context: list[str],
    ) -> GroundingResult:
        self.calls.append((answer, context))

        return GroundingResult(
            grounded=self.grounded,
            score=1.0 if self.grounded else 0.0,
            reason=(
                "Supported."
                if self.grounded
                else "Unsupported."
            ),
        )


def make_graph(
    sufficient: bool = True,
    grounded: bool = True,
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
        grader=FakeGrader(
            sufficient=sufficient,
        ),
        min_relevant=1,
    )

    generator = FakeGenerator()

    generate_node = GenerateNode(
        generator=generator,
    )

    grounding_checker = FakeGroundingChecker(
        grounded=grounded,
    )

    grounding_node = GroundingNode(
        checker=grounding_checker,
    )

    graph = build_rag_graph(
        retrieve_node=retrieve_node,
        rerank_node=rerank_node,
        grade_node=grade_node,
        generate_node=generate_node,
        grounding_node=grounding_node,
        max_retries=max_retries,
    )

    return graph, generator, grounding_checker


def test_graph_builds_successfully() -> None:
    graph, _, _ = make_graph()

    assert graph is not None


def test_graph_runs_full_retrieve_rerank_grade_generate_ground_flow() -> None:
    graph, generator, grounding_checker = make_graph(
        sufficient=True,
        grounded=True,
    )

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

    assert state["answer"] == (
        "PostgreSQL indexes improve query performance."
    )

    assert state["grounded"] is True

    assert len(generator.calls) == 1
    assert len(grounding_checker.calls) == 1


def test_graph_preserves_query() -> None:
    graph, _, _ = make_graph()

    state = graph.invoke(
        RAGState(
            query="How do PostgreSQL indexes work?"
        )
    )

    assert state["query"] == (
        "How do PostgreSQL indexes work?"
    )


def test_graph_generates_only_when_context_is_sufficient() -> None:
    graph, generator, grounding_checker = make_graph(
        sufficient=True,
        grounded=True,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["sufficient_context"] is True
    assert state["answer"] is not None
    assert state["grounded"] is True
    assert len(generator.calls) == 1
    assert len(grounding_checker.calls) == 1


def test_graph_retries_when_context_is_insufficient() -> None:
    graph, generator, grounding_checker = make_graph(
        sufficient=False,
        grounded=True,
        max_retries=2,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 2
    assert state["sufficient_context"] is False
    assert state["answer"] == (
        "I don't have enough information."
    )

    assert len(generator.calls) == 0
    assert len(grounding_checker.calls) == 0


def test_graph_falls_back_after_max_retries() -> None:
    graph, _, _ = make_graph(
        sufficient=False,
        max_retries=2,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 2
    assert state["answer"] == (
        "I don't have enough information."
    )
    assert state["grounded"] is False


def test_graph_falls_back_when_retries_are_disabled() -> None:
    graph, _, _ = make_graph(
        sufficient=False,
        max_retries=0,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["retry_count"] == 0
    assert state["answer"] == (
        "I don't have enough information."
    )
    assert state["grounded"] is False


def test_graph_falls_back_when_generated_answer_is_ungrounded() -> None:
    graph, generator, grounding_checker = make_graph(
        sufficient=True,
        grounded=False,
    )

    state = graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert state["answer"] == (
        "I don't have enough information."
    )
    assert state["grounded"] is False

    assert len(generator.calls) == 1
    assert len(grounding_checker.calls) == 1


def test_graph_does_not_run_grounding_without_generation() -> None:
    graph, generator, grounding_checker = make_graph(
        sufficient=False,
        max_retries=0,
    )

    graph.invoke(
        RAGState(
            query="PostgreSQL indexes",
        )
    )

    assert len(generator.calls) == 0
    assert len(grounding_checker.calls) == 0


def test_graph_rejects_negative_max_retries() -> None:
    retrieve_node = RetrieveNode(
        retriever=FakeRetriever(),
    )

    rerank_node = RerankNode(
        reranker=FakeReranker(),
    )

    grade_node = GradeNode(
        grader=FakeGrader(),
    )

    generate_node = GenerateNode(
        generator=FakeGenerator(),
    )

    grounding_node = GroundingNode(
        checker=FakeGroundingChecker(),
    )

    try:
        build_rag_graph(
            retrieve_node=retrieve_node,
            rerank_node=rerank_node,
            grade_node=grade_node,
            generate_node=generate_node,
            grounding_node=grounding_node,
            max_retries=-1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "max_retries cannot be negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError for negative max_retries."
        )
