import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.reranker import Reranker


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.received_pairs = None

    def predict(self, pairs):
        self.received_pairs = pairs
        return self.scores


def make_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                text="PostgreSQL supports transactions.",
                source="postgresql.txt",
            ),
            score=0.91,
            retriever="rrf",
            metadata={"rank": 1},
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                text="Redis is used for caching.",
                source="redis.txt",
            ),
            score=0.87,
            retriever="rrf",
            metadata={"rank": 2},
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="chunk_3",
                document_id="doc_1",
                text="PostgreSQL supports indexes.",
                source="postgresql.txt",
            ),
            score=0.83,
            retriever="rrf",
            metadata={"rank": 3},
        ),
    ]


def test_reranker_sorts_by_score() -> None:
    scorer = FakeScorer([0.2, 0.9, 0.5])
    reranker = Reranker(scorer)

    results = reranker.rerank(
        query="PostgreSQL features",
        results=make_results(),
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "chunk_2",
        "chunk_3",
        "chunk_1",
    ]

    assert [result.score for result in results] == [0.9, 0.5, 0.2]


def test_reranker_uses_query_and_chunk_pairs() -> None:
    scorer = FakeScorer([0.1, 0.2, 0.3])
    reranker = Reranker(scorer)

    reranker.rerank(
        query="database transactions",
        results=make_results(),
        top_k=3,
    )

    assert scorer.received_pairs == [
        ["database transactions", "PostgreSQL supports transactions."],
        ["database transactions", "Redis is used for caching."],
        ["database transactions", "PostgreSQL supports indexes."],
    ]


def test_reranker_respects_top_k() -> None:
    scorer = FakeScorer([0.2, 0.9, 0.5])
    reranker = Reranker(scorer)

    results = reranker.rerank(
        query="PostgreSQL",
        results=make_results(),
        top_k=2,
    )

    assert len(results) == 2
    assert [result.chunk.chunk_id for result in results] == [
        "chunk_2",
        "chunk_3",
    ]


def test_reranker_preserves_previous_retrieval_information() -> None:
    scorer = FakeScorer([0.2, 0.9, 0.5])
    reranker = Reranker(scorer)

    results = reranker.rerank(
        query="PostgreSQL",
        results=make_results(),
        top_k=3,
    )

    result = next(
        result for result in results if result.chunk.chunk_id == "chunk_2"
    )

    assert result.retriever == "reranker"
    assert result.metadata["previous_score"] == 0.87
    assert result.metadata["previous_retriever"] == "rrf"
    assert result.metadata["previous_rank"] == 2


def test_reranker_assigns_new_ranks() -> None:
    scorer = FakeScorer([0.2, 0.9, 0.5])
    reranker = Reranker(scorer)

    results = reranker.rerank(
        query="PostgreSQL",
        results=make_results(),
        top_k=3,
    )

    assert [result.metadata["rank"] for result in results] == [1, 2, 3]


def test_reranker_returns_empty_for_empty_results() -> None:
    scorer = FakeScorer([])
    reranker = Reranker(scorer)

    assert reranker.rerank("PostgreSQL", [], top_k=5) == []


def test_reranker_rejects_empty_query() -> None:
    scorer = FakeScorer([])
    reranker = Reranker(scorer)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank("   ", make_results(), top_k=5)


@pytest.mark.parametrize("top_k", [0, -1])
def test_reranker_rejects_invalid_top_k(top_k: int) -> None:
    scorer = FakeScorer([0.1, 0.2, 0.3])
    reranker = Reranker(scorer)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        reranker.rerank("PostgreSQL", make_results(), top_k=top_k)


def test_reranker_does_not_call_scorer_for_empty_results() -> None:
    scorer = FakeScorer([])
    reranker = Reranker(scorer)

    reranker.rerank("PostgreSQL", [], top_k=5)

    assert scorer.received_pairs is None
