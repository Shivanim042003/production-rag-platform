import pytest

from rag.bm25 import BM25Retriever
from rag.models import DocumentChunk, RetrievalResult


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=text,
        source="data/example.txt",
        metadata={"category": "database"},
    )


def make_corpus() -> list[DocumentChunk]:
    return [
        make_chunk(
            "chunk_001",
            "PostgreSQL indexes improve query performance.",
        ),
        make_chunk(
            "chunk_002",
            "Redis provides in-memory caching.",
        ),
        make_chunk(
            "chunk_003",
            "PostgreSQL supports transactions.",
        ),
    ]


def test_bm25_ranks_lexically_relevant_chunk_first() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL indexes", top_k=3)

    assert results[0].chunk.chunk_id == "chunk_001"
    assert results[0].score > results[1].score


def test_bm25_returns_retrieval_results() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL")

    assert len(results) > 0
    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )
    assert all(
        isinstance(result.score, float)
        for result in results
    )


def test_bm25_respects_top_k() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL", top_k=2)

    assert len(results) == 2


def test_bm25_preserves_chunk_provenance() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve(
        "PostgreSQL indexes",
        top_k=1,
    )

    result = results[0]

    assert result.chunk.chunk_id == "chunk_001"
    assert result.chunk.document_id == "doc_001"
    assert result.chunk.source == "data/example.txt"
    assert result.chunk.metadata == {"category": "database"}


def test_bm25_records_retrieval_metadata() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve(
        "PostgreSQL indexes",
        top_k=2,
    )

    assert results[0].retriever == "bm25"
    assert results[0].metadata["query"] == "PostgreSQL indexes"
    assert results[0].metadata["rank"] == 1
    assert results[1].metadata["rank"] == 2


def test_bm25_empty_corpus_returns_empty_results() -> None:
    retriever = BM25Retriever([])

    results = retriever.retrieve("PostgreSQL")

    assert results == []


def test_bm25_rejects_empty_query() -> None:
    retriever = BM25Retriever(make_corpus())

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")


def test_bm25_rejects_invalid_top_k() -> None:
    retriever = BM25Retriever(make_corpus())

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve("PostgreSQL", top_k=0)


def test_bm25_returns_at_most_corpus_size() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve(
        "PostgreSQL",
        top_k=100,
    )

    assert len(results) == 3
