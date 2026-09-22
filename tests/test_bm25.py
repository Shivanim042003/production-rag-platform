import pytest

from rag.bm25 import BM25Retriever
from rag.models import DocumentChunk


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

    assert results[0][0].chunk_id == "chunk_001"
    assert results[0][1] > results[1][1]


def test_bm25_returns_scores() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL")

    assert len(results) > 0
    assert all(isinstance(score, float) for _, score in results)


def test_bm25_respects_top_k() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL", top_k=2)

    assert len(results) == 2


def test_bm25_preserves_chunk_provenance() -> None:
    retriever = BM25Retriever(make_corpus())

    results = retriever.retrieve("PostgreSQL indexes", top_k=1)

    chunk, _ = results[0]

    assert chunk.chunk_id == "chunk_001"
    assert chunk.document_id == "doc_001"
    assert chunk.source == "data/example.txt"
    assert chunk.metadata == {"category": "database"}


def test_bm25_empty_corpus_returns_empty_results() -> None:
    retriever = BM25Retriever([])

    results = retriever.retrieve("PostgreSQL")

    assert results == []


def test_bm25_rejects_empty_query() -> None:
    retriever = BM25Retriever(make_corpus())

    with pytest.raises(ValueError, match="Query cannot be empty"):
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

    results = retriever.retrieve("PostgreSQL", top_k=100)

    assert len(results) == 3
