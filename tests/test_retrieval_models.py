from rag.models import DocumentChunk, RetrievalResult


def test_retrieval_result_stores_retrieval_information() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        text="PostgreSQL supports indexes.",
        source="data/postgresql.txt",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.91,
        retriever="dense",
        metadata={"query": "PostgreSQL indexes"},
    )

    assert result.chunk is chunk
    assert result.score == 0.91
    assert result.retriever == "dense"
    assert result.metadata == {"query": "PostgreSQL indexes"}


def test_retrieval_result_metadata_is_independent() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        text="PostgreSQL supports indexes.",
        source="data/postgresql.txt",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.91,
        retriever="dense",
    )

    result.metadata["rank"] = 1

    assert chunk.metadata == {}
