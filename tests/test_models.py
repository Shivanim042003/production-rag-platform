from rag.models import DocumentChunk, RawDocument


def test_raw_document_defaults_metadata() -> None:
    document = RawDocument(
        document_id="doc_001",
        text="PostgreSQL is a relational database.",
        source="postgresql.txt",
    )

    assert document.document_id == "doc_001"
    assert document.source == "postgresql.txt"
    assert document.metadata == {}


def test_document_chunk_links_to_parent_document() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        text="PostgreSQL supports indexes.",
        source="postgresql.txt",
    )

    assert chunk.chunk_id == "chunk_001"
    assert chunk.document_id == "doc_001"
    assert chunk.metadata == {}


def test_metadata_is_not_shared_between_objects() -> None:
    first = RawDocument(
        document_id="doc_001",
        text="First document",
        source="first.txt",
    )

    second = RawDocument(
        document_id="doc_002",
        text="Second document",
        source="second.txt",
    )

    first.metadata["category"] = "database"

    assert second.metadata == {}
