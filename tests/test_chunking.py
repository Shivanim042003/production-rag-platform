import pytest

from rag.chunking import fixed_size_chunks
from rag.models import DocumentChunk, RawDocument


def make_document(text: str) -> RawDocument:
    return RawDocument(
        document_id="doc_001",
        text=text,
        source="data/example.txt",
        metadata={"category": "database"},
    )


def test_fixed_size_chunks_without_overlap() -> None:
    document = make_document(
        "one two three four five six seven eight nine ten"
    )

    chunks = fixed_size_chunks(document, chunk_size=4)

    assert len(chunks) == 3
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "five six seven eight"
    assert chunks[2].text == "nine ten"


def test_fixed_size_chunks_with_overlap() -> None:
    document = make_document(
        "one two three four five six seven eight nine ten"
    )

    chunks = fixed_size_chunks(document, chunk_size=4, overlap=2)

    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"
    assert chunks[2].text == "five six seven eight"
    assert chunks[3].text == "seven eight nine ten"


def test_chunks_preserve_identity_and_provenance() -> None:
    document = make_document("one two three four five")

    chunks = fixed_size_chunks(document, chunk_size=3)

    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    assert chunks[0].chunk_id == "doc_001_chunk_001"
    assert chunks[1].chunk_id == "doc_001_chunk_002"

    assert all(chunk.document_id == "doc_001" for chunk in chunks)
    assert all(chunk.source == "data/example.txt" for chunk in chunks)
    assert all(
        chunk.metadata == {"category": "database"}
        for chunk in chunks
    )


def test_chunk_metadata_is_not_shared_with_document() -> None:
    document = make_document("one two three four")

    chunks = fixed_size_chunks(document, chunk_size=2)

    chunks[0].metadata["new_field"] = "value"

    assert "new_field" not in document.metadata


def test_empty_document_returns_empty_list() -> None:
    document = make_document("")

    chunks = fixed_size_chunks(document, chunk_size=5)

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_chunk_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    document = make_document("one two three")

    with pytest.raises(ValueError):
        fixed_size_chunks(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
