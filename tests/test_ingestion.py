import pytest

from rag.ingestion import TxtLoader, ingest_txt
from rag.models import RawDocument


def test_txt_loader_returns_raw_document() -> None:
    loader = TxtLoader("data/postgresql.txt")

    document = loader.load()

    assert isinstance(document, RawDocument)
    assert document.document_id == "postgresql"
    assert "PostgreSQL is an open-source relational database system." in document.text
    assert document.source == r"data\postgresql.txt"
    assert document.metadata == {"file_type": "txt"}


def test_txt_loader_missing_file() -> None:
    loader = TxtLoader("data/missing.txt")

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_txt_loader_rejects_unsupported_file() -> None:
    loader = TxtLoader("data/postgresql.pdf")

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load()


def test_txt_loader_cleans_text() -> None:
    loader = TxtLoader("data/postgresql.txt")

    document = loader.load()

    assert document.text == (
        "PostgreSQL is an open-source relational database system.\n"
        "It supports SQL, transactions, indexes, and complex queries.\n"
        "PostgreSQL is commonly used for applications that require reliable data storage."
    )


def test_ingest_txt_returns_chunks() -> None:
    chunks = ingest_txt(
        "data/postgresql.txt",
        chunk_size=8,
        overlap=2,
    )

    assert len(chunks) > 0
    assert all(chunk.document_id == "postgresql" for chunk in chunks)
    assert all(chunk.source == r"data\postgresql.txt" for chunk in chunks)
    assert all(chunk.metadata == {"file_type": "txt"} for chunk in chunks)
