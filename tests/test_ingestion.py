import pytest

from rag.ingestion import TxtLoader
from rag.models import RawDocument


def test_txt_loader_returns_raw_document() -> None:
    loader = TxtLoader("data/postgresql.txt")

    document = loader.load()

    assert isinstance(document, RawDocument)
    assert document.document_id == "postgresql"
    assert "PostgreSQL is an open-source relational database system." in document.text
    assert document.source == "data\\postgresql.txt"
    assert document.metadata == {"file_type": "txt"}


def test_txt_loader_missing_file() -> None:
    loader = TxtLoader("data/missing.txt")

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_txt_loader_rejects_unsupported_file() -> None:
    loader = TxtLoader("data/postgresql.pdf")

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load()
