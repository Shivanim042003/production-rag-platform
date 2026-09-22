import numpy as np
import pytest

from rag.embeddings import EmbeddingModel
from rag.models import DocumentChunk


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=text,
        source="data/example.txt",
    )


def test_embedding_dimension_is_positive(
    embedding_model: EmbeddingModel,
) -> None:
    assert embedding_model.dimension > 0


def test_embed_documents_returns_expected_shape(
    embedding_model: EmbeddingModel,
) -> None:
    chunks = [
        make_chunk("chunk_001", "PostgreSQL supports indexes."),
        make_chunk("chunk_002", "Redis is an in-memory data store."),
    ]

    embeddings = embedding_model.embed_documents(chunks)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, embedding_model.dimension)


def test_embed_query_returns_expected_shape(
    embedding_model: EmbeddingModel,
) -> None:
    embedding = embedding_model.embed_query("How do PostgreSQL indexes work?")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


def test_normalized_document_embeddings_have_unit_norm(
    embedding_model: EmbeddingModel,
) -> None:
    chunks = [
        make_chunk("chunk_001", "PostgreSQL supports indexes."),
        make_chunk("chunk_002", "Redis is an in-memory data store."),
    ]

    embeddings = embedding_model.embed_documents(chunks)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_normalized_query_embedding_has_unit_norm(
    embedding_model: EmbeddingModel,
) -> None:
    embedding = embedding_model.embed_query("PostgreSQL indexes")

    norm = np.linalg.norm(embedding)

    assert np.isclose(norm, 1.0, atol=1e-5)


def test_empty_document_list_returns_empty_matrix(
    embedding_model: EmbeddingModel,
) -> None:
    embeddings = embedding_model.embed_documents([])

    assert embeddings.shape == (0, embedding_model.dimension)


def test_empty_query_is_rejected(
    embedding_model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedding_model.embed_query("   ")
