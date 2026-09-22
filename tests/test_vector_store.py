import numpy as np
import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.vector_store import FaissVectorStore


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=text,
        source="data/example.txt",
    )


def test_store_starts_empty() -> None:
    store = FaissVectorStore(dimension=2)

    assert store.size == 0


def test_add_stores_chunks_and_vectors() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
        make_chunk("chunk_002", "B"),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    assert store.size == 2


def test_search_returns_retrieval_results_in_similarity_order() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
        make_chunk("chunk_002", "B"),
        make_chunk("chunk_003", "C"),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=3)

    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )

    assert [result.chunk.chunk_id for result in results] == [
        "chunk_001",
        "chunk_002",
        "chunk_003",
    ]

    assert np.isclose(results[0].score, 1.0)
    assert np.isclose(results[1].score, 0.0)
    assert np.isclose(results[2].score, -1.0)


def test_search_records_dense_retrieval_metadata() -> None:
    store = FaissVectorStore(dimension=2)

    chunk = make_chunk("chunk_001", "A")

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    store.add([chunk], embeddings)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=1)

    assert results[0].retriever == "dense"
    assert results[0].metadata["rank"] == 1


def test_search_respects_top_k() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
        make_chunk("chunk_002", "B"),
        make_chunk("chunk_003", "C"),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert [result.chunk.chunk_id for result in results] == [
        "chunk_001",
        "chunk_002",
    ]


def test_search_on_empty_store_returns_empty_list() -> None:
    store = FaissVectorStore(dimension=2)

    query = np.array([1.0, 0.0], dtype=np.float32)

    assert store.search(query) == []


def test_add_rejects_wrong_embedding_dimension() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
    ]

    embeddings = np.array(
        [[1.0, 0.0, 0.5]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimension does not match index dimension",
    ):
        store.add(chunks, embeddings)


def test_add_rejects_mismatched_chunk_count() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
        make_chunk("chunk_002", "B"),
    ]

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Number of embeddings must match number of chunks",
    ):
        store.add(chunks, embeddings)


def test_search_rejects_wrong_query_dimension() -> None:
    store = FaissVectorStore(dimension=2)

    query = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Query embedding dimension does not match index dimension",
    ):
        store.search(query)


def test_search_rejects_invalid_top_k() -> None:
    store = FaissVectorStore(dimension=2)

    query = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        store.search(query, top_k=0)


def test_add_rejects_non_2d_embeddings() -> None:
    store = FaissVectorStore(dimension=2)

    chunks = [
        make_chunk("chunk_001", "A"),
    ]

    embeddings = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2D array",
    ):
        store.add(chunks, embeddings)


def test_search_rejects_non_1d_query() -> None:
    store = FaissVectorStore(dimension=2)

    query = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="query_embedding must be a 1D array",
    ):
        store.search(query)
