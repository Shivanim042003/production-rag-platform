import numpy as np
import pytest

from rag.bm25 import BM25Retriever
from rag.hybrid import HybridRetriever
from rag.models import DocumentChunk
from rag.vector_store import FaissVectorStore


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension

    def embed_query(self, query: str) -> np.ndarray:
        if "postgresql" in query.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=text,
        source="data/example.txt",
        metadata={"category": "database"},
    )


def make_hybrid_retriever() -> HybridRetriever:
    chunks = [
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

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    embedding_model = FakeEmbeddingModel()

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension,
    )
    vector_store.add(chunks, embeddings)

    bm25_retriever = BM25Retriever(chunks)

    return HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )


def test_hybrid_returns_fused_results() -> None:
    retriever = make_hybrid_retriever()

    results = retriever.retrieve(
        "PostgreSQL indexes",
        top_k=3,
    )

    assert len(results) == 3
    assert all(result.retriever == "rrf" for result in results)


def test_hybrid_combines_dense_and_bm25_signals() -> None:
    retriever = make_hybrid_retriever()

    results = retriever.retrieve(
        "PostgreSQL indexes",
        top_k=3,
    )

    result_by_id = {
        result.chunk.chunk_id: result
        for result in results
    }

    assert "chunk_001" in result_by_id

    source_ranks = result_by_id["chunk_001"].metadata["source_ranks"]

    assert "dense" in source_ranks
    assert "bm25" in source_ranks


def test_hybrid_respects_final_top_k() -> None:
    retriever = make_hybrid_retriever()

    results = retriever.retrieve(
        "PostgreSQL",
        top_k=1,
    )

    assert len(results) == 1


def test_hybrid_preserves_provenance() -> None:
    retriever = make_hybrid_retriever()

    results = retriever.retrieve(
        "PostgreSQL indexes",
        top_k=1,
    )

    result = results[0]

    assert result.chunk.document_id == "doc_001"
    assert result.chunk.source == "data/example.txt"
    assert result.chunk.metadata == {"category": "database"}


def test_hybrid_handles_empty_retrieval_sources() -> None:
    embedding_model = FakeEmbeddingModel()

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension,
    )

    bm25_retriever = BM25Retriever([])

    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    results = retriever.retrieve("PostgreSQL")

    assert results == []


def test_hybrid_rejects_empty_query() -> None:
    retriever = make_hybrid_retriever()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")


@pytest.mark.parametrize(
    ("top_k", "dense_k", "bm25_k"),
    [
        (0, 20, 20),
        (-1, 20, 20),
        (5, 0, 20),
        (5, -1, 20),
        (5, 20, 0),
        (5, 20, -1),
    ],
)
def test_hybrid_rejects_invalid_retrieval_limits(
    top_k: int,
    dense_k: int,
    bm25_k: int,
) -> None:
    retriever = make_hybrid_retriever()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        retriever.retrieve(
            "PostgreSQL",
            top_k=top_k,
            dense_k=dense_k,
            bm25_k=bm25_k,
        )


def test_hybrid_rejects_invalid_rrf_k() -> None:
    embedding_model = FakeEmbeddingModel()

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension,
    )

    bm25_retriever = BM25Retriever([])

    with pytest.raises(
        ValueError,
        match="rrf_k must be greater than zero",
    ):
        HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            rrf_k=0,
        )
