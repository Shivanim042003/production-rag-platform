from collections.abc import Sequence

import numpy as np

from rag.bm25 import BM25Retriever
from rag.hybrid import HybridRetriever
from rag.models import DocumentChunk, RetrievalResult
from rag.reranker import Reranker
from rag.vector_store import FaissVectorStore


class FakeEmbeddingModel:
    dimension = 2

    def embed_query(self, query: str) -> np.ndarray:
        if "postgresql" in query.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


class FakeScorer:
    def predict(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        scores = []

        for _, chunk_text in pairs:
            if "transactions" in chunk_text.lower():
                scores.append(10.0)
            elif "indexes" in chunk_text.lower():
                scores.append(5.0)
            else:
                scores.append(1.0)

        return np.asarray(scores, dtype=np.float32)


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="postgresql",
            text="PostgreSQL supports transactions.",
            source="postgresql.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="postgresql",
            text="PostgreSQL supports indexes.",
            source="postgresql.txt",
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            document_id="redis",
            text="Redis is used for caching.",
            source="redis.txt",
        ),
    ]


def make_hybrid_retriever() -> HybridRetriever:
    chunks = make_chunks()
    embedding_model = FakeEmbeddingModel()

    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

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


def test_hybrid_results_can_be_reranked() -> None:
    hybrid = make_hybrid_retriever()
    reranker = Reranker(FakeScorer())

    candidates = hybrid.retrieve(
        query="PostgreSQL",
        top_k=3,
        dense_k=3,
        bm25_k=3,
    )

    assert len(candidates) == 3

    reranked = reranker.rerank(
        query="PostgreSQL",
        results=candidates,
        top_k=3,
    )

    assert len(reranked) == 3

    assert reranked[0].chunk.chunk_id == "chunk_1"
    assert reranked[0].score == 10.0
    assert reranked[0].retriever == "reranker"


def test_reranking_changes_candidate_order() -> None:
    hybrid = make_hybrid_retriever()
    reranker = Reranker(FakeScorer())

    candidates = hybrid.retrieve(
        query="PostgreSQL",
        top_k=3,
        dense_k=3,
        bm25_k=3,
    )

    original_ids = [result.chunk.chunk_id for result in candidates]

    reranked = reranker.rerank(
        query="PostgreSQL",
        results=candidates,
        top_k=3,
    )

    reranked_ids = [result.chunk.chunk_id for result in reranked]

    assert original_ids != reranked_ids
    assert reranked_ids == [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]


def test_reranker_preserves_hybrid_provenance() -> None:
    hybrid = make_hybrid_retriever()
    reranker = Reranker(FakeScorer())

    candidates = hybrid.retrieve(
        query="PostgreSQL",
        top_k=3,
        dense_k=3,
        bm25_k=3,
    )

    reranked = reranker.rerank(
        query="PostgreSQL",
        results=candidates,
        top_k=3,
    )

    for result in reranked:
        assert result.retriever == "reranker"
        assert "previous_score" in result.metadata
        assert result.metadata["previous_retriever"] == "rrf"
        assert "previous_rank" in result.metadata
