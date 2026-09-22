import numpy as np

from rag.bm25 import BM25Retriever
from rag.cross_encoder import CrossEncoderScorer
from rag.grader import RelevanceGrader
from rag.hybrid import HybridRetriever
from rag.local_grader import LocalRelevanceScorer
from rag.models import DocumentChunk
from rag.reranker import Reranker
from rag.vector_store import FaissVectorStore


class FakeEmbeddingModel:
    dimension = 2

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for _, text in pairs:
            if "transaction" in text.lower():
                scores.append(10.0)
            elif "index" in text.lower():
                scores.append(8.0)
            else:
                scores.append(1.0)

        return np.asarray(scores, dtype=np.float32)


class FakeGradingScorer:
    def score(self, query: str, text: str):
        from rag.grader import RelevanceGrade

        relevant = (
            "transaction" in text.lower()
            or "index" in text.lower()
        )

        return RelevanceGrade(
            relevant=relevant,
            score=0.95 if relevant else 0.10,
            reason="Relevant." if relevant else "Not relevant.",
        )


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
            text="Redis is commonly used for caching.",
            source="redis.txt",
        ),
    ]


def make_hybrid() -> HybridRetriever:
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

    return HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=BM25Retriever(chunks),
    )


def test_retrieval_reranking_and_grading_pipeline() -> None:
    hybrid = make_hybrid()

    reranker = Reranker(
        CrossEncoderScorer.__new__(CrossEncoderScorer)
    )
    reranker.scorer = FakeCrossEncoder()

    grader = RelevanceGrader(FakeGradingScorer())

    candidates = hybrid.retrieve(
        query="PostgreSQL transaction indexes",
        top_k=3,
        dense_k=3,
        bm25_k=3,
    )

    reranked = reranker.rerank(
        query="PostgreSQL transaction indexes",
        results=candidates,
        top_k=3,
    )

    graded = grader.grade_results(
        query="PostgreSQL transaction indexes",
        results=reranked,
    )

    relevant = grader.filter_relevant(graded)

    assert len(candidates) == 3
    assert len(reranked) == 3
    assert len(relevant) == 2

    assert [result.chunk.chunk_id for result in relevant] == [
        "chunk_1",
        "chunk_2",
    ]


def test_pipeline_detects_sufficient_context() -> None:
    hybrid = make_hybrid()

    reranker = Reranker(
        CrossEncoderScorer.__new__(CrossEncoderScorer)
    )
    reranker.scorer = FakeCrossEncoder()

    grader = RelevanceGrader(FakeGradingScorer())

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

    graded = grader.grade_results(
        query="PostgreSQL",
        results=reranked,
    )

    assert grader.has_sufficient_context(
        graded,
        min_relevant=2,
    ) is True


def test_pipeline_detects_insufficient_context() -> None:
    grader = RelevanceGrader(FakeGradingScorer())

    from rag.models import RetrievalResult

    irrelevant = RetrievalResult(
        chunk=DocumentChunk(
            chunk_id="irrelevant",
            document_id="redis",
            text="Redis provides caching.",
            source="redis.txt",
        ),
        score=1.0,
        retriever="reranker",
    )

    graded = grader.grade_results(
        query="How does PostgreSQL use MVCC?",
        results=[irrelevant],
    )

    assert grader.has_sufficient_context(
        graded,
        min_relevant=1,
    ) is False
