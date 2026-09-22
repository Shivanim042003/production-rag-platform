from rag.bm25 import BM25Retriever
from rag.embeddings import EmbeddingModel
from rag.fusion import reciprocal_rank_fusion
from rag.models import RetrievalResult
from rag.vector_store import FaissVectorStore


class HybridRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FaissVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 20,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if dense_k <= 0:
            raise ValueError("dense_k must be greater than zero.")

        if bm25_k <= 0:
            raise ValueError("bm25_k must be greater than zero.")

        query_embedding = self.embedding_model.embed_query(query)

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=dense_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=bm25_k,
        )

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ],
            k=self.rrf_k,
        )

        return fused_results[:top_k]
