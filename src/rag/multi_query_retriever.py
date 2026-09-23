from rag.fusion import reciprocal_rank_fusion
from rag.hybrid import HybridRetriever


class MultiQueryRetriever:
    """
    Runs hybrid retrieval for multiple generated queries
    and combines the ranked results using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        query_generator,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.retriever = retriever
        self.query_generator = query_generator
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_k: int = 10,
        bm25_k: int = 10,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if dense_k <= 0:
            raise ValueError("dense_k must be greater than zero.")

        if bm25_k <= 0:
            raise ValueError("bm25_k must be greater than zero.")

        queries = self.query_generator.generate(query)

        if not queries:
            raise ValueError(
                "Query generator returned no queries."
            )

        ranked_lists = []

        for generated_query in queries:
            results = self.retriever.retrieve(
                query=generated_query,
                top_k=top_k,
                dense_k=dense_k,
                bm25_k=bm25_k,
            )

            ranked_lists.append(results)

        fused_results = reciprocal_rank_fusion(
            ranked_lists,
            k=self.rrf_k,
        )

        return fused_results[:top_k]
