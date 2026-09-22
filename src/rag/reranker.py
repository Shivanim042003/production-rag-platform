from collections.abc import Sequence

from rag.models import RetrievalResult


class Reranker:
    """Reranks first-stage retrieval candidates using a scoring model."""

    def __init__(self, scorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not results:
            return []

        pairs = [[query, result.chunk.text] for result in results]
        scores = self.scorer.predict(pairs)

        reranked: list[RetrievalResult] = []

        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                    retriever="reranker",
                    metadata={
                        "previous_score": result.score,
                        "previous_retriever": result.retriever,
                        "previous_rank": result.metadata.get("rank"),
                    },
                )
            )

        reranked.sort(key=lambda result: result.score, reverse=True)

        final_results: list[RetrievalResult] = []

        for rank, result in enumerate(reranked[:top_k], start=1):
            result.metadata["rank"] = rank
            final_results.append(result)

        return final_results
