from rag.graph_state import RAGState
from rag.grader import RelevanceGrader
from rag.hybrid import HybridRetriever
from rag.reranker import Reranker


class RetrieveNode:
    """Runs first-stage hybrid retrieval."""

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 10,
        dense_k: int = 10,
        bm25_k: int = 10,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.dense_k = dense_k
        self.bm25_k = bm25_k

    def __call__(self, state: RAGState) -> dict:
        results = self.retriever.retrieve(
            query=state.query,
            top_k=self.top_k,
            dense_k=self.dense_k,
            bm25_k=self.bm25_k,
        )

        return {
            "candidates": results,
        }


class RerankNode:
    """Reranks retrieved candidates."""

    def __init__(
        self,
        reranker: Reranker,
        top_k: int = 5,
    ) -> None:
        self.reranker = reranker
        self.top_k = top_k

    def __call__(self, state: RAGState) -> dict:
        results = self.reranker.rerank(
            query=state.query,
            results=state.candidates,
            top_k=self.top_k,
        )

        return {
            "reranked_results": results,
        }


class GradeNode:
    """Grades reranked candidates for relevance."""

    def __init__(
        self,
        grader: RelevanceGrader,
        min_relevant: int = 1,
    ) -> None:
        self.grader = grader
        self.min_relevant = min_relevant

    def __call__(self, state: RAGState) -> dict:
        graded_results = self.grader.grade_results(
            query=state.query,
            results=state.reranked_results,
        )

        sufficient_context = self.grader.has_sufficient_context(
            graded_results,
            min_relevant=self.min_relevant,
        )

        context = [
            result.chunk.text
            for result, grade in graded_results
            if grade.relevant
        ]

        return {
            "graded_results": graded_results,
            "context": context,
            "sufficient_context": sufficient_context,
        }
