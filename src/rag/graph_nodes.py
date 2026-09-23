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
        query = (
            state.transformed_queries[-1]
            if state.transformed_queries
            else state.query
        )

        results = self.retriever.retrieve(
            query=query,
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


class GenerateNode:
    """Generates an answer from the selected context."""

    def __init__(self, generator) -> None:
        self.generator = generator

    def __call__(self, state: RAGState) -> dict:
        if not state.context:
            raise ValueError(
                "Cannot generate an answer without context."
            )

        result = self.generator.generate(
            query=state.query,
            context=state.context,
        )

        return {
            "answer": result.answer,
        }


class GroundingNode:
    """Checks whether the generated answer is supported by context."""

    def __init__(self, checker) -> None:
        self.checker = checker

    def __call__(self, state: RAGState) -> dict:
        if not state.answer:
            raise ValueError(
                "Cannot check grounding without an answer."
            )

        if not state.context:
            raise ValueError(
                "Cannot check grounding without context."
            )

        result = self.checker.check(
            answer=state.answer,
            context=state.context,
        )

        return {
            "grounded": result.grounded,
        }


class QueryTransformNode:
    """Transforms the user query for retrieval."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def __call__(self, state: RAGState) -> dict:
        transformed_query = self.transformer.transform(
            state.query
        )

        return {
            "transformed_queries": [transformed_query],
        }
from rag.graph_state import RAGState
from rag.multi_query_retriever import MultiQueryRetriever


class MultiQueryRetrieveNode:
    """Runs multi-query retrieval and RRF fusion."""

    def __init__(
        self,
        retriever: MultiQueryRetriever,
        top_k: int = 10,
        dense_k: int = 10,
        bm25_k: int = 10,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.dense_k = dense_k
        self.bm25_k = bm25_k

    def __call__(self, state: RAGState) -> dict:
        query = (
            state.transformed_queries[-1]
            if state.transformed_queries
            else state.query
        )

        results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            dense_k=self.dense_k,
            bm25_k=self.bm25_k,
        )

        return {
            "candidates": results,
        }
