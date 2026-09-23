from rag.graph_nodes import (
    GenerateNode,
    GradeNode,
    GroundingNode,
    MultiQueryRetrieveNode,
    QueryTransformNode,
    RerankNode,
)
from rag.graph import build_rag_graph
from rag.graph_state import RAGState
from rag.query_transformer import IdentityQueryTransformer


class FakeMultiQueryRetriever:
    def retrieve(
        self,
        query,
        top_k,
        dense_k,
        bm25_k,
    ):
        from rag.models import DocumentChunk, RetrievalResult

        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="PostgreSQL indexes improve query performance.",
            source="test",
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=1.0,
                retriever="rrf",
                metadata={},
            )
        ]


class FakeReranker:
    def rerank(self, query, results, top_k):
        return results[:top_k]


class FakeGrader:
    def grade_results(self, query, results):
        from rag.grader import RelevanceGrade

        return [
            (
                result,
                RelevanceGrade(
                    relevant=True,
                    score=1.0,
                    reason="relevant",
                ),
            )
            for result in results
        ]

    def has_sufficient_context(self, graded_results, min_relevant):
        return True


class FakeGenerator:
    def generate(self, query, context):
        from rag.generator import GenerationResult

        return GenerationResult(
            answer="Indexes improve PostgreSQL query performance.",
            model="fake",
        )


class FakeGroundingChecker:
    def check(self, answer, context):
        from rag.grounding import GroundingResult

        return GroundingResult(
            grounded=True,
            score=1.0,
            reason="supported",
        )


def test_graph_supports_multi_query_retrieval():
    query_transform_node = QueryTransformNode(
        transformer=IdentityQueryTransformer()
    )

    retrieve_node = MultiQueryRetrieveNode(
        retriever=FakeMultiQueryRetriever(),
        top_k=5,
        dense_k=5,
        bm25_k=5,
    )

    rerank_node = RerankNode(
        reranker=FakeReranker(),
        top_k=5,
    )

    grade_node = GradeNode(
        grader=FakeGrader(),
        min_relevant=1,
    )

    generate_node = GenerateNode(
        generator=FakeGenerator(),
    )

    grounding_node = GroundingNode(
        checker=FakeGroundingChecker(),
    )

    graph = build_rag_graph(
        query_transform_node=query_transform_node,
        retrieve_node=retrieve_node,
        rerank_node=rerank_node,
        grade_node=grade_node,
        generate_node=generate_node,
        grounding_node=grounding_node,
    )

    result = graph.invoke(
        RAGState(
            query="What are PostgreSQL indexes?"
        )
    )

    assert result["answer"] == (
        "Indexes improve PostgreSQL query performance."
    )

    assert result["grounded"] is True

    assert result["transformed_queries"] == [
        "What are PostgreSQL indexes?"
    ]
