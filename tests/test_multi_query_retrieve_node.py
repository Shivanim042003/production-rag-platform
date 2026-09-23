import pytest

from rag.graph_nodes import MultiQueryRetrieveNode
from rag.graph_state import RAGState
from rag.models import DocumentChunk, RetrievalResult


class FakeMultiQueryRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        dense_k: int,
        bm25_k: int,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
            }
        )

        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Test document",
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


def test_multi_query_retrieve_node_uses_transformed_query():
    retriever = FakeMultiQueryRetriever()

    node = MultiQueryRetrieveNode(
        retriever=retriever,
        top_k=5,
        dense_k=10,
        bm25_k=10,
    )

    state = RAGState(
        query="Why is PostgreSQL slow?",
        transformed_queries=[
            "PostgreSQL query performance problems"
        ],
    )

    result = node(state)

    assert retriever.calls[0]["query"] == (
        "PostgreSQL query performance problems"
    )

    assert len(result["candidates"]) == 1


def test_multi_query_retrieve_node_falls_back_to_original_query():
    retriever = FakeMultiQueryRetriever()

    node = MultiQueryRetrieveNode(
        retriever=retriever,
    )

    state = RAGState(
        query="What are PostgreSQL indexes?"
    )

    node(state)

    assert retriever.calls[0]["query"] == (
        "What are PostgreSQL indexes?"
    )


def test_multi_query_retrieve_node_passes_retrieval_parameters():
    retriever = FakeMultiQueryRetriever()

    node = MultiQueryRetrieveNode(
        retriever=retriever,
        top_k=7,
        dense_k=15,
        bm25_k=20,
    )

    state = RAGState(
        query="PostgreSQL performance"
    )

    node(state)

    assert retriever.calls[0] == {
        "query": "PostgreSQL performance",
        "top_k": 7,
        "dense_k": 15,
        "bm25_k": 20,
    }
