from dataclasses import dataclass

import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.multi_query_retriever import MultiQueryRetriever


@dataclass
class FakeQueryGenerator:
    queries: list[str]

    def generate(self, query: str) -> list[str]:
        return self.queries


class FakeRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
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

        return self.results_by_query[query]


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"Text for {chunk_id}",
        source="test",
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        retriever="hybrid",
        metadata={},
    )


def test_multi_query_retriever_runs_all_generated_queries():
    retriever = FakeRetriever(
        {
            "query one": [
                make_result("chunk-1", 0.9),
            ],
            "query two": [
                make_result("chunk-2", 0.8),
            ],
        }
    )

    generator = FakeQueryGenerator(
        queries=["query one", "query two"]
    )

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = multi_retriever.retrieve(
        query="original query",
        top_k=5,
    )

    assert len(retriever.calls) == 2

    assert retriever.calls[0]["query"] == "query one"
    assert retriever.calls[1]["query"] == "query two"

    assert {
        result.chunk.chunk_id
        for result in results
    } == {"chunk-1", "chunk-2"}


def test_multi_query_retriever_deduplicates_chunks_with_rrf():
    retriever = FakeRetriever(
        {
            "query one": [
                make_result("chunk-1", 0.9),
                make_result("chunk-2", 0.8),
            ],
            "query two": [
                make_result("chunk-1", 0.7),
                make_result("chunk-3", 0.6),
            ],
        }
    )

    generator = FakeQueryGenerator(
        queries=["query one", "query two"]
    )

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = multi_retriever.retrieve(
        query="original query",
        top_k=5,
    )

    chunk_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert len(chunk_ids) == 3
    assert set(chunk_ids) == {
        "chunk-1",
        "chunk-2",
        "chunk-3",
    }

    assert chunk_ids[0] == "chunk-1"


def test_multi_query_retriever_respects_top_k():
    retriever = FakeRetriever(
        {
            "query one": [
                make_result("chunk-1", 0.9),
                make_result("chunk-2", 0.8),
                make_result("chunk-3", 0.7),
            ],
            "query two": [
                make_result("chunk-4", 0.6),
                make_result("chunk-5", 0.5),
            ],
        }
    )

    generator = FakeQueryGenerator(
        queries=["query one", "query two"]
    )

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = multi_retriever.retrieve(
        query="original query",
        top_k=3,
    )

    assert len(results) == 3


def test_multi_query_retriever_rejects_empty_query():
    retriever = FakeRetriever({})
    generator = FakeQueryGenerator(
        queries=["query"]
    )

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    with pytest.raises(ValueError):
        multi_retriever.retrieve("   ")


def test_multi_query_retriever_rejects_empty_generated_queries():
    retriever = FakeRetriever({})
    generator = FakeQueryGenerator(
        queries=[]
    )

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    with pytest.raises(ValueError):
        multi_retriever.retrieve("test query")


def test_multi_query_retriever_rejects_invalid_rrf_k():
    retriever = FakeRetriever({})
    generator = FakeQueryGenerator(
        queries=["query"]
    )

    with pytest.raises(ValueError):
        MultiQueryRetriever(
            retriever=retriever,
            query_generator=generator,
            rrf_k=0,
        )
