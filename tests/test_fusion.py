import pytest

from rag.fusion import reciprocal_rank_fusion
from rag.models import DocumentChunk, RetrievalResult


def make_result(
    chunk_id: str,
    retriever: str,
    score: float,
) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=f"Text for {chunk_id}",
        source="data/example.txt",
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        retriever=retriever,
    )


def test_rrf_combines_rankings() -> None:
    a_dense = make_result("A", "dense", 0.90)
    b_dense = make_result("B", "dense", 0.80)
    c_dense = make_result("C", "dense", 0.70)

    b_bm25 = make_result("B", "bm25", 5.0)
    a_bm25 = make_result("A", "bm25", 4.0)
    d_bm25 = make_result("D", "bm25", 3.0)

    results = reciprocal_rank_fusion(
        [
            [a_dense, b_dense, c_dense],
            [b_bm25, a_bm25, d_bm25],
        ],
        k=60,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "A",
        "B",
        "C",
        "D",
    ]

    assert all(result.retriever == "rrf" for result in results)


def test_rrf_score_is_sum_of_rank_contributions() -> None:
    a_dense = make_result("A", "dense", 0.90)
    a_bm25 = make_result("A", "bm25", 5.0)

    results = reciprocal_rank_fusion(
        [
            [a_dense],
            [a_bm25],
        ],
        k=60,
    )

    expected = (1 / 61) + (1 / 61)

    assert results[0].chunk.chunk_id == "A"
    assert results[0].score == pytest.approx(expected)


def test_rrf_preserves_chunk() -> None:
    chunk = DocumentChunk(
        chunk_id="A",
        document_id="doc_001",
        text="Text for A",
        source="data/example.txt",
    )

    result = RetrievalResult(
        chunk=chunk,
        score=0.5,
        retriever="dense",
    )

    results = reciprocal_rank_fusion([[result]])

    assert results[0].chunk is chunk


def test_rrf_records_source_ranks() -> None:
    dense = make_result("A", "dense", 0.9)
    bm25 = make_result("A", "bm25", 4.0)

    results = reciprocal_rank_fusion(
        [
            [dense],
            [bm25],
        ]
    )

    assert results[0].metadata["rank"] == 1
    assert results[0].metadata["source_ranks"] == {
        "dense": 1,
        "bm25": 1,
    }


def test_rrf_handles_empty_lists() -> None:
    result = make_result("A", "dense", 0.5)

    results = reciprocal_rank_fusion(
        [
            [],
            [result],
            [],
        ]
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "A"


def test_rrf_handles_completely_empty_input() -> None:
    results = reciprocal_rank_fusion([])

    assert results == []


def test_rrf_rejects_invalid_k() -> None:
    result = make_result("A", "dense", 0.5)

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [[result]],
            k=0,
        )
