import pytest

from rag.fusion import reciprocal_rank_fusion
from rag.models import DocumentChunk


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_001",
        text=f"Text for {chunk_id}",
        source="data/example.txt",
    )


def test_rrf_combines_rankings() -> None:
    a = make_chunk("A")
    b = make_chunk("B")
    c = make_chunk("C")
    d = make_chunk("D")

    dense_results = [
        (a, 0.90),
        (b, 0.80),
        (c, 0.70),
    ]

    bm25_results = [
        (b, 5.0),
        (a, 4.0),
        (d, 3.0),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=60,
    )

    assert [chunk.chunk_id for chunk, _ in results] == [
        "A",
        "B",
        "C",
        "D",
    ]


def test_rrf_score_is_sum_of_rank_contributions() -> None:
    a = make_chunk("A")

    dense_results = [
        (a, 0.90),
    ]

    bm25_results = [
        (a, 5.0),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=60,
    )

    expected = (1 / 61) + (1 / 61)

    assert results[0][0].chunk_id == "A"
    assert results[0][1] == pytest.approx(expected)


def test_rrf_preserves_chunk_object() -> None:
    a = make_chunk("A")

    results = reciprocal_rank_fusion(
        [[(a, 0.5)]],
    )

    assert results[0][0] is a


def test_rrf_handles_empty_lists() -> None:
    a = make_chunk("A")

    results = reciprocal_rank_fusion(
        [
            [],
            [(a, 0.5)],
            [],
        ]
    )

    assert len(results) == 1
    assert results[0][0].chunk_id == "A"


def test_rrf_handles_completely_empty_input() -> None:
    results = reciprocal_rank_fusion([])

    assert results == []


def test_rrf_rejects_invalid_k() -> None:
    a = make_chunk("A")

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [[(a, 0.5)]],
            k=0,
        )
