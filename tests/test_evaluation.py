import pytest

from rag.evaluation import (
    candidate_recall,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


RELEVANT = {"chunk_1", "chunk_3"}


def test_precision_at_k() -> None:
    retrieved = ["chunk_1", "chunk_2", "chunk_4", "chunk_3"]

    assert precision_at_k(retrieved, RELEVANT, k=4) == 0.5


def test_recall_at_k() -> None:
    retrieved = ["chunk_1", "chunk_2", "chunk_4", "chunk_3"]

    assert recall_at_k(retrieved, RELEVANT, k=4) == 1.0


def test_recall_at_k_when_one_relevant_chunk_is_missing() -> None:
    retrieved = ["chunk_1", "chunk_2", "chunk_4"]

    assert recall_at_k(retrieved, RELEVANT, k=3) == 0.5


def test_mean_reciprocal_rank() -> None:
    retrieved = ["chunk_7", "chunk_4", "chunk_3", "chunk_1"]

    assert mean_reciprocal_rank(retrieved, RELEVANT) == pytest.approx(1 / 3)


def test_mean_reciprocal_rank_when_no_relevant_result() -> None:
    retrieved = ["chunk_7", "chunk_4"]

    assert mean_reciprocal_rank(retrieved, RELEVANT) == 0.0


def test_ndcg_at_k() -> None:
    retrieved = ["chunk_1", "chunk_2", "chunk_3"]

    score = ndcg_at_k(retrieved, RELEVANT, k=3)

    assert score == pytest.approx(
        (
            1.0 + (1.0 / 2.0)
        )
        / (
            1.0 + (1.0 / 1.584962500721156)
        )
    )


def test_ndcg_is_one_when_all_relevant_results_are_at_top() -> None:
    retrieved = ["chunk_1", "chunk_3", "chunk_7"]

    assert ndcg_at_k(retrieved, RELEVANT, k=3) == pytest.approx(1.0)


def test_candidate_recall() -> None:
    candidates = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_7",
    ]

    assert candidate_recall(candidates, RELEVANT) == 1.0


def test_candidate_recall_when_candidate_pool_misses_relevant_chunk() -> None:
    candidates = [
        "chunk_1",
        "chunk_2",
        "chunk_7",
    ]

    assert candidate_recall(candidates, RELEVANT) == 0.5


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        precision_at_k(["chunk_1"], RELEVANT, k)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k(["chunk_1"], RELEVANT, k)


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        ndcg_at_k(["chunk_1"], RELEVANT, k)
