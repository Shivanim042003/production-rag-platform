import math
from collections.abc import Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero.")


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    hits = sum(chunk_id in relevant_ids for chunk_id in top_k)
    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(chunk_id in relevant_ids for chunk_id in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def candidate_recall(
    candidate_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0

    hits = sum(chunk_id in relevant_ids for chunk_id in candidate_ids)
    return hits / len(relevant_ids)
