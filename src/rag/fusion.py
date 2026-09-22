from collections.abc import Sequence

from rag.models import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    fused_scores: dict[str, float] = {}
    results_by_id: dict[str, RetrievalResult] = {}
    source_ranks: dict[str, dict[str, int]] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.chunk.chunk_id

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (k + rank)
            )

            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = result

            source_ranks.setdefault(chunk_id, {})[
                result.retriever
            ] = rank

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    results: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        original = results_by_id[chunk_id]

        results.append(
            RetrievalResult(
                chunk=original.chunk,
                score=fused_scores[chunk_id],
                retriever="rrf",
                metadata={
                    "rank": rank,
                    "source_ranks": source_ranks[chunk_id],
                },
            )
        )

    return results
