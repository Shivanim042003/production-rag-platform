from collections.abc import Sequence

from rag.models import DocumentChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[
        Sequence[tuple[DocumentChunk, float]]
    ],
    k: int = 60,
) -> list[tuple[DocumentChunk, float]]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, DocumentChunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunk_id = chunk.chunk_id

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (k + rank)
            )

            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = chunk

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    return [
        (chunks_by_id[chunk_id], fused_scores[chunk_id])
        for chunk_id in ranked_ids
    ]
