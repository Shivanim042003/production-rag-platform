import statistics
import time

from run_retrieval_benchmark import (
    CANDIDATE_K,
    FINAL_K,
    build_hybrid,
    load_benchmark,
    load_chunks,
)
from rag.cross_encoder import CrossEncoderScorer
from rag.evaluation import (
    candidate_recall,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.reranker import Reranker

ROUNDS = 3
CONFIGS = (5, 10)


def run_query(hybrid, reranker, item, rerank_k):
    query = item["query"]
    relevant = set(item["relevant_chunks"])

    start = time.perf_counter()
    candidates = hybrid.retrieve(
        query=query,
        top_k=CANDIDATE_K,
        dense_k=CANDIDATE_K,
        bm25_k=CANDIDATE_K,
    )
    retrieval_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    reranked = reranker.rerank(
        query=query,
        results=candidates[:rerank_k],
        top_k=FINAL_K,
    )
    rerank_ms = (time.perf_counter() - start) * 1000

    ids = [r.chunk.chunk_id for r in reranked]
    candidate_ids = [r.chunk.chunk_id for r in candidates]

    metrics = {
        "precision": precision_at_k(ids, relevant, FINAL_K),
        "recall": recall_at_k(ids, relevant, FINAL_K),
        "mrr": mean_reciprocal_rank(ids, relevant),
        "ndcg": ndcg_at_k(ids, relevant, FINAL_K),
        "candidate_recall": candidate_recall(candidate_ids, relevant),
    }

    return metrics, retrieval_ms, rerank_ms


def summarize(values):
    return (
        f"mean={statistics.mean(values):.2f} ms, "
        f"median={statistics.median(values):.2f} ms"
    )


def main():
    chunks = load_chunks()
    benchmark = load_benchmark()
    hybrid = build_hybrid(chunks)
    reranker = Reranker(CrossEncoderScorer())

    print(f"Chunks: {len(chunks)} | Queries: {len(benchmark)}")
    print(f"Measured rounds: {ROUNDS}")

    # Warm up both stages before collecting measurements.
    warmup_item = benchmark[0]
    candidates = hybrid.retrieve(
        query=warmup_item["query"],
        top_k=CANDIDATE_K,
        dense_k=CANDIDATE_K,
        bm25_k=CANDIDATE_K,
    )
    reranker.rerank(warmup_item["query"], candidates[:10], top_k=FINAL_K)

    metrics_by_config = {k: [] for k in CONFIGS}
    retrieval_times = {k: [] for k in CONFIGS}
    rerank_times = {k: [] for k in CONFIGS}
    total_times = {k: [] for k in CONFIGS}

    for round_index in range(ROUNDS):
        order = CONFIGS if round_index % 2 == 0 else tuple(reversed(CONFIGS))

        for rerank_k in order:
            for item in benchmark:
                metrics, retrieval_ms, rerank_ms = run_query(
                    hybrid, reranker, item, rerank_k
                )
                metrics_by_config[rerank_k].append(metrics)
                retrieval_times[rerank_k].append(retrieval_ms)
                rerank_times[rerank_k].append(rerank_ms)
                total_times[rerank_k].append(retrieval_ms + rerank_ms)

    for rerank_k in CONFIGS:
        rows = metrics_by_config[rerank_k]
        print(f"\n{'=' * 60}")
        print(f"RERANK CANDIDATES: {rerank_k} | RETURN TOP: {FINAL_K}")
        print(f"{'=' * 60}")

        for name in rows[0]:
            value = statistics.mean(row[name] for row in rows)
            print(f"{name:20s}: {value:.4f}")

        print("\nLatency:")
        print(f"  Retrieval: {summarize(retrieval_times[rerank_k])}")
        print(f"  Reranking: {summarize(rerank_times[rerank_k])}")
        print(f"  Total:     {summarize(total_times[rerank_k])}")


if __name__ == "__main__":
    main()
