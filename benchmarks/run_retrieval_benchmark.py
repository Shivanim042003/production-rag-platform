import json
from pathlib import Path

from rag.bm25 import BM25Retriever
from rag.cross_encoder import CrossEncoderScorer
from rag.embeddings import EmbeddingModel
from rag.evaluation import (
    candidate_recall,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.hybrid import HybridRetriever
from rag.ingestion import ingest_txt
from rag.reranker import Reranker
from rag.vector_store import FaissVectorStore


DATA_DIR = Path("benchmarks/data")
RELEVANCE_FILE = Path("benchmarks/relevance.json")

CHUNK_SIZE = 35
CHUNK_OVERLAP = 5

CANDIDATE_K = 10
FINAL_K = 5


def load_chunks():
    files = sorted(DATA_DIR.glob("*.txt"))

    chunks = []

    for path in files:
        chunks.extend(
            ingest_txt(
                path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )

    return chunks


def load_benchmark():
    with open(RELEVANCE_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_hybrid(chunks):
    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(chunks)

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension,
    )

    vector_store.add(chunks, embeddings)

    bm25_retriever = BM25Retriever(chunks)

    hybrid = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    return hybrid


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"  Precision@5:      {metrics['precision_at_5']:.4f}")
    print(f"  Recall@5:         {metrics['recall_at_5']:.4f}")
    print(f"  MRR:              {metrics['mrr']:.4f}")
    print(f"  NDCG@5:           {metrics['ndcg_at_5']:.4f}")
    print(f"  Candidate Recall: {metrics['candidate_recall']:.4f}")


def main() -> None:
    chunks = load_chunks()
    benchmark = load_benchmark()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loaded {len(benchmark)} benchmark queries.")

    hybrid = build_hybrid(chunks)

    reranker = Reranker(
        CrossEncoderScorer()
    )

    hybrid_metrics = []
    reranked_metrics = []

    for item in benchmark:
        query = item["query"]
        relevant_ids = set(item["relevant_chunks"])

        candidates = hybrid.retrieve(
            query=query,
            top_k=CANDIDATE_K,
            dense_k=CANDIDATE_K,
            bm25_k=CANDIDATE_K,
        )

        candidate_ids = [
            result.chunk.chunk_id
            for result in candidates
        ]

        hybrid_top_k = candidates[:FINAL_K]

        hybrid_ids = [
            result.chunk.chunk_id
            for result in hybrid_top_k
        ]

        reranked = reranker.rerank(
            query=query,
            results=candidates,
            top_k=FINAL_K,
        )

        reranked_ids = [
            result.chunk.chunk_id
            for result in reranked
        ]

        hybrid_result = {
            "precision_at_5": precision_at_k(
                hybrid_ids,
                relevant_ids,
                FINAL_K,
            ),
            "recall_at_5": recall_at_k(
                hybrid_ids,
                relevant_ids,
                FINAL_K,
            ),
            "mrr": mean_reciprocal_rank(
                hybrid_ids,
                relevant_ids,
            ),
            "ndcg_at_5": ndcg_at_k(
                hybrid_ids,
                relevant_ids,
                FINAL_K,
            ),
            "candidate_recall": candidate_recall(
                candidate_ids,
                relevant_ids,
            ),
        }

        reranked_result = {
            "precision_at_5": precision_at_k(
                reranked_ids,
                relevant_ids,
                FINAL_K,
            ),
            "recall_at_5": recall_at_k(
                reranked_ids,
                relevant_ids,
                FINAL_K,
            ),
            "mrr": mean_reciprocal_rank(
                reranked_ids,
                relevant_ids,
            ),
            "ndcg_at_5": ndcg_at_k(
                reranked_ids,
                relevant_ids,
                FINAL_K,
            ),
            "candidate_recall": candidate_recall(
                candidate_ids,
                relevant_ids,
            ),
        }

        hybrid_metrics.append(hybrid_result)
        reranked_metrics.append(reranked_result)

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        print("\nHYBRID TOP-5")
        for rank, result in enumerate(hybrid_top_k, start=1):
            print(
                f"{rank}. {result.chunk.chunk_id} "
                f"(score={result.score:.6f})"
            )

        print("\nRERANKED TOP-5")
        for rank, result in enumerate(reranked, start=1):
            print(
                f"{rank}. {result.chunk.chunk_id} "
                f"(score={result.score:.4f})"
            )

        print(
            f"\nCandidate Recall@{CANDIDATE_K}: "
            f"{hybrid_result['candidate_recall']:.4f}"
        )

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "candidate_recall",
    ]

    hybrid_average = {
        metric: sum(row[metric] for row in hybrid_metrics)
        / len(hybrid_metrics)
        for metric in metric_names
    }

    reranked_average = {
        metric: sum(row[metric] for row in reranked_metrics)
        / len(reranked_metrics)
        for metric in metric_names
    }

    print("\n" + "=" * 80)
    print("AVERAGE BENCHMARK RESULTS")
    print("=" * 80)

    print_metrics("HYBRID", hybrid_average)
    print_metrics("HYBRID + CROSS-ENCODER", reranked_average)


if __name__ == "__main__":
    main()
