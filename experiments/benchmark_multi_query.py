import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.bm25 import BM25Retriever
from rag.chunking import fixed_size_chunks
from rag.embeddings import EmbeddingModel
from rag.evaluation import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.hybrid import HybridRetriever
from rag.models import RawDocument
from rag.multi_query import QwenMultiQueryGenerator
from rag.multi_query_retriever import MultiQueryRetriever
from rag.vector_store import FaissVectorStore


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_gold():
    with open(
        "benchmarks/relevance.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_documents():
    documents = []

    for filename in [
        "postgresql.txt",
        "postgresql_performance.txt",
        "redis.txt",
        "redis_performance.txt",
    ]:
        path = f"benchmarks/data/{filename}"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        document_id = filename.removesuffix(".txt")

        documents.append(
            RawDocument(
                document_id=document_id,
                text=text,
                source=filename,
                metadata={},
            )
        )

    return documents


def build_chunks(documents):
    all_chunks = []

    for document in documents:
        chunks = fixed_size_chunks(
            document=document,
            chunk_size=35,
            overlap=5,
        )

        all_chunks.extend(chunks)

    return all_chunks


def get_relevant_ids(item):
    return set(item["relevant_chunks"])


def evaluate(results, relevant_ids, k=5):
    retrieved_ids = [
        result.chunk.chunk_id
        for result in results[:k]
    ]

    return {
        "precision": precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "recall": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "mrr": mean_reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
        "ndcg": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
    }


def main():
    print("Loading benchmark...")

    gold = load_gold()
    documents = load_documents()
    chunks = build_chunks(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")

    print("\nChunk IDs:")

    for chunk in chunks:
        print(f"  {chunk.chunk_id}")

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print("Building vector store...")

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension
    )

    embeddings = np.asarray(
        embedding_model.embed_documents(chunks),
        dtype=np.float32,
    )

    vector_store.add(
        chunks,
        embeddings,
    )

    # ---------------------------------------------------------
    # BM25 + Hybrid Retrieval
    # ---------------------------------------------------------

    print("Building BM25...")

    bm25 = BM25Retriever(chunks)

    hybrid = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

    # ---------------------------------------------------------
    # Qwen Multi-Query Generator
    # ---------------------------------------------------------

    print("\nLoading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    device = torch.device("cpu")
    model.to(device)

    generator = QwenMultiQueryGenerator(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=MODEL_NAME,
        num_queries=3,
    )

    multi_query = MultiQueryRetriever(
        retriever=hybrid,
        query_generator=generator,
    )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    metrics = []

    print("\n" + "=" * 70)
    print("MULTI-QUERY RETRIEVAL BENCHMARK")
    print("=" * 70)

    for item in gold:
        query = item["query"]
        relevant_ids = get_relevant_ids(item)

        print("\n" + "-" * 70)
        print(f"Query: {query}")

        results = multi_query.retrieve(
            query=query,
            top_k=5,
            dense_k=20,
            bm25_k=20,
        )

        result = evaluate(
            results,
            relevant_ids,
            k=5,
        )

        metrics.append(result)

        print(
            f"Precision@5: {result['precision']:.4f}"
        )

        print(
            f"Recall@5:    {result['recall']:.4f}"
        )

        print(
            f"MRR:         {result['mrr']:.4f}"
        )

        print(
            f"NDCG@5:      {result['ndcg']:.4f}"
        )

        print("\nRetrieved:")

        for rank, retrieved in enumerate(
            results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{retrieved.chunk.chunk_id}"
            )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(
        f"Precision@5: "
        f"{np.mean([m['precision'] for m in metrics]):.4f}"
    )

    print(
        f"Recall@5:    "
        f"{np.mean([m['recall'] for m in metrics]):.4f}"
    )

    print(
        f"MRR:         "
        f"{np.mean([m['mrr'] for m in metrics]):.4f}"
    )

    print(
        f"NDCG@5:      "
        f"{np.mean([m['ndcg'] for m in metrics]):.4f}"
    )


if __name__ == "__main__":
    main()
