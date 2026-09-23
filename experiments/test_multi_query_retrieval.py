import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.bm25 import BM25Retriever
from rag.embeddings import EmbeddingModel
from rag.hybrid import HybridRetriever
from rag.models import DocumentChunk
from rag.multi_query import QwenMultiQueryGenerator
from rag.multi_query_retriever import MultiQueryRetriever
from rag.vector_store import FaissVectorStore


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_benchmark():
    with open("benchmarks/relevance.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    chunks = []

    for filename in [
        "postgresql.txt",
        "postgresql_performance.txt",
        "redis.txt",
        "redis_performance.txt",
    ]:
        path = f"benchmarks/data/{filename}"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks.append(
            DocumentChunk(
                chunk_id=filename,
                document_id=filename,
                text=text,
                source=filename,
                metadata={},
            )
        )

    return chunks


def main():
    print("Loading benchmark...")

    benchmark = load_benchmark()
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} benchmark documents.")

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

    print("Building BM25...")

    bm25 = BM25Retriever(chunks)

    hybrid = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

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

    print("\n" + "=" * 70)
    print("MULTI-QUERY RETRIEVAL EXPERIMENT")
    print("=" * 70)

    for item in benchmark:
        query = item["query"]

        print("\n" + "-" * 70)
        print(f"Query: {query}")

        generated_queries = generator.generate(query)

        print("\nGenerated queries:")

        for i, generated_query in enumerate(
            generated_queries,
            start=1,
        ):
            print(f"  {i}. {generated_query}")

        results = multi_query.retrieve(
            query=query,
            top_k=5,
            dense_k=10,
            bm25_k=10,
        )

        print("\nFinal retrieved chunks:")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{result.chunk.chunk_id} "
                f"(score={result.score:.4f})"
            )

    print("\n" + "=" * 70)
    print("Experiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
