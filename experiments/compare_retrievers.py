from rag.bm25 import BM25Retriever
from rag.embeddings import EmbeddingModel
from rag.ingestion import ingest_txt
from rag.vector_store import FaissVectorStore


def main() -> None:
    chunks = (
        ingest_txt("data/postgresql.txt", chunk_size=50)
        + ingest_txt("data/redis.txt", chunk_size=50)
    )

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(chunks)

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension
    )
    vector_store.add(chunks, embeddings)

    bm25 = BM25Retriever(chunks)

    queries = [
        "PostgreSQL indexes",
        "database query performance",
        "in-memory caching",
        "Redis key value data structures",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        dense_query = embedding_model.embed_query(query)
        dense_results = vector_store.search(
            dense_query,
            top_k=2,
        )

        bm25_results = bm25.retrieve(
            query,
            top_k=2,
        )

        print("\nDENSE / FAISS")
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            print(
                f"{rank}. {chunk.chunk_id} "
                f"(score={score:.4f})"
            )
            print(f"   {chunk.text}")

        print("\nBM25")
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            print(
                f"{rank}. {chunk.chunk_id} "
                f"(score={score:.4f})"
            )
            print(f"   {chunk.text}")


if __name__ == "__main__":
    main()
