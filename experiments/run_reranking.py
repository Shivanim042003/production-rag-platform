from rag.bm25 import BM25Retriever
from rag.cross_encoder import CrossEncoderScorer
from rag.embeddings import EmbeddingModel
from rag.hybrid import HybridRetriever
from rag.ingestion import ingest_txt
from rag.reranker import Reranker
from rag.vector_store import FaissVectorStore


def main() -> None:
    chunks = (
        ingest_txt("data/postgresql.txt", chunk_size=50)
        + ingest_txt("data/redis.txt", chunk_size=50)
    )

    print(f"Loaded {len(chunks)} chunks.")

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

    scorer = CrossEncoderScorer()

    reranker = Reranker(scorer)

    queries = [
        "PostgreSQL indexes",
        "database query performance",
        "in-memory caching",
        "Redis key value data structures",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        candidates = hybrid.retrieve(
            query=query,
            top_k=2,
            dense_k=4,
            bm25_k=4,
        )

        print("\nHYBRID RETRIEVAL")
        for rank, result in enumerate(candidates, start=1):
            print(
                f"{rank}. {result.chunk.chunk_id} "
                f"(RRF={result.score:.6f})"
            )
            print(f"   {result.chunk.text}")

        reranked = reranker.rerank(
            query=query,
            results=candidates,
            top_k=2,
        )

        print("\nCROSS-ENCODER RERANKING")
        for rank, result in enumerate(reranked, start=1):
            print(
                f"{rank}. {result.chunk.chunk_id} "
                f"(score={result.score:.4f}, "
                f"previous_rank={result.metadata['previous_rank']})"
            )
            print(f"   {result.chunk.text}")


if __name__ == "__main__":
    main()
