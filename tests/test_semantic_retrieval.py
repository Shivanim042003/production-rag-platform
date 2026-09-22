import pytest

from rag.embeddings import EmbeddingModel
from rag.ingestion import ingest_txt
from rag.vector_store import FaissVectorStore


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


def test_semantic_retrieval_finds_relevant_document(
    embedding_model: EmbeddingModel,
) -> None:
    postgresql_chunks = ingest_txt(
        "data/postgresql.txt",
        chunk_size=50,
        overlap=0,
    )

    redis_chunks = ingest_txt(
        "data/redis.txt",
        chunk_size=50,
        overlap=0,
    )

    chunks = postgresql_chunks + redis_chunks

    embeddings = embedding_model.embed_documents(chunks)

    store = FaissVectorStore(
        dimension=embedding_model.dimension,
    )

    store.add(chunks, embeddings)

    query_embedding = embedding_model.embed_query(
        "How do PostgreSQL indexes work?"
    )

    results = store.search(
        query_embedding,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0].document_id == "postgresql"
    assert results[0][1] >= results[1][1]
