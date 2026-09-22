from collections.abc import Sequence

import faiss
import numpy as np

from rag.models import DocumentChunk, RetrievalResult


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match index dimension."
            )

        if not chunks:
            return

        vectors = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D array.")

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                "Query embedding dimension does not match index dimension."
            )

        if self.size == 0:
            return []

        query_vector = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32,
        )

        limit = min(top_k, self.size)

        scores, indices = self.index.search(query_vector, limit)

        results: list[RetrievalResult] = []

        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if index == -1:
                continue

            chunk = self.chunks[int(index)]

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    retriever="dense",
                    metadata={
                        "rank": rank,
                    },
                )
            )

        return results
