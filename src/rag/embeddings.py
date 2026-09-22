from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.models import DocumentChunk


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 32,
    ) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
