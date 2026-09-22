from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag.models import DocumentChunk


class BM25Retriever:
    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)

        if self.chunks:
            tokenized_documents = [
                self._tokenize(chunk.text)
                for chunk in self.chunks
            ]

            self.bm25: BM25Okapi | None = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not self.chunks:
            return []

        tokenized_query = self._tokenize(query)

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_results = sorted(
            zip(self.chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (chunk, float(score))
            for chunk, score in ranked_results[:top_k]
        ]
