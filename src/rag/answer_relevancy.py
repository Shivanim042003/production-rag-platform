from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.embeddings import EmbeddingModel
from rag.models import DocumentChunk


@dataclass
class AnswerRelevancyResult:
    relevant: bool
    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")


class AnswerRelevancyScorer(Protocol):
    def score(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevancyResult:
        ...


class EmbeddingAnswerRelevancyScorer:
    """
    Measures semantic similarity between a question and its answer.

    This provides a deterministic baseline for answer relevancy.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        relevant_threshold: float = 0.60,
        partially_relevant_threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= partially_relevant_threshold <= 1.0:
            raise ValueError(
                "partially_relevant_threshold must be between 0.0 and 1.0"
            )

        if not 0.0 <= relevant_threshold <= 1.0:
            raise ValueError(
                "relevant_threshold must be between 0.0 and 1.0"
            )

        if partially_relevant_threshold > relevant_threshold:
            raise ValueError(
                "partially_relevant_threshold cannot exceed "
                "relevant_threshold"
            )

        self.embedding_model = (
            embedding_model or EmbeddingModel()
        )

        self.relevant_threshold = relevant_threshold
        self.partially_relevant_threshold = (
            partially_relevant_threshold
        )

    def score(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevancyResult:
        if not query.strip():
            raise ValueError("query cannot be empty.")

        if not answer.strip():
            raise ValueError("answer cannot be empty.")

        query_embedding = self.embedding_model.embed_query(
            query
        )

        answer_chunk = DocumentChunk(
            chunk_id="answer",
            document_id="answer",
            text=answer,
            source="answer",
        )

        answer_embedding = (
            self.embedding_model.embed_documents(
                [answer_chunk]
            )[0]
        )

        similarity = float(
            np.dot(
                query_embedding,
                answer_embedding,
            )
        )

        similarity = max(
            0.0,
            min(1.0, similarity),
        )

        if similarity >= self.relevant_threshold:
            return AnswerRelevancyResult(
                relevant=True,
                score=1.0,
                reason=(
                    "The answer has high semantic similarity "
                    "to the question."
                ),
            )

        if similarity >= self.partially_relevant_threshold:
            return AnswerRelevancyResult(
                relevant=True,
                score=0.5,
                reason=(
                    "The answer is semantically related to "
                    "the question but may be incomplete."
                ),
            )

        return AnswerRelevancyResult(
            relevant=False,
            score=0.0,
            reason=(
                "The answer has low semantic similarity "
                "to the question."
            ),
        )