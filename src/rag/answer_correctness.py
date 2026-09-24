from dataclasses import dataclass

import numpy as np

from rag.embeddings import EmbeddingModel
from rag.models import DocumentChunk


@dataclass
class AnswerCorrectnessResult:
    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")


class EmbeddingAnswerCorrectnessScorer:
    """
    Measures semantic similarity between a generated answer
    and its reference answer.

    This is a baseline answer-correctness metric.

    It does NOT measure:
    - faithfulness to retrieved context
    - exact factual correctness
    - citation correctness
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.embedding_model = (
            embedding_model or EmbeddingModel()
        )

    def score(
        self,
        answer: str,
        reference_answer: str,
    ) -> AnswerCorrectnessResult:
        if not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not reference_answer.strip():
            raise ValueError(
                "Reference answer cannot be empty."
            )

        answer_chunk = DocumentChunk(
            chunk_id="generated-answer",
            document_id="generated-answer",
            text=answer,
            source="generated-answer",
        )

        reference_chunk = DocumentChunk(
            chunk_id="reference-answer",
            document_id="reference-answer",
            text=reference_answer,
            source="reference-answer",
        )

        embeddings = self.embedding_model.embed_documents(
            [
                answer_chunk,
                reference_chunk,
            ]
        )

        answer_embedding = embeddings[0]
        reference_embedding = embeddings[1]

        similarity = float(
            np.dot(
                answer_embedding,
                reference_embedding,
            )
        )

        similarity = max(
            0.0,
            min(1.0, similarity),
        )

        return AnswerCorrectnessResult(
            score=similarity,
            reason=(
                "Score represents semantic similarity between "
                "the generated answer and reference answer."
            ),
        )