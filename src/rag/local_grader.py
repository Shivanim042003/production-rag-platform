from rag.cross_encoder import CrossEncoderScorer
from rag.grader import RelevanceGrade


class LocalRelevanceScorer:
    """Scores relevance using a local cross-encoder."""

    def __init__(
        self,
        scorer: CrossEncoderScorer | None = None,
        threshold: float = 0.0,
    ) -> None:
        self.scorer = scorer or CrossEncoderScorer()
        self.threshold = threshold

    def score(
        self,
        query: str,
        text: str,
    ) -> RelevanceGrade:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not text.strip():
            raise ValueError("Document text cannot be empty.")

        scores = self.scorer.predict(
            [[query, text]]
        )

        if len(scores) != 1:
            raise RuntimeError(
                "Expected exactly one relevance score."
            )

        raw_score = float(scores[0])

        return RelevanceGrade(
            relevant=raw_score >= self.threshold,
            score=raw_score,
            reason=(
                f"Cross-encoder score {raw_score:.4f} "
                f"{'meets' if raw_score >= self.threshold else 'does not meet'} "
                f"the relevance threshold {self.threshold:.4f}."
            ),
        )
