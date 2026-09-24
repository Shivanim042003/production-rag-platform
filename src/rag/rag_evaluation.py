from dataclasses import dataclass, field


@dataclass
class RAGEvaluationCase:
    query: str
    reference_answer: str
    relevant_chunks: list[str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.reference_answer.strip():
            raise ValueError("Reference answer cannot be empty.")

        if not self.relevant_chunks:
            raise ValueError("Relevant chunks cannot be empty.")

        if any(not chunk.strip() for chunk in self.relevant_chunks):
            raise ValueError(
                "Relevant chunks cannot contain empty strings."
            )


@dataclass
class RAGEvaluationSample:
    query: str
    contexts: list[str]
    answer: str
    reference_answer: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.contexts:
            raise ValueError("Contexts cannot be empty.")

        if any(not context.strip() for context in self.contexts):
            raise ValueError(
                "Contexts cannot contain empty strings."
            )

        if not self.answer.strip():
            raise ValueError("Answer cannot be empty.")

        if (
            self.reference_answer is not None
            and not self.reference_answer.strip()
        ):
            raise ValueError(
                "Reference answer cannot be empty."
            )


@dataclass
class RAGEvaluationResult:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        metrics = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

        for name, value in metrics.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0"
                )