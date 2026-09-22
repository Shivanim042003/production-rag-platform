from dataclasses import dataclass, field

from rag.grader import RelevanceGrade
from rag.models import RetrievalResult


@dataclass
class RAGState:
    query: str

    transformed_queries: list[str] = field(default_factory=list)

    candidates: list[RetrievalResult] = field(default_factory=list)

    reranked_results: list[RetrievalResult] = field(default_factory=list)

    graded_results: list[
        tuple[RetrievalResult, RelevanceGrade]
    ] = field(default_factory=list)

    context: list[str] = field(default_factory=list)

    answer: str | None = None

    grounded: bool | None = None

    retry_count: int = 0

    sufficient_context: bool = False
