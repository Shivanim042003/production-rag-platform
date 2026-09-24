from rag.answer_relevancy import AnswerRelevancyScorer
from rag.faithfulness import FaithfulnessScorer
from rag.rag_evaluation import (
    RAGEvaluationResult,
    RAGEvaluationSample,
)


class RAGEvaluator:
    """Interface for evaluating RAG responses."""

    def evaluate(
        self,
        sample: RAGEvaluationSample,
    ) -> RAGEvaluationResult:
        raise NotImplementedError


class BasicRAGEvaluator(RAGEvaluator):
    """
    Evaluation engine for RAG responses.

    Metric implementations are injected so the evaluator
    remains independent of a specific model.
    """

    def __init__(
        self,
        faithfulness_scorer: FaithfulnessScorer | None = None,
        answer_relevancy_scorer: AnswerRelevancyScorer | None = None,
    ) -> None:
        self.faithfulness_scorer = faithfulness_scorer
        self.answer_relevancy_scorer = answer_relevancy_scorer

    def evaluate(
        self,
        sample: RAGEvaluationSample,
    ) -> RAGEvaluationResult:
        faithfulness = None
        answer_relevancy = None

        if self.faithfulness_scorer is not None:
            faithfulness = self.faithfulness_scorer.score(
                answer=sample.answer,
                context=sample.contexts,
            )

        if self.answer_relevancy_scorer is not None:
            answer_relevancy_result = (
                self.answer_relevancy_scorer.score(
                    query=sample.query,
                    answer=sample.answer,
                )
            )

            answer_relevancy = answer_relevancy_result.score

        return RAGEvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            metadata={
                "query": sample.query,
                "num_contexts": len(sample.contexts),
                "has_reference_answer": (
                    sample.reference_answer is not None
                ),
            },
        )