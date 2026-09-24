from pathlib import Path

from rag.answer_correctness import (
    EmbeddingAnswerCorrectnessScorer,
)
from rag.answer_relevancy import EmbeddingAnswerRelevancyScorer
from rag.bm25 import BM25Retriever
from rag.cross_encoder import CrossEncoderScorer
from rag.embeddings import EmbeddingModel
from rag.evaluation import (
    candidate_recall,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.faithfulness import QwenFaithfulnessScorer
from rag.graph import build_rag_graph
from rag.graph_nodes import (
    GenerateNode,
    GradeNode,
    GroundingNode,
    QueryTransformNode,
    RerankNode,
    RetrieveNode,
)
from rag.grader import RelevanceGrader
from rag.grounding import GroundingChecker
from rag.hybrid import HybridRetriever
from rag.ingestion import ingest_txt
from rag.local_grader import LocalRelevanceScorer
from rag.qwen_generator import QwenLocalGenerator
from rag.qwen_grounding import QwenGroundingScorer
from rag.rag_evaluation import RAGEvaluationSample
from rag.rag_evaluation_dataset import load_rag_evaluation_dataset
from rag.rag_evaluator import BasicRAGEvaluator
from rag.reranker import Reranker
from rag.query_transformer import IdentityQueryTransformer
from rag.vector_store import FaissVectorStore


BENCHMARK_PATH = Path("benchmarks/rag_evaluation.json")
DATA_PATH = Path("benchmarks/data")

CHUNK_SIZE = 35
CHUNK_OVERLAP = 5

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

EVALUATION_K = 5


def load_chunks():
    """Load all benchmark documents."""

    chunks = []

    for path in sorted(DATA_PATH.glob("*.txt")):
        chunks.extend(
            ingest_txt(
                path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )

    if not chunks:
        raise RuntimeError(
            "No benchmark chunks were loaded."
        )

    return chunks


def create_rag_graph():
    """Build the complete RAG graph."""

    chunks = load_chunks()

    # ---------------------------------------------------------
    # Embeddings + vector store
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(
        chunks
    )

    vector_store = FaissVectorStore(
        dimension=embedding_model.dimension
    )

    vector_store.add(
        chunks,
        embeddings,
    )

    # ---------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    # ---------------------------------------------------------
    # Cross-encoder
    # ---------------------------------------------------------

    cross_encoder = CrossEncoderScorer()

    reranker = Reranker(
        scorer=cross_encoder,
    )

    relevance_scorer = LocalRelevanceScorer(
        scorer=cross_encoder,
        threshold=0.0,
    )

    relevance_grader = RelevanceGrader(
        scorer=relevance_scorer,
    )

    # ---------------------------------------------------------
    # Qwen generation
    # ---------------------------------------------------------

    generator = QwenLocalGenerator(
        model_name=MODEL_NAME,
        max_new_tokens=128,
    )

    # ---------------------------------------------------------
    # Qwen grounding
    # ---------------------------------------------------------

    grounding_scorer = QwenGroundingScorer(
        model_name=MODEL_NAME,
    )

    grounding_checker = GroundingChecker(
        scorer=grounding_scorer,
    )

    # ---------------------------------------------------------
    # Graph nodes
    # ---------------------------------------------------------

    retrieve_node = RetrieveNode(
        retriever=hybrid_retriever,
        top_k=10,
        dense_k=10,
        bm25_k=10,
    )

    rerank_node = RerankNode(
        reranker=reranker,
        top_k=5,
    )

    grade_node = GradeNode(
        grader=relevance_grader,
        min_relevant=1,
    )

    generate_node = GenerateNode(
        generator=generator,
    )

    grounding_node = GroundingNode(
        checker=grounding_checker,
    )

    query_transform_node = QueryTransformNode(
        transformer=IdentityQueryTransformer(),
    )

    return build_rag_graph(
        retrieve_node=retrieve_node,
        rerank_node=rerank_node,
        grade_node=grade_node,
        generate_node=generate_node,
        grounding_node=grounding_node,
        max_retries=2,
        query_transform_node=query_transform_node,
    )


def calculate_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> dict[str, float]:
    """Calculate retrieval metrics for one benchmark case."""

    return {
        "precision_at_5": precision_at_k(
            retrieved_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids,
            k=EVALUATION_K,
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids,
            k=EVALUATION_K,
        ),
        "mrr": mean_reciprocal_rank(
            retrieved_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids,
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids,
            k=EVALUATION_K,
        ),
        "candidate_recall": candidate_recall(
            candidate_ids=retrieved_chunk_ids,
            relevant_ids=relevant_chunk_ids,
        ),
    }


def evaluate_case(
    graph,
    case,
    evaluator,
    answer_correctness_scorer,
):
    """Run one benchmark case through the RAG pipeline."""

    initial_state = {
        "query": case.query,
        "transformed_queries": [],
        "candidates": [],
        "reranked_results": [],
        "graded_results": [],
        "context": [],
        "answer": None,
        "grounded": None,
        "retry_count": 0,
        "sufficient_context": False,
    }

    final_state = graph.invoke(
        initial_state
    )

    answer = final_state.get(
        "answer"
    )

    contexts = final_state.get(
        "context",
        [],
    )

    if not answer:
        raise RuntimeError(
            f"No answer generated for query: {case.query}"
        )

    if not contexts:
        raise RuntimeError(
            f"No context generated for query: {case.query}"
        )

    # ---------------------------------------------------------
    # Reranked retrieval results
    # ---------------------------------------------------------

    reranked_results = final_state.get(
        "reranked_results",
        [],
    )

    retrieved_chunk_ids = [
        result.chunk.chunk_id
        for result in reranked_results
    ]

    # ---------------------------------------------------------
    # Retrieval metrics
    # ---------------------------------------------------------

    retrieval_metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids=case.relevant_chunks,
    )

    # ---------------------------------------------------------
    # RAG evaluation sample
    # ---------------------------------------------------------

    sample = RAGEvaluationSample(
        query=case.query,
        contexts=contexts,
        answer=answer,
        reference_answer=case.reference_answer,
    )

    evaluation = evaluator.evaluate(
        sample
    )

    # ---------------------------------------------------------
    # Answer correctness
    # ---------------------------------------------------------

    answer_correctness = (
        answer_correctness_scorer.score(
            answer=answer,
            reference_answer=case.reference_answer,
        )
    )

    return {
        "query": case.query,
        "reference_answer": case.reference_answer,
        "answer": answer,
        "contexts": contexts,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "grounded": final_state.get(
            "grounded"
        ),
        "retry_count": final_state.get(
            "retry_count",
            0,
        ),
        "sufficient_context": final_state.get(
            "sufficient_context",
            False,
        ),
        "faithfulness": evaluation.faithfulness,
        "answer_relevancy": evaluation.answer_relevancy,
        "answer_correctness": answer_correctness.score,
        "retrieval_metrics": retrieval_metrics,
    }


def print_case_result(
    index: int,
    total: int,
    result: dict,
):
    """Print one case result."""

    metrics = result["retrieval_metrics"]

    print("\n" + "=" * 70)
    print(
        f"CASE {index}/{total}"
    )
    print("=" * 70)

    print("\nQUERY")
    print(result["query"])

    print("\nREFERENCE ANSWER")
    print(result["reference_answer"])

    print("\nGENERATED ANSWER")
    print(result["answer"])

    print("\nRETRIEVAL METRICS")

    print(
        f"Precision@5:     "
        f"{metrics['precision_at_5']:.4f}"
    )

    print(
        f"Recall@5:        "
        f"{metrics['recall_at_5']:.4f}"
    )

    print(
        f"MRR:             "
        f"{metrics['mrr']:.4f}"
    )

    print(
        f"NDCG@5:          "
        f"{metrics['ndcg_at_5']:.4f}"
    )

    print(
        f"Candidate Recall: "
        f"{metrics['candidate_recall']:.4f}"
    )

    print("\nGENERATION / GROUNDING")

    print(
        f"Grounded: "
        f"{result['grounded']}"
    )

    print(
        f"Sufficient context: "
        f"{result['sufficient_context']}"
    )

    print(
        f"Retry count: "
        f"{result['retry_count']}"
    )

    print(
        f"Faithfulness: "
        f"{result['faithfulness']:.2f}"
    )

    print(
        f"Answer Relevancy: "
        f"{result['answer_relevancy']:.2f}"
    )

    print(
        f"Answer Correctness: "
        f"{result['answer_correctness']:.4f}"
    )

    print(
        f"Context count: "
        f"{len(result['contexts'])}"
    )

    print("\nRERANKED CHUNK IDS")

    for chunk_id in result[
        "retrieved_chunk_ids"
    ]:
        print(
            f"- {chunk_id}"
        )


def print_summary(
    results: list[dict],
):
    """Print aggregate evaluation results."""

    total = len(results)

    grounded_count = sum(
        result["grounded"] is True
        for result in results
    )

    sufficient_context_count = sum(
        result["sufficient_context"] is True
        for result in results
    )

    total_retries = sum(
        result["retry_count"]
        for result in results
    )

    faithfulness_scores = [
        result["faithfulness"]
        for result in results
        if result["faithfulness"] is not None
    ]

    average_faithfulness = (
        sum(faithfulness_scores)
        / len(faithfulness_scores)
        if faithfulness_scores
        else None
    )

    answer_relevancy_scores = [
        result["answer_relevancy"]
        for result in results
        if result["answer_relevancy"] is not None
    ]

    average_answer_relevancy = (
        sum(answer_relevancy_scores)
        / len(answer_relevancy_scores)
        if answer_relevancy_scores
        else None
    )

    answer_correctness_scores = [
        result["answer_correctness"]
        for result in results
        if result["answer_correctness"] is not None
    ]

    average_answer_correctness = (
        sum(answer_correctness_scores)
        / len(answer_correctness_scores)
        if answer_correctness_scores
        else None
    )

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "candidate_recall",
    ]

    average_metrics = {}

    for metric_name in metric_names:
        values = [
            result["retrieval_metrics"][metric_name]
            for result in results
        ]

        average_metrics[metric_name] = (
            sum(values) / len(values)
        )

    # ---------------------------------------------------------
    # Aggregate report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGGREGATE RAG EVALUATION")
    print("=" * 70)

    print(
        f"\nCases evaluated: "
        f"{total}"
    )

    print(
        f"Grounded cases: "
        f"{grounded_count}/{total}"
    )

    print(
        f"Sufficient-context cases: "
        f"{sufficient_context_count}/{total}"
    )

    print(
        f"Total retries: "
        f"{total_retries}"
    )

    print("\nAVERAGE RETRIEVAL METRICS")

    print(
        f"Precision@5:      "
        f"{average_metrics['precision_at_5']:.4f}"
    )

    print(
        f"Recall@5:         "
        f"{average_metrics['recall_at_5']:.4f}"
    )

    print(
        f"MRR:              "
        f"{average_metrics['mrr']:.4f}"
    )

    print(
        f"NDCG@5:           "
        f"{average_metrics['ndcg_at_5']:.4f}"
    )

    print(
        f"Candidate Recall: "
        f"{average_metrics['candidate_recall']:.4f}"
    )

    print("\nGENERATION / GROUNDING")

    if average_faithfulness is not None:
        print(
            f"Average Qwen faithfulness: "
            f"{average_faithfulness:.2f}"
        )
    else:
        print(
            "Average Qwen faithfulness: N/A"
        )

    if average_answer_relevancy is not None:
        print(
            f"Average embedding answer relevancy: "
            f"{average_answer_relevancy:.2f}"
        )
    else:
        print(
            "Average embedding answer relevancy: N/A"
        )

    if average_answer_correctness is not None:
        print(
            f"Average embedding answer correctness: "
            f"{average_answer_correctness:.4f}"
        )
    else:
        print(
            "Average embedding answer correctness: N/A"
        )

    print("\nCASE SUMMARY")

    print(
        f"{'Case':<8}"
        f"{'P@5':<10}"
        f"{'R@5':<10}"
        f"{'MRR':<10}"
        f"{'NDCG@5':<10}"
        f"{'Cand.R':<10}"
        f"{'Grounded':<12}"
        f"{'Faithful':<10}"
        f"{'Ans.Rel.':<10}"
        f"{'Ans.Corr.':<12}"
    )

    print("-" * 105)

    for index, result in enumerate(
        results,
        start=1,
    ):
        metrics = result[
            "retrieval_metrics"
        ]

        print(
            f"{index:<8}"
            f"{metrics['precision_at_5']:<10.4f}"
            f"{metrics['recall_at_5']:<10.4f}"
            f"{metrics['mrr']:<10.4f}"
            f"{metrics['ndcg_at_5']:<10.4f}"
            f"{metrics['candidate_recall']:<10.4f}"
            f"{str(result['grounded']):<12}"
            f"{result['faithfulness']:<10.2f}"
            f"{result['answer_relevancy']:<10.2f}"
            f"{result['answer_correctness']:<12.4f}"
        )


def main() -> None:
    """Evaluate all benchmark cases."""

    print("=" * 70)
    print("PRODUCTION RAG — FULL EVALUATION")
    print("=" * 70)

    print("\nLoading evaluation dataset...")

    cases = load_rag_evaluation_dataset(
        BENCHMARK_PATH
    )

    if not cases:
        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    print(
        f"Loaded {len(cases)} evaluation cases."
    )

    print("\nBuilding RAG graph...")

    graph = create_rag_graph()

    # ---------------------------------------------------------
    # Faithfulness evaluator
    # ---------------------------------------------------------

    print(
        "\nLoading Qwen faithfulness evaluator..."
    )

    faithfulness_scorer = QwenFaithfulnessScorer(
        model_name=MODEL_NAME,
    )

    # ---------------------------------------------------------
    # Answer relevancy evaluator
    # ---------------------------------------------------------

    print(
        "\nLoading embedding answer relevancy evaluator..."
    )

    answer_relevancy_scorer = (
        EmbeddingAnswerRelevancyScorer()
    )

    # ---------------------------------------------------------
    # Answer correctness evaluator
    # ---------------------------------------------------------

    print(
        "\nLoading embedding answer correctness evaluator..."
    )

    answer_correctness_scorer = (
        EmbeddingAnswerCorrectnessScorer()
    )

    # ---------------------------------------------------------
    # Combined RAG evaluator
    # ---------------------------------------------------------

    evaluator = BasicRAGEvaluator(
        faithfulness_scorer=faithfulness_scorer,
        answer_relevancy_scorer=answer_relevancy_scorer,
    )

    # ---------------------------------------------------------
    # Evaluate all cases
    # ---------------------------------------------------------

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"\nRunning case "
            f"{index}/{len(cases)}..."
        )

        result = evaluate_case(
            graph=graph,
            case=case,
            evaluator=evaluator,
            answer_correctness_scorer=answer_correctness_scorer,
        )

        results.append(
            result
        )

        print_case_result(
            index=index,
            total=len(cases),
            result=result,
        )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    print_summary(
        results
    )


if __name__ == "__main__":
    main()