import json
from pathlib import Path

from rag.rag_evaluation import RAGEvaluationCase


def load_rag_evaluation_dataset(
    path: str | Path,
) -> list[RAGEvaluationCase]:
    """Load and validate RAG evaluation cases from JSON."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    cases: list[RAGEvaluationCase] = []

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(
                f"Evaluation record {index} must be an object."
            )

        required_fields = {
            "query",
            "reference_answer",
            "relevant_chunks",
        }

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise ValueError(
                f"Evaluation record {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        relevant_chunks = record["relevant_chunks"]

        if not isinstance(relevant_chunks, list):
            raise ValueError(
                f"Evaluation record {index} relevant_chunks "
                "must be a list."
            )

        cases.append(
            RAGEvaluationCase(
                query=record["query"],
                reference_answer=record["reference_answer"],
                relevant_chunks=relevant_chunks,
            )
        )

    return cases