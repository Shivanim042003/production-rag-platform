import json

import pytest

from rag.rag_evaluation_dataset import (
    load_rag_evaluation_dataset,
)


def test_load_rag_evaluation_dataset_loads_all_cases() -> None:
    cases = load_rag_evaluation_dataset(
        "benchmarks/rag_evaluation.json"
    )

    assert len(cases) == 8


def test_load_rag_evaluation_dataset_preserves_first_case() -> None:
    cases = load_rag_evaluation_dataset(
        "benchmarks/rag_evaluation.json"
    )

    first = cases[0]

    assert first.query == (
        "What are PostgreSQL indexes used for?"
    )

    assert first.relevant_chunks == [
        "postgresql_chunk_001",
        "postgresql_chunk_002",
        "postgresql_performance_chunk_001",
    ]

    assert first.reference_answer


def test_load_rag_evaluation_dataset_preserves_last_case() -> None:
    cases = load_rag_evaluation_dataset(
        "benchmarks/rag_evaluation.json"
    )

    last = cases[-1]

    assert last.query == (
        "What factors affect Redis performance?"
    )

    assert last.relevant_chunks == [
        "redis_performance_chunk_001",
        "redis_performance_chunk_004",
    ]


def test_loader_rejects_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation dataset not found",
    ):
        load_rag_evaluation_dataset(missing_path)


def test_loader_rejects_non_list_dataset(tmp_path) -> None:
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps({"query": "question"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a JSON list",
    ):
        load_rag_evaluation_dataset(path)


def test_loader_rejects_non_object_record(tmp_path) -> None:
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps(["invalid"]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be an object",
    ):
        load_rag_evaluation_dataset(path)


def test_loader_rejects_missing_required_field(tmp_path) -> None:
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps(
            [
                {
                    "query": "question",
                    "reference_answer": "answer",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        load_rag_evaluation_dataset(path)


def test_loader_rejects_non_list_relevant_chunks(
    tmp_path,
) -> None:
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps(
            [
                {
                    "query": "question",
                    "reference_answer": "answer",
                    "relevant_chunks": "chunk_001",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="relevant_chunks must be a list",
    ):
        load_rag_evaluation_dataset(path)