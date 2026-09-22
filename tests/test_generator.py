from dataclasses import fields

from rag.generator import GenerationResult


def test_generation_result_stores_answer_and_model() -> None:
    result = GenerationResult(
        answer="PostgreSQL indexes can improve query performance.",
        model="test-model",
    )

    assert result.answer == (
        "PostgreSQL indexes can improve query performance."
    )
    assert result.model == "test-model"


def test_generation_result_has_expected_fields() -> None:
    field_names = {
        field.name
        for field in fields(GenerationResult)
    }

    assert field_names == {"answer", "model"}
