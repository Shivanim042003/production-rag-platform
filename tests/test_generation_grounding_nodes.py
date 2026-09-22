import pytest

from rag.generator import GenerationResult
from rag.graph_nodes import GenerateNode, GroundingNode
from rag.graph_state import RAGState
from rag.grounding import GroundingResult


class FakeGenerator:
    def __init__(self) -> None:
        self.received_args = None

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> GenerationResult:
        self.received_args = (query, context)

        return GenerationResult(
            answer="PostgreSQL indexes improve query performance.",
            model="fake-model",
        )


class FakeGroundingChecker:
    def __init__(
        self,
        result: GroundingResult,
    ) -> None:
        self.result = result
        self.received_args = None

    def check(
        self,
        answer: str,
        context: list[str],
    ) -> GroundingResult:
        self.received_args = (answer, context)
        return self.result


def test_generate_node_creates_answer() -> None:
    generator = FakeGenerator()
    node = GenerateNode(generator)

    state = RAGState(
        query="What are PostgreSQL indexes used for?",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
    )

    update = node(state)

    assert update == {
        "answer": "PostgreSQL indexes improve query performance."
    }


def test_generate_node_passes_query_and_context() -> None:
    generator = FakeGenerator()
    node = GenerateNode(generator)

    state = RAGState(
        query="How does Redis reduce database load?",
        context=[
            "Redis can reduce database load by serving values from memory.",
            "Redis is commonly used for caching.",
        ],
    )

    node(state)

    assert generator.received_args == (
        "How does Redis reduce database load?",
        [
            "Redis can reduce database load by serving values from memory.",
            "Redis is commonly used for caching.",
        ],
    )


def test_generate_node_rejects_empty_context() -> None:
    generator = FakeGenerator()
    node = GenerateNode(generator)

    state = RAGState(
        query="What are PostgreSQL indexes?",
        context=[],
    )

    with pytest.raises(
        ValueError,
        match="Cannot generate an answer without context",
    ):
        node(state)


def test_grounding_node_sets_grounded_true() -> None:
    checker = FakeGroundingChecker(
        GroundingResult(
            grounded=True,
            score=1.0,
            reason="Supported.",
        )
    )

    node = GroundingNode(checker)

    state = RAGState(
        query="What are PostgreSQL indexes?",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
        answer="PostgreSQL indexes improve query performance.",
    )

    update = node(state)

    assert update == {
        "grounded": True,
    }


def test_grounding_node_sets_grounded_false() -> None:
    checker = FakeGroundingChecker(
        GroundingResult(
            grounded=False,
            score=0.0,
            reason="Unsupported.",
        )
    )

    node = GroundingNode(checker)

    state = RAGState(
        query="What are PostgreSQL indexes?",
        context=[
            "Indexes improve PostgreSQL query performance."
        ],
        answer="PostgreSQL indexes reduce memory usage by 90%.",
    )

    update = node(state)

    assert update == {
        "grounded": False,
    }


def test_grounding_node_passes_answer_and_context() -> None:
    checker = FakeGroundingChecker(
        GroundingResult(
            grounded=True,
            score=1.0,
            reason="Supported.",
        )
    )

    node = GroundingNode(checker)

    answer = "Redis can reduce database load."
    context = [
        "Redis can reduce database load by serving values from memory."
    ]

    state = RAGState(
        query="How does Redis reduce database load?",
        context=context,
        answer=answer,
    )

    node(state)

    assert checker.received_args == (
        answer,
        context,
    )


def test_grounding_node_rejects_empty_answer() -> None:
    checker = FakeGroundingChecker(
        GroundingResult(
            grounded=True,
            score=1.0,
        )
    )

    node = GroundingNode(checker)

    state = RAGState(
        query="test",
        context=["test context"],
        answer=None,
    )

    with pytest.raises(
        ValueError,
        match="Cannot check grounding without an answer",
    ):
        node(state)


def test_grounding_node_rejects_empty_context() -> None:
    checker = FakeGroundingChecker(
        GroundingResult(
            grounded=True,
            score=1.0,
        )
    )

    node = GroundingNode(checker)

    state = RAGState(
        query="test",
        context=[],
        answer="test answer",
    )

    with pytest.raises(
        ValueError,
        match="Cannot check grounding without context",
    ):
        node(state)
