from types import SimpleNamespace

from rag.rag_service import RAGService


def make_result(chunk_id: str, score: float):
    return SimpleNamespace(
        chunk=SimpleNamespace(chunk_id=chunk_id, text=chunk_id),
        score=score,
    )


class FakeGraph:
    def __init__(self, state: dict) -> None:
        self.state = state

    def invoke(self, initial_state: dict) -> dict:
        return self.state


def test_query_sources_only_include_relevant_chunks(monkeypatch) -> None:
    used = make_result("used_chunk", 7.7)
    skipped = make_result("skipped_chunk", -0.8)

    state = {
        "candidates": [used, skipped],
        "reranked_results": [used, skipped],
        "graded_results": [
            (used, SimpleNamespace(relevant=True)),
            (skipped, SimpleNamespace(relevant=False)),
        ],
        "answer": "An answer.",
        "grounded": True,
        "retry_count": 0,
    }

    service = object.__new__(RAGService)
    monkeypatch.setattr(
        service,
        "_build_graph",
        lambda trace, on_token=None: FakeGraph(state),
    )

    result = service.query("question")

    assert result["sources"] == [
        {"chunk_id": "used_chunk", "score": 7.7}
    ]
    assert result["diagnostics"]["reranked_count"] == 2
    assert result["diagnostics"]["relevant_context_count"] == 1
