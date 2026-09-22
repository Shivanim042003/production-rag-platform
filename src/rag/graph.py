from langgraph.graph import END, START, StateGraph

from rag.graph_nodes import (
    GenerateNode,
    GradeNode,
    GroundingNode,
    RerankNode,
    RetrieveNode,
)
from rag.graph_routing import route_after_grading
from rag.grounding_routing import route_after_grounding
from rag.graph_state import RAGState


def retry_node(state: RAGState) -> dict:
    """Increment retry count before another retrieval attempt."""
    return {
        "retry_count": state.retry_count + 1,
    }


def fallback_node(state: RAGState) -> dict:
    """Return a safe fallback answer."""
    return {
        "answer": "I don't have enough information.",
        "grounded": False,
    }


def build_rag_graph(
    retrieve_node: RetrieveNode,
    rerank_node: RerankNode,
    grade_node: GradeNode,
    generate_node: GenerateNode,
    grounding_node: GroundingNode,
    max_retries: int = 2,
):
    """Build the complete corrective RAG workflow."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("grade", grade_node)
    builder.add_node("retry", retry_node)
    builder.add_node("generate", generate_node)
    builder.add_node("ground", grounding_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade")

    builder.add_conditional_edges(
        "grade",
        lambda state: route_after_grading(
            state,
            max_retries=max_retries,
        ),
        {
            "generate": "generate",
            "retry": "retry",
            "fallback": "fallback",
        },
    )

    builder.add_edge("retry", "retrieve")
    builder.add_edge("generate", "ground")

    builder.add_conditional_edges(
        "ground",
        route_after_grounding,
        {
            "end": END,
            "fallback": "fallback",
        },
    )

    builder.add_edge("fallback", END)

    return builder.compile()
