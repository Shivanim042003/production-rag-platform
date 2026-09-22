from langgraph.graph import END, START, StateGraph

from rag.graph_nodes import GradeNode, RerankNode, RetrieveNode
from rag.graph_routing import route_after_grading
from rag.graph_state import RAGState


def generate_placeholder(state: RAGState) -> dict:
    """Temporary generation node."""
    return {}


def retry_node(state: RAGState) -> dict:
    """Increment retry count before another retrieval attempt."""
    return {
        "retry_count": state.retry_count + 1,
    }


def fallback_placeholder(state: RAGState) -> dict:
    """Temporary fallback when retrieval remains insufficient."""
    return {
        "answer": "I don't have enough information.",
    }


def build_rag_graph(
    retrieve_node: RetrieveNode,
    rerank_node: RerankNode,
    grade_node: GradeNode,
    max_retries: int = 2,
):
    """Build the bounded corrective-retrieval workflow."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("grade", grade_node)
    builder.add_node("retry", retry_node)
    builder.add_node("generate", generate_placeholder)
    builder.add_node("fallback", fallback_placeholder)

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
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()
