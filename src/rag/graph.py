from langgraph.graph import END, START, StateGraph

from rag.graph_nodes import GradeNode, RerankNode, RetrieveNode
from rag.graph_routing import route_after_grading
from rag.graph_state import RAGState


def generate_placeholder(state: RAGState) -> dict:
    """Temporary generation node used while building the graph."""
    return {}


def retry_placeholder(state: RAGState) -> dict:
    """Temporary retry node used while building the graph."""
    return {
        "retry_count": state.retry_count + 1,
    }


def build_rag_graph(
    retrieve_node: RetrieveNode,
    rerank_node: RerankNode,
    grade_node: GradeNode,
):
    """Build the conditional retrieval workflow."""

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_placeholder)
    builder.add_node("retry", retry_placeholder)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade")

    builder.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "retry": "retry",
        },
    )

    builder.add_edge("generate", END)
    builder.add_edge("retry", END)

    return builder.compile()
