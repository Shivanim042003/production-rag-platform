
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to answer using the RAG system.",
    )


class RAGSource(BaseModel):
    chunk_id: str
    score: float


class RAGStageDiagnostic(BaseModel):
    name: str
    duration_ms: float


class RAGDiagnostics(BaseModel):
    candidate_count: int
    reranked_count: int
    relevant_context_count: int
    retry_count: int
    grounded: bool
    total_duration_ms: float
    stages: list[RAGStageDiagnostic]


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    sources: list[RAGSource]
    diagnostics: RAGDiagnostics


class RAGStreamEvent(BaseModel):
    type: str
    data: dict
