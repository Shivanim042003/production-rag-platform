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


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    sources: list[RAGSource]


class RAGStreamEvent(BaseModel):
    type: str
    data: dict