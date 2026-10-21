
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.api_models import RAGQueryRequest, RAGQueryResponse
from rag.rag_service import RAGService


rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global rag_service

    try:
        rag_service = RAGService()
        yield
    finally:
        rag_service = None


app = FastAPI(
    title="Production RAG Platform",
    description="Production-oriented Retrieval-Augmented Generation platform.",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the application process is responding."""
    return {"status": "ok"}


@app.get("/ready")
def readiness() -> dict[str, str]:
    """Check whether the RAG service has initialized."""
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    return {"status": "ready"}


@app.post(
    "/api/v1/rag/query",
    response_model=RAGQueryResponse,
)
def query_rag(
    request: RAGQueryRequest,
) -> RAGQueryResponse:
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    try:
        result = rag_service.query(request.query)
        return RAGQueryResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="RAG query failed.",
        ) from exc


async def rag_event_stream(
    query: str,
) -> AsyncGenerator[str, None]:
    """Stream RAG response events using Server-Sent Events."""

    def encode_event(event: dict[str, Any]) -> str:
        return f"data: {json.dumps(event)}\n\n"

    yield encode_event({
        "type": "status",
        "message": "Processing query...",
    })

    if rag_service is None:
        yield encode_event({
            "type": "error",
            "message": "RAG service is not ready.",
        })
        return

    try:
        result = rag_service.query(query)

        yield encode_event({
            "type": "answer",
            "answer": result["answer"],
        })

        yield encode_event({
            "type": "grounding",
            "grounded": result["grounded"],
        })

        yield encode_event({
            "type": "sources",
            "sources": result["sources"],
        })

        yield encode_event({"type": "complete"})

    except ValueError as exc:
        yield encode_event({
            "type": "error",
            "message": str(exc),
        })

    except Exception:
        yield encode_event({
            "type": "error",
            "message": "RAG query failed.",
        })


@app.post("/api/v1/rag/query/stream")
async def stream_rag(
    request: RAGQueryRequest,
) -> StreamingResponse:
    return StreamingResponse(
        rag_event_stream(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
