import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.api_models import RAGQueryRequest, RAGQueryResponse
from rag.rag_service import RAGService


rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    rag_service = RAGService()

    yield

    rag_service = None


app = FastAPI(
    title="Production RAG Platform",
    description="Production-oriented Retrieval-Augmented Generation platform.",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        result = rag_service.query(
            request.query
        )

        return RAGQueryResponse(
            **result
        )

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
    """Stream a RAG response using Server-Sent Events."""

    yield (
        "data: "
        + json.dumps(
            {
                "type": "status",
                "message": "Processing query...",
            }
        )
        + "\n\n"
    )

    if rag_service is None:
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "message": "RAG service is not ready.",
                }
            )
            + "\n\n"
        )
        return

    try:
        result = rag_service.query(query)

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "answer",
                    "answer": result["answer"],
                }
            )
            + "\n\n"
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "grounding",
                    "grounded": result["grounded"],
                }
            )
            + "\n\n"
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "sources",
                    "sources": result["sources"],
                }
            )
            + "\n\n"
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "complete",
                }
            )
            + "\n\n"
        )

    except ValueError as exc:
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
            + "\n\n"
        )

    except Exception:
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "message": "RAG query failed.",
                }
            )
            + "\n\n"
        )


@app.post(
    "/api/v1/rag/query/stream",
)
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
