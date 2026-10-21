
import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.api_models import RAGQueryRequest, RAGQueryResponse
from rag.rag_service import RAGService


logger = logging.getLogger("rag.api")

rag_service: RAGService | None = None
initialization_error: str | None = None
initialization_stage = "not_started"


async def initialize_service() -> None:
    """Initialize the RAG service in a background thread."""
    global rag_service, initialization_error, initialization_stage

    try:
        initialization_stage = "initializing"
        logger.info("Starting background RAG initialization")

        service = await asyncio.to_thread(RAGService)

        rag_service = service
        initialization_error = None
        initialization_stage = "ready"

        logger.info("RAG service is ready")

    except Exception:
        initialization_error = "RAG service initialization failed."
        initialization_stage = "failed"
        logger.exception("RAG service initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global rag_service, initialization_error, initialization_stage

    rag_service = None
    initialization_error = None
    initialization_stage = "initializing"

    initialization_task = asyncio.create_task(initialize_service())

    try:
        yield
    finally:
        initialization_task.cancel()

        with suppress(asyncio.CancelledError):
            await initialization_task

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
    """Check whether the API process is responding."""
    return {"status": "ok"}


@app.get("/startup")
def startup_status() -> dict[str, str]:
    """Expose the current RAG initialization state."""
    return {
        "stage": initialization_stage,
        "error": initialization_error or "",
    }


@app.get("/ready")
def readiness() -> dict[str, str]:
    """Check whether the RAG service has initialized."""
    if rag_service is None:
        detail = (
            initialization_error
            if initialization_error is not None
            else "RAG service is still initializing."
        )
        raise HTTPException(status_code=503, detail=detail)

    return {"status": "ready"}


def get_rag_service() -> RAGService:
    """Return the service or report that it is unavailable."""
    if rag_service is None:
        detail = (
            initialization_error
            if initialization_error is not None
            else "RAG service is still initializing."
        )
        raise HTTPException(status_code=503, detail=detail)

    return rag_service


@app.post(
    "/api/v1/rag/query",
    response_model=RAGQueryResponse,
)
def query_rag(
    request: RAGQueryRequest,
) -> RAGQueryResponse:
    service = get_rag_service()

    try:
        result = service.query(request.query)
        return RAGQueryResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=500,
            detail="RAG query failed.",
        ) from exc


async def rag_event_stream(
    query: str,
) -> AsyncGenerator[str, None]:
    def encode_event(event: dict[str, Any]) -> str:
        return f"data: {json.dumps(event)}\n\n"

    yield encode_event({
        "type": "status",
        "message": "Processing query...",
    })

    try:
        service = get_rag_service()
        result = await asyncio.to_thread(service.query, query)

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

    except HTTPException as exc:
        yield encode_event({
            "type": "error",
            "message": exc.detail,
        })

    except ValueError as exc:
        yield encode_event({
            "type": "error",
            "message": str(exc),
        })

    except Exception:
        logger.exception("Streaming RAG query failed")
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
