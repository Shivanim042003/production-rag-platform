from fastapi import FastAPI

app = FastAPI(
    title="Production RAG Platform",
    description="Production-oriented Retrieval-Augmented Generation platform.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
