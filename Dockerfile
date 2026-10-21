
FROM python:3.11-slim

ARG INSTALL_DEV=false

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency metadata first
COPY pyproject.toml README.md ./

# Copy application source and benchmark documents
COPY src ./src
COPY benchmarks ./benchmarks

# Install production dependencies, or include development tools for testing
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        pip install --no-cache-dir ".[dev]"; \
    else \
        pip install --no-cache-dir .; \
    fi

EXPOSE 8000

CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
