FROM python:3.11-slim

ARG INSTALL_DEV=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch: the default wheel bundles several GB of CUDA libraries
# that are useless without a GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Third-party dependencies. This layer is cached until pyproject.toml changes.
COPY pyproject.toml README.md ./
RUN python -c "import os, tomllib; project = tomllib.load(open('pyproject.toml', 'rb'))['project']; deps = list(project['dependencies']); deps += project.get('optional-dependencies', {}).get('dev', []) if os.environ.get('INSTALL_DEV') == 'true' else []; open('/tmp/requirements.txt', 'w').write('\n'.join(deps))" \
    && pip install -r /tmp/requirements.txt

# Application code. Changing it only rebuilds from here.
COPY src ./src
COPY benchmarks ./benchmarks
RUN pip install --no-deps .

EXPOSE 8000

CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
