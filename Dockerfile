FROM python:3.12-slim

# true: include sentence-transformers with CPU-only torch, for local embeddings (a larger
# image). false: a small image that needs RAG_EMBEDDING_BACKEND=openai-compatible or hashing.
ARG LOCAL_EMBEDDINGS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
RUN if [ "$LOCAL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[local]"; \
    else \
        pip install --no-cache-dir .; \
    fi
COPY sample_data ./sample_data
COPY scripts ./scripts
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
