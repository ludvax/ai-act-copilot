# The image ships the code, not the corpus: data/index is built by `aiact ingest && aiact
# index` and mounted at runtime, so the image stays small and the index stays reproducible.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    AIACT_API_HOST=0.0.0.0

WORKDIR /app

# Dependencies first: this layer is cached until the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY prompts ./prompts
COPY data/sources.yaml ./data/sources.yaml
RUN uv sync --locked --no-dev

EXPOSE 8000

# Embeddings come from Ollama on the host; point AIACT_OLLAMA_BASE_URL at it.
CMD ["uv", "run", "--no-dev", "aiact", "serve"]
