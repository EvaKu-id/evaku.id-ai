FROM ghcr.io/astral-sh/uv:0.12.13 AS uv

FROM python:3.11-slim AS base
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS api-base
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev --extra api
RUN useradd --create-home --uid 10001 app && chown -R app:app /app/src /app/config
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8080
EXPOSE 8080

FROM api-base AS dev
CMD ["sh", "-c", "uvicorn evaku_ai.api.app:app --host 0.0.0.0 --port ${PORT} --reload --reload-dir /app/src"]

# cloud run
FROM api-base AS api
CMD ["sh", "-c", "uvicorn evaku_ai.api.app:app --host 0.0.0.0 --port ${PORT}"]

FROM base AS pipeline
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra pipeline --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev --extra pipeline
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["evaku-pipeline"]
CMD ["run", "--mode", "live"]
