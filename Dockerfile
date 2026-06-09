FROM python:3.12.13-slim-trixie@sha256:ec948fa5f90f4f8907e89f4800cfd2d2e91e391a4bce4a6afa77ba265bc3a2fe AS base
ENV DEBIAN_FRONTEND=noninteractive \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PYTHON_DOWNLOADS=0 \
  STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  STREAMLIT_SERVER_PORT=8000 \
  PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN --mount=from=astral/uv:0.11.7-python3.12.13-trixie-slim@sha256:760df02ce4a80b395949f5ac7bf9741c5123fb829d9b62092363bfdca0088059,source=/usr/local/bin/uv,target=/bin/uv \
  --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  uv sync --locked --no-dev --all-extras --no-extra cu130

FROM base AS dev
CMD ["app"]

FROM base AS prod
COPY ./assistant/ ./assistant/
COPY settings.yaml ./
COPY ./db-docs/ ./db-docs/
CMD ["app"]
