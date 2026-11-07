# Streamlit app image: docker build -t trulens-pg-app . && docker run -p 8502:8502 \
#   -e TRULENS_PG_DATABASE_URL=... -e TRULENS_PG_AUTH=proxy trulens-pg-app
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY .git ./.git
RUN uv sync --frozen --no-dev --extra app --extra openai --extra litellm \
    && rm -rf .git
ENV PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN useradd --create-home app
USER app
EXPOSE 8502
ENTRYPOINT ["trulens-pg", "app", "--port", "8502"]
