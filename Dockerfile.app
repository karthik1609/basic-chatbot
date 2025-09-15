# Base Linux image; install curl and uv via official installer
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project definition first (better caching)
COPY pyproject.toml uv.lock /app/
RUN uv python install 3.13 \
 && uv venv --python 3.13 --seed \
 && uv sync --frozen --python 3.13

# Copy the rest of the application
COPY . /app

ENV UV_LINK_MODE=copy
EXPOSE 8000
CMD ["sh", "-lc", "uv run --python 3.13 python /app/main.py"]
