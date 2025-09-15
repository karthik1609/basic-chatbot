# Base Linux image; install curl and uv via official installer
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates curl git netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /srv

COPY pyproject.toml uv.lock /srv/
RUN uv python install 3.13 \
 && uv venv --python 3.13 --seed \
 && uv sync --frozen --python 3.13

COPY . /srv

EXPOSE 3000
CMD ["sh", "-lc", "uv run --python 3.13 python server.py"]
