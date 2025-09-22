# Base Linux image; install curl and uv via official installer
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates curl git netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN cp /root/.local/bin/uv /usr/local/bin/uv \
 && cp /root/.local/bin/uvx /usr/local/bin/uvx \
 && chmod 755 /usr/local/bin/uv /usr/local/bin/uvx

WORKDIR /srv

COPY pyproject.toml uv.lock /srv/
# Create non-root user before creating the venv, so the venv is owned by that user
RUN useradd -m -u 10002 mcpdocs \
 && chown -R mcpdocs:mcpdocs /srv
USER mcpdocs
RUN uv python install 3.13 \
 && uv venv --python 3.13 --seed \
 && uv sync --frozen --python 3.13

# Copy the rest as the app user
COPY --chown=mcpdocs:mcpdocs . /srv

EXPOSE 3000
CMD ["sh", "-lc", "uv run --python 3.13 python server.py"]
