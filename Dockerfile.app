# Base Linux image; install curl and uv via official installer
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential cmake ninja-build python3-dev libopenblas-dev libpq-dev pkg-config \
    git-lfs \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN cp /root/.local/bin/uv /usr/local/bin/uv \
 && cp /root/.local/bin/uvx /usr/local/bin/uvx \
 && chmod 755 /usr/local/bin/uv /usr/local/bin/uvx

# Download GGUF tokenizer/model files at build time (embed + chat), strip git metadata to keep image small
RUN git lfs install && \
    mkdir -p /models && \
    git clone --depth=1 https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF /models/nomic-embed-text-v1.5-GGUF && \
    rm -rf /models/nomic-embed-text-v1.5-GGUF/.git && \
    true

# Point the tokenizer hook to the embedding GGUF directory (file autodetected by llama.cpp)
ENV LLAMA_GGUF_PATH=/models/nomic-embed-text-v1.5-GGUF

WORKDIR /app

# Copy project definition first (better caching)
COPY pyproject.toml uv.lock .python-version /app/
RUN useradd -m -u 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser
ENV CMAKE_ARGS="-DLLAMA_METAL=OFF -DLLAMA_CUBLAS=OFF -DLLAMA_CLBLAST=OFF -DGGML_NATIVE=OFF -DGGML_CPU_DISABLE_DOTPROD=ON -DGGML_CPU_ENABLE_ARM_V8_2=OFF -DGGML_CPU_ENABLE_ARM_V8_6=OFF"
# Pull models before dependency sync to avoid extra rebuilds when only app code changes
RUN git lfs install && \
    mkdir -p /models && \
    git clone --depth=1 https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF /models/nomic-embed-text-v1.5-GGUF || true && \
    rm -rf /models/nomic-embed-text-v1.5-GGUF/.git || true

RUN uv python install 3.12 \
 && uv venv --python 3.12 --seed \
 && uv sync --frozen --python 3.12

# Copy the rest of the application
COPY --chown=appuser:appuser . /app

ENV UV_LINK_MODE=copy
# Allow directory value and auto-pick a file in hook
ENV LLAMA_GGUF_PATH=/models/nomic-embed-text-v1.5-GGUF
# user already set
EXPOSE 8000
CMD ["sh", "-lc", "/app/.venv/bin/python /app/main.py"]
