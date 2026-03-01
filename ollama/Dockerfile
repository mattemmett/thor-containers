FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

# Install Go for building Ollama from source
RUN apt-get update && apt-get install -y golang-go git curl && rm -rf /var/lib/apt/lists/*

# Build Ollama v0.17.4 from source
RUN git clone --depth 1 --branch v0.17.4 https://github.com/ollama/ollama /tmp/ollama && \
    cd /tmp/ollama && \
    go build -o /usr/local/bin/ollama . && \
    rm -rf /tmp/ollama
