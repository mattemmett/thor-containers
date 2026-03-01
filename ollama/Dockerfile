FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04 AS builder

# Install Go and build dependencies
RUN apt-get update && apt-get install -y \
    golang-go \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone Ollama v0.17.4
RUN git clone --depth 1 --branch v0.17.4 https://github.com/ollama/ollama /build/ollama

# Build with CUDA support
WORKDIR /build/ollama
RUN go generate ./... && \
    go build \
    -ldflags "-X=github.com/ollama/ollama/version.Version=0.17.4" \
    -o /usr/local/bin/ollama .

# Final image - same base, just replace the binary
FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

COPY --from=builder /usr/local/bin/ollama /usr/local/bin/ollama

