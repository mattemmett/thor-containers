FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04 AS builder

RUN apt-get update && apt-get install -y \
    golang-go git cmake build-essential zstd \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v0.20.3 https://github.com/ollama/ollama /build/ollama

WORKDIR /build/ollama

RUN cmake --preset 'CUDA 13' && \
    cmake --build --parallel $(nproc) --preset 'CUDA 13'

RUN go build \
    -ldflags "-X=github.com/ollama/ollama/version.Version=0.20.3" \
    -o /build/ollama/dist/bin/ollama .

FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

RUN rm /opt/ollama/ollama
COPY --from=builder /build/ollama/dist/bin/ollama /opt/ollama/ollama
COPY --from=builder /build/ollama/build/lib/ollama/ /opt/lib/ollama/cuda_v13/

ENV IS_SBSA=True
ENV LD_LIBRARY_PATH=/opt/ollama/build/lib/ollama:/opt/lib/ollama/cuda_v13:${LD_LIBRARY_PATH}

CMD ["/bin/bash", "-c", "exec ollama serve"]
