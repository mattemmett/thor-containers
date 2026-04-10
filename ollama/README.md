# thor-containers/ollama

Custom Ollama build for NVIDIA Thor (L4T R38.x / JetPack 7.x, SBSA).

## Why this exists

The official `nvidia-ai-iot/ollama` container is rebuilt infrequently (~6 month cadence).
Upstream Ollama releases new model architecture support (e.g. `qwen3.5`, `qwen35moe`) far faster
than nvidia-ai-iot publishes new images. This repo provides a build pipeline to stay current.

## Hardware target

| Property | Value |
|---|---|
| Board | NVIDIA Thor DevKit |
| L4T | R38.4.0 |
| JetPack | 7.1 |
| CUDA | 13.0 |
| Compute Capability | SM 121 (11.0) |
| Unified Memory | 122 GB |

## How it works

The build uses the `nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04` image as both the
build environment and runtime base. This gives us:

- nvcc (CUDA 13.0) for compiling the CUDA backend
- All L4T-specific CUDA runtime libraries
- The correct SBSA library paths

### Build stages

**Stage 1 (builder):**
1. Install Go, CMake, build-essential
2. Clone Ollama at the target version tag
3. Run `cmake --preset 'CUDA 13'` — this compiles `libggml-cuda.so` targeting SM 121
   (among other architectures). Takes ~17 minutes on Thor's 14-core Grace CPU.
4. Build the Go binary with the correct version string

**Stage 2 (runtime):**
1. Replace `/opt/ollama/ollama` (the symlink target) with the new binary
2. Copy compiled CUDA libs to `/opt/lib/ollama/cuda_v13/`

### Why the lib path matters

Ollama's `ml.LibOllamaPath` resolves to `../lib/ollama` relative to the **real** binary path
(after symlink resolution). Since the binary lives at `/opt/ollama/ollama`, it looks for backends
in `/opt/lib/ollama/`. It then globs for `*/libggml-*` subdirectories — so libs must be in a
named subdirectory like `cuda_v13/`, not directly in `/opt/lib/ollama/`.

At runtime you'll see:
```
libdirs=ollama,cuda_v13
load_backend: loaded CUDA backend from /opt/lib/ollama/cuda_v13/libggml-cuda.so
```

### Why not JetPack preset?

Upstream Ollama only has `JetPack 5` (SM 72/87) and `JetPack 6` (SM 87) presets.
Thor is JetPack 7 / SM 121, which upstream doesn't have a preset for.
The `CUDA 13` preset targets `75-virtual` through `121-virtual` and works correctly.

## Building

```bash
docker build -t ollama:v0.17.4-r38-sbsa .
```

First build takes ~20 minutes (CUDA compilation). Subsequent builds of the same Ollama version
are fully cached and take ~5 minutes (Go build only).

To update to a new Ollama version, change the `--branch` tag in the Dockerfile clone step and
rebuild with `--no-cache` for the Go/CMake steps only (the base image layer is still cached).

## Running

```bash
docker run -d \
  --runtime nvidia \
  --network host \
  -v ~/ollama-data:/data \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_MODELS=/data/models/ollama/models \
  --name ollama \
  --restart unless-stopped \
  ollama:v0.17.4-r38-sbsa \
  /bin/bash -c "exec ollama serve"
```

`IS_SBSA=True` and the required `LD_LIBRARY_PATH` are baked into the image via `ENV` directives.

## Verified models

| Model | Size | VRAM used | Layers offloaded |
|---|---|---|---|
| llama3.2:3b | 1.9 GB | 22 GB | 29/29 |
| qwen3.5:35b | 24 GB | 32 GB | 41/41 |

## Updating Ollama version

1. Change `--branch vX.Y.Z` in the Dockerfile
2. `docker build -t ollama:vX.Y.Z-r38-sbsa .`
3. Stop old container, start new one
4. Models are preserved in `~/ollama-data` (shared volume)

## What does NOT work (and why)

- **`OLLAMA_LLM_LIBRARY=cuda_v13`**: This was the old runner's mechanism. The new Go-native
  engine ignores it entirely — library selection happens through directory scanning, not this env var.
- **`go generate ./...`**: Produces nothing useful on this platform. The CUDA backend must be
  built explicitly via CMake.
- **Placing libs directly in `/usr/local/lib/ollama/`**: Ollama expects a subdirectory (e.g.
  `cuda_v13/`), not libs at the top level.
- **Overwriting `/usr/local/bin/ollama` via COPY**: That path is a symlink to `/opt/ollama/ollama`.
  Docker COPY replaces the symlink but the resolved path logic still uses the symlink target
  for `LibOllamaPath` calculation, pointing to the wrong lib directory.
- **Cross-compiling on x86**: The CUDA backend must be compiled natively on aarch64 with the
  L4T CUDA toolkit. The Thor is the build machine.

## Pre-built image

A pre-built image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/mattemmett/ollama:v0.20.3-r38-sbsa
```

Note: this image is built for aarch64 (SBSA) only. It will not run on x86.
