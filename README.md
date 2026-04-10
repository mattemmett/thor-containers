# thor-containers

Reproducible container build pipeline for NVIDIA Thor DevKit (L4T R38.4, JetPack 7.1, CUDA 13.0). Bridges the gap between upstream project release cadence and NVIDIA's official container ecosystem.

## Hardware

| Property | Value |
|---|---|
| Board | NVIDIA Thor DevKit |
| Host | 172.16.1.94 |
| L4T | R38.4.0 |
| JetPack | 7.1 |
| CUDA | 13.0 |
| Compute Capability | SM 121 (11.0) |
| Unified Memory | 122 GB |
| Storage | 937 GB NVMe |

## Containers

| Directory | Description | Status | Image |
|---|---|---|---|
| ollama/ | Ollama inference server with CUDA 13 GPU support | ✅ v0.20.3 | `ghcr.io/mattemmett/ollama:v0.20.3-r38-sbsa` |

## General principles

- All builds use `nvidia-ai-iot` base images to inherit L4T CUDA runtime libs
- CUDA backends are compiled natively on Thor — cross-compilation is not supported
- Build artifacts are cached by Docker layer; only changed stages rebuild
- Models and data live outside of containers in a mounted volume

## Repository layout

    thor-containers/
      ollama/
        Dockerfile      # Multi-stage build
        README.md       # Detailed build notes and lessons learned
        build.log       # Last build output (gitignored)

## Adding a new container

1. Create a subdirectory: `mkdir <service>`
2. Write a `Dockerfile` using an appropriate `nvidia-ai-iot` base image
3. Document build gotchas in a `README.md`
4. Test, tag, and commit
5. Push image to `ghcr.io/mattemmett/<service>:<version>-r38-sbsa`
