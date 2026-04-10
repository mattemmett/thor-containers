# thor-containers

Version-controlled custom container builds for NVIDIA Thor (L4T R38.4.0, JetPack 7.1, SBSA, CUDA 13.0).

## Why this exists

NVIDIA's official container ecosystem for Thor lags behind upstream projects by months.
This repo provides a reproducible build pipeline to stay current with upstream releases
while maintaining compatibility with Thor's unique hardware stack (SM 121, unified memory, SBSA).

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

| Directory | Description | Status |
|---|---|---|
| ollama/ | Ollama inference server with CUDA 13 GPU support | ✅ v0.17.4 |

## General principles

- All builds use `nvidia-ai-iot` base images to inherit L4T CUDA runtime libs
- CUDA backends are compiled natively on Thor — cross-compilation is not supported
- Build artifacts are cached by Docker layer; only changed stages rebuild
- Models and data live in `~/ollama-data` outside of containers

## Repository layout

```
thor-containers/
  ollama/
    Dockerfile      # Multi-stage build
    README.md       # Detailed build notes and lessons learned
    build.log       # Last build output (gitignored)
```

## Adding a new container

1. Create a subdirectory: `mkdir <service>`
2. Write a `Dockerfile` using an appropriate `nvidia-ai-iot` base image
3. Document build gotchas in a `README.md`
4. Test, tag, and commit
