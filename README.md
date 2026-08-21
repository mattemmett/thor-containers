# thor-containers

Container builds and deployment configuration for the NVIDIA Jetson AGX Thor DevKit
(L4T R38.4, JetPack 7.1, CUDA 13.0, SM 121).

Two kinds of thing live here:

- **Builds** — where upstream ships nothing usable on Thor, a Dockerfile that compiles
  it (see `ollama/`).
- **Configs** — where an official image exists, the tuning and systemd wiring to run it
  well on this hardware (see `vllm/`).

Both exist for the same reason: Thor is new enough and unusual enough (SBSA aarch64,
SM 121, unified memory) that upstream projects and NVIDIA's own container cadence
routinely lag behind what you actually want to run.

## Hardware

| Property | Value |
|---|---|
| Board | NVIDIA Jetson AGX Thor DevKit |
| Host | 172.16.1.94 |
| L4T / JetPack | R38.4.0 / 7.1 |
| Driver / CUDA | 580.00 / 13.0 |
| Compute Capability | SM 121 (reported as 11.0) |
| Unified Memory | 122.8 GB |
| Memory Bandwidth | ~273 GB/s |
| Storage | 937 GB NVMe |
| Power Mode | MAXN (mode 0, default) |

**Memory bandwidth is the binding constraint for inference on this box**, not compute.
A dense model must read its entire weight set per token, so ~22 GiB of weights caps you
near 12 tok/s no matter how good the kernels are. Everything in `vllm/` follows from
that fact.

## Contents

| Directory | What it is | Status | Image |
|---|---|---|---|
| `ollama/` | Custom Ollama build with CUDA 13 / SM 121 support | ✅ v0.20.3 | `ghcr.io/mattemmett/ollama:v0.20.3-r38-sbsa` |
| `vllm/` | vLLM config + tuning on NVIDIA's NGC container | ✅ 0.24.0 | `nvcr.io/nvidia/vllm:26.07-py3` (upstream) |

## Current deployment

Both run as systemd services, started at boot in dependency order:

| Service | Port | Serves |
|---|---|---|
| `vllm.service` | 8000 | `qwen3.8-27b` (Qwen3.8-27B-NVFP4), OpenAI-compatible, tool calling enabled |
| `ollama.service` | 11434 | `nomic-embed-text` embeddings only |

vLLM starts first and claims its memory; Ollama follows. See
`vllm/systemd/DEPLOY.md` for the full deployment, memory budget, and rationale.

## General principles

**For builds:**

- Use `nvidia-ai-iot` base images to inherit L4T CUDA runtime libraries
- Compile CUDA backends natively on Thor — cross-compilation is not supported
- Let Docker layer caching handle incremental rebuilds

**For configs:**

- Prefer official NGC containers where they exist; they already target SM 121
- Drop page cache before starting anything that checks free GPU memory — on unified
  memory, Linux page cache counts against it
- Put anything needing quotes into a script rather than a systemd `ExecStart=` line

**For both:**

- Models and data live outside containers in mounted volumes
- Document the dead ends, not just the working config

## Repository layout

    thor-containers/
      ollama/
        Dockerfile          # Multi-stage CUDA 13 build
        README.md           # Build notes and lessons learned
      vllm/
        README.md           # Tuning notes, benchmarks, throughput model
        serve-qwen38.sh     # Interactive / benchmarking launcher
        bench.sh            # Benchmark helper with warmup
        systemd/
          vllm.service
          ollama.service
          ollama-warmup.sh  # Pins the embedding model (installed to /usr/local/bin)
          DEPLOY.md         # Deployment guide and memory budget

## Adding something new

**If upstream has no working Thor image** — build it:

1. `mkdir <service>` and write a `Dockerfile` on an `nvidia-ai-iot` base
2. Document build gotchas in `<service>/README.md`
3. Test, tag, commit
4. Push to `ghcr.io/mattemmett/<service>:<version>-r38-sbsa`

**If an official image exists** — configure it:

1. `mkdir <service>` and document the working flags in `<service>/README.md`
2. Record what you tried that did *not* work, and why
3. Add a systemd unit under `<service>/systemd/` with a deployment note
