# thor-containers/vllm

Running vLLM on NVIDIA Thor (L4T R38.x / JetPack 7.x, SBSA) using NVIDIA's official
NGC containers, with Thor-specific tuning notes.

## Why this is different from `ollama/`

Unlike the Ollama build, **no custom container is required**. NVIDIA publishes monthly
vLLM containers on NGC that already support Thor's SM 121 architecture, including native
NVFP4 tensor-core kernels. This directory documents configuration and tuning rather than
a build process.

## Container sources

| Source | Image | Notes |
|---|---|---|
| NGC (official, monthly) | `nvcr.io/nvidia/vllm:26.07-py3` | **Current choice.** vLLM 0.24.0, transformers 5.6.1, CUDA 13.3 |
| NVIDIA-AI-IOT | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` | vLLM 0.19.0, r38.2 SBSA base (same base as our Ollama build). Stale. |
| dusty-nv | `dustynv/vllm:*` | JetPack 6 / Orin only. No Thor builds. Useful for reading build recipes. |

NGC requires authentication:

    docker login nvcr.io
    # Username: $oauthtoken   (literal string, not your email)
    # Password: <NGC API key from https://org.ngc.nvidia.com/setup>

## Hardware target

| Property | Value |
|---|---|
| Board | NVIDIA Jetson AGX Thor DevKit |
| Host | 172.16.1.94 |
| L4T / JetPack | R38.4.0 / 7.1 |
| Driver | 580.00 (CUDA 13.0) |
| Compute Capability | SM 121 (reported as compute 11.0) |
| Unified Memory | 122.8 GB |
| Memory Bandwidth | ~273 GB/s <- **this is the binding constraint** |
| Power Mode | MAXN (mode 0) - already default, verified with `nvpmodel -q` |

## The bandwidth wall

Thor's performance ceiling for dense models is memory bandwidth, not compute.

Every token in a dense model requires reading the entire weight set through memory:

    24.57 GiB weights / 273 GB/s = ~11 tokens/sec theoretical maximum

Measured baseline was 10.8 tok/s - about 98% of theory. **The FP4 kernels are not the
problem; there is nothing to tune here.** The only way past this wall is speculative
decoding, which amortizes one weight read across multiple accepted tokens.

This is also why MoE models are a strong fit for Thor: GLM-4.7-Flash is 30B total but
only ~3B active per token, so it moves a fraction of the bytes per forward pass.

## Benchmark results

32K context, fp8 KV cache, `--gpu-memory-utilization 0.7`, single request,
thinking disabled, warmed up.

| Checkpoint | Config | tok/s | Mean accept length |
|---|---|---|---|
| Inferact (24.57 GiB) | no speculative decoding | 10.8 | - |
| Inferact | MTP-3 | 24.7 | 3.53 / 4 |
| Inferact | MTP-5 | 24.4 | 4.28 / 6 |
| unsloth (22.1 GiB) | MTP-3 + FlashInfer | **27.4** | ~3.2 avg |
| unsloth | MTP-3 + Triton attn | 27.8* | 3.23 |

\* single short run; within noise of the FlashInfer result. Not worth overriding
vLLM's auto-selection.

**Result: 2.5x from baseline** (10.8 -> 27.4), from one config flag plus a smaller
checkpoint.

### MTP depth: 3 is the sweet spot

Going to 5 draft tokens increases accepted length but degrades acceptance rate enough
to cancel out the gain:

    MTP-3 per-position:  0.943, 0.843, 0.743              -> 84.3%
    MTP-5 per-position:  0.877, 0.719, 0.649, 0.544, 0.491 -> 65.6%

Qwen's MTP is a **single layer run repeatedly**, so each draft position feeds on its own
previous guess and error compounds with depth. vLLM warns about this at startup. This is
exactly the structural limitation that block-diffusion drafters (DFlash2) are built to fix.

### The throughput model

Over a 2.5-minute sustained generation (14 intervals), throughput oscillated between
22.2 and 30.0 tok/s while acceptance length moved with it. Dividing one by the other
gives a constant:

    22.2 / 2.64 = 8.4        27.3 / 3.25 = 8.4
    26.4 / 3.11 = 8.5        29.1 / 3.46 = 8.4
    27.3 / 3.25 = 8.4        30.0 / 3.57 = 8.4

    tok/s  =  8.4  x  mean acceptance length

All the interval-to-interval variance is content difficulty moving acceptance around -
predictable text (docstrings, counting) drafts near 100%, novel code drops to ~55%.
Throughput itself is stable.

This makes future work predictable rather than speculative. Two independent levers:

- **The 8.4 constant** is the per-forward-pass rate. Theoretical ceiling is
  12.2 (22.1 GiB / 273 GB/s), so ~69% efficiency; the missing 31% is draft compute
  and verification overhead. A smaller checkpoint raises this proportionally -
  a 20.4 GB checkpoint would give roughly 9.1.
- **Acceptance length** is the multiplier. DFlash2's entire pitch is holding acceptance
  flat with depth instead of decaying. At depth 7 with acceptance ~5.0 that projects
  to **~42 tok/s**.

### Measurement methodology

- **Warm up first.** GPU clocks ramp under sustained load. The first measurement read
  6.9 tok/s and climbed to 10.8 mid-generation.
- **Generate at least ~2000 tokens.** Short runs finish before reaching steady state,
  and the problem gets worse as throughput improves - an 800-token run at 28 tok/s is
  only 3 logging intervals, all of them still ramping.
- **Disable thinking** (`"chat_template_kwargs": {"enable_thinking": false}`) or the
  model spends the entire budget in the `reasoning` field.
- **Vary prompt content.** Acceptance swings ~2x between predictable and novel text,
  so a single prompt is not representative.

## Required Thor-specific setup

### Drop page cache before every launch

Thor has **unified memory** - there is no separate VRAM pool. Linux page cache counts
against what vLLM sees as free GPU memory, so after reading a 24 GB checkpoint vLLM will
report only ~58 GB free of 122 GB and refuse to start:

    ValueError: Free memory on device cuda:0 (57.93/122.82 GiB) on startup is less
    than desired GPU memory utilization (0.7, 85.98 GiB).

Fix, run on the **host** (not in the container) before each launch:

    sudo sysctl -w vm.drop_caches=3

This is safe - it only discards clean cached copies of data still on disk. NVIDIA's own
Thor tutorials include this step. The systemd unit does it automatically.

### Docker run flags

vLLM needs shared memory for its worker processes:

    --ipc host
    --shm-size=16g
    --ulimit memlock=-1
    --ulimit stack=67108864

### CUDA minor version compatibility

The 26.07 container is built against CUDA 13.3 but JetPack 7.1 ships driver 580.00 with
CUDA 13.0. The container runs in Minor Version Compatibility mode and prints a warning.
It works, but if `no kernel image available` or similar kernel errors appear, fall back
to `26.01` or `26.04` (built against CUDA 13.1).

## Known gotchas

- **Thinking mode is on by default.** Qwen3.8 will spend its entire token budget in the
  `reasoning` field and return `"content": null` with `finish_reason: "length"`. Pass
  `"chat_template_kwargs": {"enable_thinking": false}` or budget generously.
- **Ctrl-C doesn't always clean up.** vLLM logs `destroy_process_group() was not called
  before program exit`. Check free memory before relaunching; usually it's page cache
  (see above) rather than a leaked process, but verify with `nvidia-smi`.
- **Uncalibrated fp8 KV scales.** The checkpoint ships no q/prob scaling factors, so vLLM
  falls back to 1.0 and warns about accuracy. If output quality seems degraded, drop
  `--kv-cache-dtype fp8` as the first diagnostic.
- **CUDA graphs downgrade to PIECEWISE** when speculative decoding is combined with the
  FlashInfer backend. Triton may support full capture - worth revisiting the backend
  comparison with that in mind.
- **Prefix caching is experimental** with this hybrid-attention model (Mamba cache
  'align' mode). Drop `--enable-prefix-caching` first if long sessions misbehave.
- **Driver can wedge.** After ~4 months uptime the nvidia runtime hung on container
  creation - new containers stuck in `Created`, `nvidia-smi` hung, containers unkillable
  (blocked in uninterruptible syscalls). Reboot cleared it.

## Models

| Model | HF ID | Size | Notes |
|---|---|---|---|
| Qwen3.8-27B NVFP4 (unsloth) | `unsloth/Qwen3.8-27B-NVFP4` | 22.1 GiB | **Current choice.** Dynamic V3.0 quant - mixed NVFP4/FP8, resolves as `compressed-tensors` |
| Qwen3.8-27B NVFP4 | `Inferact/Qwen3.8-27B-NVFP4` | 24.57 GiB | 9% larger, ~11% slower |
| Qwen3.8-27B DFlash2 drafter | `incoai/Qwen3.8-27B-DFlash2` | - | Requires unmerged vLLM PR 52816 |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | 62.5 GB bf16 | 30B-A3B MoE - only ~3B active/token, excellent fit for Thor's bandwidth limit. Reports of garbage output on vLLM/SGLang; llama.cpp works. |

Qwen3.8-27B is multimodal (`Qwen3_5ForConditionalGeneration`). Text serving is the
verified path. The vision encoder loaded and warmed up without issue on 0.24.0, despite
a known upstream bug about SM80-compiled FlashAttention crashing the encoder on Thor.

## Next steps

Flag-level tuning is exhausted. What remains is structural.

1. **DFlash2 speculative decoding** - the largest remaining gain, projected ~42 tok/s.
   Block-diffusion drafting proposes a whole block in one pass, so acceptance doesn't
   decay with depth the way single-layer MTP does. Not merged in vLLM (PR 52816);
   SGLang has it. Requires a patched vLLM build or switching engines.
   Drafter: `incoai/Qwen3.8-27B-DFlash2`.
2. **A smaller checkpoint.** The 8.4 constant scales inversely with weight size.
   A 20.4 GB NVFP4 build would give ~9.1, worth roughly +3 tok/s at current acceptance.
3. **SGLang comparison** - isolates engine overhead from checkpoint and config
   differences, and is the shortest path to DFlash2.
4. **GLM-4.7-Flash (30B-A3B MoE)** - structurally the best fit for Thor's bandwidth
   limit, since only ~3B params are active per token. Blocked on reports of garbage
   output under vLLM/SGLang; llama.cpp reportedly works.

### Tried and rejected

- **MTP depth 5** - acceptance rate degrades faster than accepted length improves.
- **`--attention-backend TRITON_ATTN`** - no measurable difference from vLLM's
  auto-selected FlashInfer.
- **Power tuning** - `nvpmodel` already reports MAXN (mode 0) by default.

## Reference

- vLLM recipe for this model: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- NGC vLLM release notes: https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/
- Jetson AI Lab: https://www.jetson-ai-lab.com/

## systemd quoting

systemd does not use a shell, so nested quotes in `ExecStart`/`ExecStartPost` get
mangled. This bit us twice:

- `--speculative-config {"method":"mtp",...}` lost its double quotes and arrived as
  `{method:mtp,...}` - invalid JSON, `status=2/INVALIDARGUMENT`. Fixed by wrapping
  the whole value in single quotes.
- A `curl -d "{\"model\":...}"` warmup call arrived malformed and returned 400,
  silently swallowed by `|| true`. Fixed by moving it to a script
  (`/usr/local/bin/ollama-warmup.sh`).

Rule of thumb: if a systemd command line needs escaped quotes, put it in a script
instead. Scripts are testable standalone; `ExecStart=` lines are not.

Also: any `ExecStartPost` health poll must be **bounded**. An unbounded
`until curl ...; do sleep 5; done` will keep the unit in `activating` forever if
`ExecStart` dies, and `systemctl start` will hang rather than report the failure.
