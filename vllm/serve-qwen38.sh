#!/usr/bin/env bash
#
# Launch vLLM serving Qwen3.8-27B-NVFP4 on NVIDIA Thor.
#
# Run this on the HOST. It drops page cache (required on Thor's unified memory
# architecture) and then launches the NGC vLLM container.
#
# For production use the systemd unit instead (see systemd/DEPLOY.md).
# This script is for interactive work and benchmarking.
#
# Usage:
#   ./serve-qwen38.sh                 # production-equivalent config
#   ./serve-qwen38.sh --bench         # minimal config for benchmarking
#   MODEL=Inferact/Qwen3.8-27B-NVFP4 ./serve-qwen38.sh --bench
#
set -euo pipefail

# ---- configuration ---------------------------------------------------------

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.07-py3}"
MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
PORT="${PORT:-8000}"

# 32768 for benchmarking, 131072 for agentic work.
# Measured: 131072 gives 31.07 GiB KV cache = 808,052 tokens = 6.16x concurrency.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

# MTP-3 is the measured sweet spot. 5 draft tokens degrades acceptance enough
# to cancel the gain (see README).
SPEC_TOKENS="${SPEC_TOKENS:-3}"

HF_CACHE="${HF_CACHE:-$HOME/data/models/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/data/vllm_cache}"

# ---- pre-flight ------------------------------------------------------------

mkdir -p "$HF_CACHE" "$VLLM_CACHE"

# Thor's unified memory means page cache counts against "free GPU memory".
# Without this, vLLM refuses to start after a large checkpoint read.
echo ">> Dropping page cache (required on Thor unified memory)..."
sudo sysctl -w vm.drop_caches=3 >/dev/null
free -h | head -2

if command -v nvpmodel >/dev/null 2>&1; then
    echo ">> Power mode: $(sudo nvpmodel -q 2>/dev/null | head -1)"
fi

# ---- vLLM arguments --------------------------------------------------------

VLLM_ARGS=(
    serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "$SERVED_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --kv-cache-dtype fp8
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --reasoning-parser qwen3
    --max-num-batched-tokens 8192
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
)

# Production extras - omitted in --bench mode to keep comparisons clean
if [[ "${1:-}" != "--bench" ]]; then
    VLLM_ARGS+=(
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --enable-prefix-caching
    )
fi

# ---- launch ----------------------------------------------------------------

echo ">> Serving $MODEL as '$SERVED_NAME' on port $PORT"

exec docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$VLLM_CACHE:/root/.cache/vllm" \
    --name vllm-manual \
    "$IMAGE" \
    vllm "${VLLM_ARGS[@]}"
