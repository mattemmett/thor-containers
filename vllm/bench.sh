#!/usr/bin/env bash
#
# Benchmark helper for vLLM on Thor.
#
# Fires a warmup request (GPU clocks ramp under sustained load - without this
# you will measure a cold number ~35% low), then a long measured request.
#
# Watch the vLLM server terminal for:
#   "Avg generation throughput: X tokens/s"
#   "SpecDecoding metrics: Mean acceptance length: X"
#
# Requires jq.
#
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen3.8-27b}"
URL="http://${HOST}:${PORT}/v1/chat/completions"

req() {
    local prompt="$1" max_tokens="$2"
    jq -n \
        --arg m "$MODEL" \
        --arg p "$prompt" \
        --argjson t "$max_tokens" \
        '{model:$m, messages:[{role:"user",content:$p}], max_tokens:$t,
          chat_template_kwargs:{enable_thinking:false}}' \
    | curl -s "$URL" -H "Content-Type: application/json" -d @- > /dev/null
}

# Runs must be long enough to reach steady state. At ~28 tok/s an 800-token run
# is only 3 logging intervals, all of them still ramping. 4000 tokens gives ~14.
echo ">> Warmup (ignore this number - clocks are ramping)..."
req "Count to fifty." 400

echo ">> Measured run: code generation (~2.5 min)..."
req "Write a complete Python implementation of a B-tree with insert, delete, search, \
and range query. Include detailed docstrings and inline comments explaining the \
algorithm at each step, then write a thorough test suite covering edge cases." 4000

echo ">> Done."
echo "   Read 'Avg generation throughput' from the server terminal - take the"
echo "   plateau, not the peak. Expect oscillation as acceptance tracks content"
echo "   difficulty; tok/s should land near 8.4 x mean acceptance length."
