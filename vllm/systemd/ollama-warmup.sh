#!/usr/bin/env bash
# Wait for Ollama, then pin the embedding model in memory.
# Run from ollama.service ExecStartPost. Never fails the unit.
set -uo pipefail

MODEL="${MODEL:-nomic-embed-text}"
URL="${URL:-http://localhost:11434}"

for _ in {1..60}; do
    curl -sf "$URL/api/version" >/dev/null && break
    sleep 2
done

resp=$(curl -sf "$URL/api/embed" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"warmup\",\"keep_alive\":-1}" 2>&1)

if [[ -n "$resp" ]]; then
    echo "warmup: pinned $MODEL"
else
    echo "warmup: FAILED to pin $MODEL"
fi
exit 0
