# Deploying vLLM + Ollama as systemd services on Thor

Two units, ordered so vLLM allocates unified memory first and Ollama serves only
embeddings alongside it.

## Install

    sudo cp vllm.service ollama.service /etc/systemd/system/
    sudo systemctl daemon-reload

Stop the existing manually-run containers first, and remove Docker's restart policy
so the two don't fight over the same container name:

    docker stop ollama vllm 2>/dev/null
    docker rm ollama vllm 2>/dev/null

Then:

    sudo systemctl enable --now vllm.service
    sudo systemctl enable --now ollama.service

## Watch the first start

Cold start is slow (~6.5 min) - weights, torch.compile, FlashInfer autotune, and CUDA
graph capture. `systemctl start` will appear to hang; that's the `ExecStartPost` health
poll doing its job.

    journalctl -u vllm.service -f

The unit reaches `active (running)` only once `/health` answers.

## Why the ordering

vLLM performs a strict free-memory check at startup and aborts if it can't reserve its
full `--gpu-memory-utilization` share:

    ValueError: Free memory on device cuda:0 (57.93/122.82 GiB) on startup is less
    than desired GPU memory utilization (0.7, 85.98 GiB).

Ollama has no such check - it just allocates as models load. So if Ollama comes up first
and pulls in a large model, vLLM won't start at all. `ollama.service` therefore has
`After=vllm.service`, and `vllm.service` doesn't report started until it's serving.

`ExecStartPre=/sbin/sysctl -w vm.drop_caches=3` handles the other half: on unified memory
the kernel's page cache counts against vLLM's free-memory check, so a fresh boot or a
recent large file read can starve it.

## Memory budget

| Component | Size |
|---|---|
| vLLM weights (target + MTP drafter) | 22.1 GiB |
| KV cache @ 131K context, fp8 | 31.07 GiB |
| CUDA graphs | 0.47 GiB |
| Activation headroom (reserved) | ~32 GiB |
| Ollama + `nomic-embed-text` | ~1 GiB |
| **Total reserved** | **~86 GiB of 122.8** |

At 131,072 max context this supports **6.16x concurrency** - six simultaneous
full-length sessions. Real sessions rarely run at max context, so effective concurrency
is considerably higher.

If you want more KV cache (for 262K context, or more concurrent agents), raise
`--gpu-memory-utilization` toward 0.85 - Ollama's ~1 GiB leaves plenty of slack.

## Constraining Ollama

    OLLAMA_MAX_LOADED_MODELS=1
    OLLAMA_KEEP_ALIVE=-1

These keep Ollama in its lane. Without them, anything requesting `qwen3.5:35b` would try
to pull 32 GiB into a pool vLLM has already largely claimed. `KEEP_ALIVE=-1` pins the
embedding model so it never unloads and reloads mid-request.

**Point OpenClaw's agent models at vLLM (port 8000) and leave only `memorySearch`
pointed at Ollama (port 11434).**

## Client configuration

vLLM exposes an OpenAI-compatible API at `http://172.16.1.94:8000/v1`. No API key is set;
add `--api-key <key>` to the unit if you want one.

Model name is `qwen3.8-27b` (the `--served-model-name`), not the HF path.

**Thinking mode is on by default.** Clients that don't budget for it will get
`"content": null` with `finish_reason: "length"` because the whole budget went to the
`reasoning` field. Either allow generous `max_tokens` or pass:

    "chat_template_kwargs": {"enable_thinking": false}

Tool calling is enabled (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`), so
GasCity and other coding agents get proper structured `tool_calls` rather than raw text.

## Operations

    systemctl status vllm ollama
    journalctl -u vllm -f
    curl -s localhost:8000/v1/models | python3 -m json.tool
    curl -s localhost:8000/metrics | grep -E "throughput|running|waiting"

Restarting vLLM re-runs the cache drop automatically:

    sudo systemctl restart vllm

To change the model or flags, edit `/etc/systemd/system/vllm.service`, then
`daemon-reload` and `restart`.

## Known rough edges

- **Prefix caching is experimental** with this hybrid-attention model - vLLM warns that
  Mamba cache 'align' mode support is experimental. If long agent sessions produce odd
  output, drop `--enable-prefix-caching` first.
- **CUDA graphs are downgraded to PIECEWISE** because speculative decoding plus the
  FlashInfer backend doesn't support full capture. Triton may support it; worth retesting
  the backend comparison with that in mind.
- **Ollama's `ExecStartPost` warmup call is best-effort** (`|| true`). If
  `nomic-embed-text` isn't pulled yet it fails silently rather than blocking startup.
