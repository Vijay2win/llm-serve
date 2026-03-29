#!/bin/bash
# start-minimax.sh — llama.cpp inference server for MiniMax M2.5 (230B MoE)
#
# Model:   Unsloth MiniMax-M2.5-UD-Q3_K_XL (~101 GB, 4-part GGUF)
# Context: 196,608 tokens (native max_position_embeddings)
# KV:      tq3_turbo (3-bit TurboQuant WHT+Lloyd-Max) — ~10 GB for 196K context
# Port:    8001 (vLLM for Qwen uses 8000)
# Managed by minimax.service — stdout/stderr go to logs/minimax.log via systemd.

LLAMA_SERVER="$HOME/llama.cpp-src/build/bin/llama-server"
MODEL="$HOME/.cache/minimax-m2.5/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf"

if pgrep -f "llama-server.*8001" > /dev/null; then
    echo "[$(date)] llama-server already running on port 8001, exiting."
    exit 0
fi

# Guard: MiniMax (~101 GB) + Qwen (~72 GB) > 128 GB unified memory
if systemctl --user is-active --quiet vllm.service 2>/dev/null; then
    echo "[$(date)] ERROR: vllm.service is still running."
    echo "[$(date)] MiniMax M2.5 and Qwen3.5-122B cannot coexist in 128 GB."
    echo "[$(date)] Run: systemctl --user stop vllm.service"
    exit 1
fi

[[ -x "$LLAMA_SERVER" ]] || {
    echo "[$(date)] ERROR: llama-server not found at $LLAMA_SERVER"
    echo "[$(date)] Run install-minimax.sh first."
    exit 1
}

[[ -f "$MODEL" ]] || {
    echo "[$(date)] ERROR: model shard not found at $MODEL"
    echo "[$(date)] Run install-minimax.sh first."
    exit 1
}

# GB10 unified memory: drop page cache so prior model mmap doesn't inflate
# the apparent memory usage. Requires passwordless sudo (configured by install.sh).
# Switch LiteLLM config to MiniMax backend and reload
echo "[$(date)] Switching LiteLLM routing → MiniMax M2.5 (port 8001)..."
ln -sf litellm-config-minimax.yaml "$HOME/llm-serve/litellm-config.yaml"
systemctl --user reload-or-restart litellm.service 2>/dev/null || true

echo "[$(date)] Dropping page cache..."
sudo sysctl vm.drop_caches=3

echo "[$(date)] Starting llama-server (MiniMax M2.5 UD-Q3_K_XL)..."
echo "[$(date)] Context: 196,608 tokens | KV: tq3_turbo | Port: 8001"

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --alias "minimax-m2.5" \
    --ctx-size 196608 \
    --n-gpu-layers 999 \
    --no-mmap \
    --flash-attn on \
    --cache-type-k tq3_turbo \
    --cache-type-v tq3_turbo \
    --parallel 1 \
    --batch-size 8192 \
    --ubatch-size 4096 \
    --reasoning-budget 4096 \
    --host 0.0.0.0 \
    --port 8001 \
    --api-key "sk-spark-llm"
