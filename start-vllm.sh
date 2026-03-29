#!/bin/bash
# start-vllm.sh — vLLM inference server for Qwen3.5-122B-A10B-NVFP4
#
# Requires a custom-patched vLLM built for GB10 (sm_121a). Run install.sh first.
# Managed by vllm.service — stdout/stderr go to logs/vllm.log via systemd.

export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoid CUDA context corruption on fork
export PYTHONWARNINGS="ignore::UserWarning:vllm.model_executor.layers.fla"  # suppress false-positive shape warning on short prompts
export CUDA_VISIBLE_DEVICES=0
export PATH="$HOME/miniconda3/envs/llm/bin:/usr/local/cuda/bin:$PATH"

MODEL="RedHatAI/Qwen3.5-122B-A10B-NVFP4"

if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "[$(date)] vLLM already running, exiting."
    exit 0
fi

# GB10 unified memory: model mmap (~72 GB) stays in page cache after the previous
# run exits. vLLM's startup check counts this as "used" and aborts. Drop caches
# to restore full pool visibility (~119 GiB). Requires passwordless sudo — see README.
# Switch LiteLLM config to Qwen backend and reload
echo "[$(date)] Switching LiteLLM routing → Qwen3.5-122B (port 8000)..."
ln -sf litellm-config-qwen.yaml "$HOME/llm-serve/litellm-config.yaml"
systemctl --user reload-or-restart litellm.service 2>/dev/null || true

echo "[$(date)] Dropping page cache..."
sudo sysctl vm.drop_caches=3

echo "[$(date)] Starting vLLM..."

exec "$HOME/miniconda3/envs/llm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "qwen3.5-122b" \
    --max-model-len 1048576 \
    --gpu-memory-utilization 0.90 \
    --moe-backend cutlass \
    --attention-backend flashinfer \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --api-key "sk-spark-llm" \
    --limit-mm-per-prompt '{"image": 5, "video": 1}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --trust-remote-code
