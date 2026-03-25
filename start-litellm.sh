#!/bin/bash
export PATH="$HOME/miniconda3/envs/llm/bin:$PATH"
export no_proxy=localhost,127.0.0.1,10.0.0.6
export NO_PROXY=localhost,127.0.0.1,10.0.0.6
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/litellm"

# Guard against duplicate instances
if pgrep -f "litellm.*4000|litellm.*config" | grep -v $$ > /dev/null; then
    echo "[$(date)] LiteLLM already running, exiting."
    exit 0
fi

echo "[$(date)] Starting LiteLLM proxy on 0.0.0.0:4000"

exec ~/miniconda3/envs/llm/bin/litellm \
    --config "$HOME/llm-serve/litellm-config.yaml" \
    --host 0.0.0.0 \
    --port 4000
