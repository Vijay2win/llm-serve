#!/bin/bash
# =============================================================================
# start-litellm.sh — LiteLLM proxy (OpenAI-compatible gateway, port 4000)
#
# Managed by litellm.service. Logs are written exclusively by systemd via
# StandardOutput/StandardError=append: in the service unit — do NOT add
# tee here, as that would write every line twice.
# =============================================================================

export PATH="$HOME/miniconda3/envs/llm/bin:$PATH"
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/litellm"

# Ensure PostgreSQL is running (used for LiteLLM usage tracking)
if ! pg_ctl -D "$HOME/llm-serve/pgdata" status > /dev/null 2>&1; then
    echo "[$(date)] Starting PostgreSQL..."
    pg_ctl -D "$HOME/llm-serve/pgdata" -l "$HOME/llm-serve/logs/postgres.log" start
    sleep 3
fi

# Guard against duplicate instances (systemd Restart= can race with a slow exit)
if pgrep -f "litellm.*4000\|litellm.*config" > /dev/null; then
    echo "[$(date)] LiteLLM already running, exiting."
    exit 0
fi

echo "[$(date)] Starting LiteLLM proxy on 0.0.0.0:4000"

exec ~/miniconda3/envs/llm/bin/litellm \
    --config "$HOME/llm-serve/litellm-config.yaml" \
    --host 0.0.0.0 \
    --port 4000
