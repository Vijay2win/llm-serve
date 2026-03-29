#!/bin/bash
# install-minimax.sh — Deploy MiniMax M2.5 (230B MoE) on NVIDIA DGX Spark (GB10)
#
# Runtime: llama.cpp (GGUF format — not vLLM)
# Weights:  Unsloth MiniMax-M2.5-UD-Q3_K_XL (~101 GB, 4-part GGUF)
# Context:  196,608 tokens with tq3_0 KV cache (~9 GB overhead)
# GPU:      GB10 Blackwell sm_121 via GGML_CUDA
#
# NOTE: MiniMax M2.5 and Qwen3.5-122B cannot run simultaneously in 128 GB.
#       Stop vllm.service before starting minimax.service.
# Safe to re-run — every step is idempotent.

set -euo pipefail

LLAMA_SRC="$HOME/llama.cpp-src"
LLAMA_BUILD="$LLAMA_SRC/build"
LLAMA_BIN="$LLAMA_BUILD/bin"
MODEL_DIR="$HOME/.cache/minimax-m2.5"
MODEL_REPO="unsloth/MiniMax-M2.5-GGUF"
MODEL_SUBDIR="UD-Q3_K_XL"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CONDA_BIN="$HOME/miniconda3/envs/llm/bin"
PYTHON="$CONDA_BIN/python"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[install]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}    $*"; }
error()   { echo -e "${RED}[error]${NC}   $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── Preflight ────────────────────────────────────────────────────────────────
section "Preflight"

[[ "$(uname -m)" == "aarch64" ]] || warn "Expected aarch64 (GB10 Grace). Got $(uname -m) — proceed with caution."

if ! nvidia-smi -L 2>/dev/null | grep -q "GB10"; then
    warn "No GB10 GPU detected. Build targets sm_121 (Blackwell) — other GPUs may not work."
fi

[[ -x "$PYTHON" ]] || error "Python not found at $PYTHON. Create conda env first:
  conda create -n llm python=3.11 && conda activate llm"
[[ -d /usr/local/cuda ]] || error "/usr/local/cuda not found. Install CUDA Toolkit first."
command -v cmake >/dev/null 2>&1 || error "cmake not found. Install: sudo apt install cmake build-essential"
command -v ninja >/dev/null 2>&1 || warn "ninja not found — build will use make (slower). Install: sudo apt install ninja-build"

CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
info "CUDA: $CUDA_VER  |  Python: $($PYTHON --version)  |  Arch: $(uname -m)"

# Check huggingface_hub is available for download
"$PYTHON" -c "import huggingface_hub" 2>/dev/null \
    || "$PYTHON" -m pip install -q "huggingface_hub[hf_transfer]"

mkdir -p "$LOG_DIR"

# ── 1. Clone llama.cpp ───────────────────────────────────────────────────────
section "Step 1/4 — llama.cpp source"

if [[ -d "$LLAMA_SRC/.git" ]]; then
    info "llama.cpp source already cloned at $LLAMA_SRC."
else
    info "Cloning llama.cpp (latest main)..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
fi

# ── 2. Build llama.cpp for GB10 (sm_121 Blackwell) ──────────────────────────
section "Step 2/4 — Building llama.cpp for sm_121 (~10 min)"

LLAMA_SERVER="$LLAMA_BIN/llama-server"

if [[ -x "$LLAMA_SERVER" ]]; then
    info "llama-server already built at $LLAMA_SERVER."
else
    info "Configuring CMake for Blackwell (sm_121)..."
    cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DGGML_CPU_AARCH64=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee "$LOG_DIR/minimax-cmake.log"

    info "Building llama-server and llama-cli..."
    cmake --build "$LLAMA_BUILD" --config Release -j "$(nproc)" \
        --target llama-server llama-cli \
        2>&1 | tee "$LOG_DIR/minimax-build.log"

    [[ -x "$LLAMA_SERVER" ]] || error "Build failed — check $LOG_DIR/minimax-build.log"
    info "Build complete: $LLAMA_SERVER"
fi

# Verify CUDA is actually compiled in (use ldd — more reliable than parsing --version output)
info "Verifying CUDA build..."
if ldd "$LLAMA_SERVER" 2>/dev/null | grep -q "libggml-cuda\|libcudart"; then
    info "  ✓ CUDA backend present (libggml-cuda linked)"
else
    warn "  ✗ CUDA backend NOT found in ldd output — check build flags"
fi

# ── 3. Download MiniMax M2.5 UD-Q3_K_XL GGUF (~101 GB) ─────────────────────
section "Step 3/4 — Model download (~101 GB, 4 shards)"

mkdir -p "$MODEL_DIR"
FIRST_SHARD="$MODEL_DIR/$MODEL_SUBDIR/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf"
LAST_SHARD="$MODEL_DIR/$MODEL_SUBDIR/MiniMax-M2.5-UD-Q3_K_XL-00004-of-00004.gguf"

if [[ -f "$FIRST_SHARD" && -f "$LAST_SHARD" ]]; then
    info "Model already downloaded at $MODEL_DIR/$MODEL_SUBDIR/"
else
    info "Downloading $MODEL_REPO/$MODEL_SUBDIR/ ..."
    HF_HUB_ENABLE_HF_TRANSFER=1 "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_REPO',
    allow_patterns='$MODEL_SUBDIR/*.gguf',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('Download complete.')
" 2>&1 | tee "$LOG_DIR/minimax-download.log"

    [[ -f "$FIRST_SHARD" ]] || error "Download failed — check $LOG_DIR/minimax-download.log"
    info "Model downloaded to $MODEL_DIR/$MODEL_SUBDIR/"
fi

# ── 4. Services ───────────────────────────────────────────────────────────────
section "Step 4/4 — Systemd service"

SERVICE_SRC="$SCRIPT_DIR/minimax.service"
SERVICE_DST="$HOME/.config/systemd/user/minimax.service"
mkdir -p "$(dirname "$SERVICE_DST")"

if [[ -f "$SERVICE_SRC" ]]; then
    sed "s|/home/vijay|$HOME|g" "$SERVICE_SRC" > "$SERVICE_DST"
    systemctl --user daemon-reload
    systemctl --user enable minimax.service && info "minimax.service enabled."
else
    warn "minimax.service template not found at $SERVICE_SRC — skipping."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
section "Done"

cat <<EOF

${GREEN}Installation complete.${NC}

IMPORTANT — memory conflict: MiniMax M2.5 (~101 GB) and Qwen3.5-122B (~72 GB)
cannot run simultaneously in 128 GB unified memory.

Switch to MiniMax:
  systemctl --user stop vllm.service
  systemctl --user start minimax.service
  tail -f $LOG_DIR/minimax.log

Switch back to Qwen:
  systemctl --user stop minimax.service
  systemctl --user start vllm.service

Test MiniMax API (model loads in ~2 min):
  curl http://localhost:8001/health
  curl http://localhost:8001/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"minimax-m2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'

EOF
