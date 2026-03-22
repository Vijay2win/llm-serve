#!/bin/bash
# install.sh — Deploy Qwen3.5-122B-A10B-NVFP4 on NVIDIA DGX Spark (GB10)
#
# Clones vLLM, applies three GB10 patches, builds for sm_121a, configures
# the host, installs systemd services, and downloads the model.
# Safe to re-run — every step is idempotent.

set -euo pipefail

VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_COMMIT="e3126cd10"
VLLM_SRC="$HOME/vllm-src"
CONDA_ENV="llm"
CONDA_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin"
PYTHON="$CONDA_BIN/python"
PIP="$CONDA_BIN/pip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[install]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}    $*"; }
error()   { echo -e "${RED}[error]${NC}   $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── Preflight ────────────────────────────────────────────────────────────────
section "Preflight"

[[ "$(uname -m)" == "aarch64" ]] || warn "Expected aarch64 (GB10). Got $(uname -m) — proceed with caution."

if ! nvidia-smi -L 2>/dev/null | grep -q "GB10"; then
    warn "No GB10 GPU detected. These patches target sm_121a — other GPUs may not need them."
fi

[[ -x "$PYTHON" ]] || error "Python not found at $PYTHON. Create the conda env first:
  conda create -n $CONDA_ENV python=3.11 && conda activate $CONDA_ENV"

[[ -d /usr/local/cuda ]] || error "/usr/local/cuda not found. Install CUDA Toolkit first."

CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
info "CUDA: $CUDA_VER  |  Python: $($PYTHON --version)  |  Conda env: $CONDA_ENV"

# ── 1. Clone vLLM ────────────────────────────────────────────────────────────
section "Step 1/6 — vLLM source"

if [[ -d "$VLLM_SRC/.git" ]]; then
    info "vllm-src already exists, skipping clone."
else
    info "Cloning vLLM @ $VLLM_COMMIT ..."
    git clone "$VLLM_REPO" "$VLLM_SRC"
    git -C "$VLLM_SRC" checkout "$VLLM_COMMIT"
fi

# ── 2. Apply patches ─────────────────────────────────────────────────────────
section "Step 2/6 — Applying GB10 source patches"

# Patch A: add sm_121a to CUTLASS MoE data kernel arch list (CMakeLists.txt)
# The CUDA >= 13.0 path omitted 12.1a, causing an undefined symbol at link time.
CMAKEFILE="$VLLM_SRC/CMakeLists.txt"
BROKEN_MOE='cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")'
FIXED_MOE='cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f;12.1a" "${CUDA_ARCHS}")'

if grep -qF "$FIXED_MOE" "$CMAKEFILE"; then
    info "Patch A already applied."
elif grep -qF "$BROKEN_MOE" "$CMAKEFILE"; then
    sed -i "s|${BROKEN_MOE}|${FIXED_MOE}|" "$CMAKEFILE"
    info "Patch A applied."
else
    warn "Patch A: pattern not found — may already be upstream."
fi

# Patch B: route GB10 to FLA/Triton for GDN prefill (qwen3_next.py)
# is_device_capability(90) is an exact match — fails on sm=12.1.
# FlashInfer's GDN kernel uses Hopper TMA hardware absent on Blackwell.
# Excluding sm >= 12.0 routes GB10 to the FLA/Triton path (Triton JIT-compiles for sm_121a).
GDN_FILE="$VLLM_SRC/vllm/model_executor/models/qwen3_next.py"

if grep -qF "not current_platform.has_device_capability(120)" "$GDN_FILE"; then
    info "Patch B already applied."
else
    "$PYTHON" - "$GDN_FILE" <<'PYEOF'
import sys, re

path = sys.argv[1]
src = open(path).read()

broken = re.compile(
    r'supports_flashinfer\s*=\s*\(\s*\n'
    r'\s*current_platform\.is_cuda\(\)\s+and\s+current_platform\.is_device_capability\(90\)\s*\n'
    r'\s*\)',
    re.MULTILINE
)
partial = re.compile(
    r'supports_flashinfer\s*=\s*\(\s*\n'
    r'\s*current_platform\.is_cuda\(\)\s+and\s+current_platform\.has_device_capability\(90\)\s*\n'
    r'\s*\)',
    re.MULTILINE
)
replacement = (
    'supports_flashinfer = (\n'
    '        current_platform.is_cuda()\n'
    '        and current_platform.has_device_capability(90)\n'
    '        and not current_platform.has_device_capability(120)\n'
    '    )'
)

new_src, n = broken.subn(replacement, src)
if n == 0:
    new_src, n = partial.subn(replacement, src)
if n > 0:
    open(path, 'w').write(new_src)
    print(f"Patch B applied.")
else:
    print("Patch B: pattern not found — may already be upstream or manually patched.")
PYEOF
fi

# Patch C: restrict FlashAttn to sm < 12.0 for ViT (flash_attn.py)
# Without the upper bound, GB10 (sm=12.1) fell through to the wrong backend.
FLASH_FILE="$VLLM_SRC/vllm/v1/attention/backends/flash_attn.py"
BROKEN_FLASH='return DeviceCapability(8, 0) <= capability'
FIXED_FLASH='return DeviceCapability(8, 0) <= capability < DeviceCapability(12, 0)'

if grep -qF "$FIXED_FLASH" "$FLASH_FILE"; then
    info "Patch C already applied."
elif grep -qF "$BROKEN_FLASH" "$FLASH_FILE"; then
    sed -i "s/${BROKEN_FLASH}/${FIXED_FLASH}/" "$FLASH_FILE"
    info "Patch C applied."
else
    warn "Patch C: pattern not found — may already be upstream."
fi

# ── 3. Build ──────────────────────────────────────────────────────────────────
section "Step 3/6 — Building vLLM for sm_121a (~30 min)"

mkdir -p "$LOG_DIR"
(
    export PATH="$CONDA_BIN:/usr/local/cuda/bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="12.1a"
    export MAX_JOBS=16
    export CUDA_HOME=/usr/local/cuda
    cd "$VLLM_SRC"
    "$PIP" install -e . --no-build-isolation 2>&1 | tee "$LOG_DIR/vllm-build.log"
)

info "Verifying build symbols ..."
SO=$(find "$HOME/miniconda3/envs/$CONDA_ENV" -name "_C.abi3.so" 2>/dev/null | head -1)
[[ -z "$SO" ]] && error "_C.abi3.so not found — build failed. Check $LOG_DIR/vllm-build.log"

nm -D "$SO" 2>/dev/null | grep -q "scaled_fp4_quant_sm1" \
    && info "  ✓ NVFP4 kernel present" \
    || warn "  ✗ NVFP4 kernel NOT found — NVFP4 inference will fail"
nm -D "$SO" 2>/dev/null | grep -q "get_cutlass_moe_mm_data_caller" \
    && info "  ✓ MoE data kernel present" \
    || warn "  ✗ MoE data kernel NOT found — model will crash at load"

# ── 4. Host configuration ─────────────────────────────────────────────────────
section "Step 4/6 — Host configuration"

# 80 GB swapfile: model mmap (~72 GB) and CUDA weights (~72 GB) overlap during
# load. Without swap the OOM killer fires. NVMe speed makes this a brief pause.
SWAPFILE="/swapfile2"
if swapon --show | grep -q "$SWAPFILE"; then
    info "Swapfile already active."
elif [[ -f "$SWAPFILE" ]]; then
    sudo swapon "$SWAPFILE" && info "Swapfile activated."
else
    info "Creating 80 GB swapfile at $SWAPFILE ..."
    sudo fallocate -l 80G "$SWAPFILE"
    sudo chmod 600 "$SWAPFILE"
    sudo mkswap "$SWAPFILE"
    sudo swapon "$SWAPFILE"
    grep -q "$SWAPFILE" /etc/fstab || echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
    info "Swapfile created."
fi

# vm.swappiness=10: prefers evicting file-backed pages over CUDA allocations,
# but non-zero so swap is actually used during the loading spike.
SYSCTL_CONF="/etc/sysctl.d/99-vllm.conf"
if [[ -f "$SYSCTL_CONF" ]] && grep -q "vm.swappiness = 10" "$SYSCTL_CONF"; then
    info "vm.swappiness=10 already configured."
else
    echo -e "# GB10: prefer evicting file-backed pages over CUDA allocations\nvm.swappiness = 10" \
        | sudo tee "$SYSCTL_CONF" > /dev/null
    sudo sysctl --system -q
    info "vm.swappiness=10 applied."
fi

# Passwordless sudo for drop_caches (required by start-vllm.sh on each start)
SUDOERS_FILE="/etc/sudoers.d/vllm-drop-caches"
if [[ -f "$SUDOERS_FILE" ]]; then
    info "Sudoers entry already exists."
else
    echo "$(whoami) ALL=(ALL) NOPASSWD: /sbin/sysctl vm.drop_caches=*" \
        | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 0440 "$SUDOERS_FILE"
    info "Sudoers entry written."
fi

# ── 5. Services ───────────────────────────────────────────────────────────────
section "Step 5/6 — Systemd services"

mkdir -p "$LOG_DIR" "$HOME/.config/systemd/user"

install_service() {
    local name="$1"
    local src="$SCRIPT_DIR/$name.service"
    local dst="$HOME/.config/systemd/user/$name.service"
    [[ -f "$src" ]] || { warn "$name.service not found, skipping."; return; }
    sed "s|/home/vijay|$HOME|g" "$src" > "$dst"
    info "Installed $name.service"
}

install_service vllm
install_service litellm
systemctl --user daemon-reload
systemctl --user enable vllm.service && info "vllm.service enabled"
systemctl --user enable litellm.service 2>/dev/null && info "litellm.service enabled" || true

loginctl show-user "$(whoami)" 2>/dev/null | grep -q "Linger=yes" || {
    sudo loginctl enable-linger "$(whoami)"
    info "Linger enabled — services start at boot."
}

# ── 6. Model download ─────────────────────────────────────────────────────────
section "Step 6/6 — Model download"

MODEL_ID="RedHatAI/Qwen3.5-122B-A10B-NVFP4"

if find "$HOME/.cache/huggingface/hub" -name "*.safetensors" 2>/dev/null | grep -q "Qwen3.5-122B"; then
    info "Model already cached."
else
    info "Downloading $MODEL_ID (~72 GB) — this will take a while ..."
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir_use_symlinks=False)
" 2>&1 | tee "$LOG_DIR/download.log"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
section "Done"

cat <<EOF

${GREEN}Installation complete.${NC}

Start the server:
  systemctl --user start vllm.service
  tail -f $LOG_DIR/vllm.log

Test the API (model loads in ~2 min):
  curl http://localhost:8000/health
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"qwen3.5-122b","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'

EOF
