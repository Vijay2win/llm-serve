# Qwen3.5-122B on NVIDIA DGX Spark (GB10)

Run [RedHatAI/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) on the DGX Spark with an OpenAI-compatible API, 200K context, and chain-of-thought reasoning.

The DGX Spark's GB10 chip (sm_121a) requires three vLLM source patches that aren't upstream yet. This repo documents each bug, provides a one-command installer, and includes systemd services for auto-start on boot.

---

## Hardware

| | |
|---|---|
| SoC | NVIDIA GB10 (Blackwell, sm_121a) |
| Memory | 119 GiB unified LPDDR5X — CPU and GPU share one pool, no discrete VRAM |
| CUDA | 13.x |
| OS | Ubuntu 24.04 (aarch64) |

---

## Prerequisites

- **Conda environment** named `llm` with Python 3.11
  `conda create -n llm python=3.11 && conda activate llm`
- **CUDA Toolkit 13.x** at `/usr/local/cuda`
- **`sudo` access** (for swapfile, sysctl, and sudoers entry)
- **~200 GB free disk** (72 GB model + build artifacts)

---

## Quick Start

```bash
git clone <this-repo> llm-serve
cd llm-serve
./install.sh
```

`install.sh` takes ~30–40 minutes (mostly the vLLM build). Once done:

```bash
systemctl --user start vllm.service
tail -f logs/vllm.log        # ready when you see "Application startup complete"
curl http://localhost:8000/health
```

---

## What `install.sh` Does

1. Clones vLLM from source at a tested commit
2. Applies three GB10 patches (see below)
3. Builds the wheel with `TORCH_CUDA_ARCH_LIST="12.1a"`
4. Verifies that NVFP4 and MoE symbols are present in `_C.abi3.so`
5. Creates an 80 GB swapfile and sets `vm.swappiness=10`
6. Installs and enables `vllm.service` and `litellm.service`
7. Downloads the model (~72 GB) if not already cached

---

## The Three Patches

These are the bugs preventing the model from running on GB10 out of the box. All three are applied automatically by `install.sh`.

### Patch A — Missing sm_121a in CUTLASS MoE Kernel

**File:** `CMakeLists.txt` (~line 821, CUDA ≥ 13.0 path)

**Error:**
```
ImportError: undefined symbol: get_cutlass_moe_mm_data_caller
```

**Root cause:** `CUTLASS_MOE_DATA_ARCHS` listed `"9.0a;10.0f;11.0f;12.0f"` in the CUDA ≥ 13.0 path — `12.1a` was missing. The FP4 block sets `-DENABLE_CUTLASS_MOE_SM120=1`, which causes `scaled_mm_entry.cu` to reference `get_cutlass_moe_mm_data_caller()`. Since `moe_data.cu` was never compiled for `sm_121a`, the symbol doesn't exist. (The CUDA < 13.0 path already had `12.1a` — this was a CUDA 13.0-specific oversight.)

```cmake
# Before:
cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")

# After:
cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f;12.1a" "${CUDA_ARCHS}")
```

---

### Patch B — GDN Backend Crashes on First Inference

**File:** `vllm/model_executor/models/qwen3_next.py` (~line 178)

**Error:**
```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```
Stack trace points to `chunk_gated_delta_rule` in the FLA/Triton kernel.

**Root cause:** The GDN prefill backend is selected with `is_device_capability(90)` — an **exact** sm=9.0 match. GB10 is sm=12.1, so this returns `False` and falls back to a Triton kernel that isn't compiled for sm_121a.

The naive fix (`has_device_capability(90)`, a ≥ check) routes GB10 to FlashInfer's `gdn_prefill_sm90` kernel — but that kernel uses Hopper-specific TMA hardware that doesn't exist on Blackwell and raises `"initialize failed"`. The correct fix excludes Blackwell so GB10 uses the FLA/Triton path, which Triton 3.x JIT-compiles natively for sm_121a.

```python
# Before (exact sm=9.0 match — False on GB10):
supports_flashinfer = (
    current_platform.is_cuda() and current_platform.is_device_capability(90)
)

# After (Hopper+ but not Blackwell — routes GB10 to FLA/Triton):
supports_flashinfer = (
    current_platform.is_cuda()
    and current_platform.has_device_capability(90)
    and not current_platform.has_device_capability(120)
)
```

---

### Patch C — Wrong Attention Backend for ViT

**File:** `vllm/v1/attention/backends/flash_attn.py` (~line 186)

**Root cause:** `supports_compute_capability` had no upper bound, so GB10 (sm=12.1) fell through to FlashAttn instead of TRITON_ATTN.

```python
# Before (no upper bound):
return DeviceCapability(8, 0) <= capability

# After (FlashAttn bounded to sm < 12.0):
return DeviceCapability(8, 0) <= capability < DeviceCapability(12, 0)
```

---

## GB10 Memory — Why It Needs Extra Care

GB10 has no discrete VRAM. The CPU and GPU share a single 119 GiB LPDDR5X pool. Loading this model requires ~72 GB for safetensors (mmap'd) plus ~72 GB for CUDA weight tensors — 144 GB total, exceeding the pool. Two mitigations are required:

**1. Drop page cache before each start.** After a previous run, the 72 GB model mmap stays in the kernel page cache. vLLM's startup check sees this as "used" memory (~44 GiB available vs ~110 GiB needed) and aborts. `start-vllm.sh` runs `sudo sysctl vm.drop_caches=3` before launching.

**2. 80 GB swapfile.** During model loading, both the mmap and CUDA allocations are simultaneously in memory. Without swap, the OOM killer fires mid-load. The NVMe swapfile gives the kernel room to temporarily page out file-backed mmap pages. `vm.swappiness=10` keeps CUDA allocations resident during normal inference while still allowing swap use during the loading spike.

> **If vLLM crashes with `cudaErrorIllegalInstruction`:** the CUDA UVM driver may hold ~75 GiB of pages past process exit. `nvidia-smi --gpu-reset` is unavailable on the primary GB10 GPU. The only recovery is a full reboot.

---

## API Usage

The server is OpenAI API-compatible on port 8000. Thinking mode is enabled by default — the model reasons internally before answering. **Use `max_tokens >= 4096`** to leave room for the reasoning chain (complex problems can use 5000+ thinking tokens before writing the answer).

The thinking chain is returned in the `reasoning` field:
```python
response = client.chat.completions.create(model="qwen3.5-122b", ...)
print(response.choices[0].message.reasoning)  # internal chain-of-thought
print(response.choices[0].message.content)    # final answer
```

```bash
# Direct API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b",
    "messages": [{"role": "user", "content": "Solve x^2 - 5x + 6 = 0"}],
    "max_tokens": 4096
  }'

# Disable thinking for a specific request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
  }'

# Via LiteLLM proxy (port 4000, requires API key)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b",
    "messages": [{"role": "user", "content": "Explain the P vs NP problem"}],
    "max_tokens": 4096
  }'
```

### LiteLLM Proxy

`litellm.service` runs an OpenAI-compatible proxy on port 4000 with key auth and PostgreSQL usage tracking. Two model aliases are available:

| Alias | Thinking |
|---|---|
| `qwen3.5-122b` | On (default, can disable per-request) |
| `qwen3.5-122b-no-thinking` | Off (faster for simple tasks) |

Change the API key in `litellm-config.yaml` (`master_key`) before exposing to a network.

---

## Server Flags

| Flag | Value | Why |
|---|---|---|
| `--max-model-len` | `204800` | 200K context; fits in 119 GiB with NVFP4 (~72 GB) + KV cache (~38 GB) |
| `--gpu-memory-utilization` | `0.85` | Leaves headroom for the loading spike; 0.92 causes OOM on GB10 |
| `--moe-backend` | `cutlass` | vLLM CUTLASS compiled for sm_121a. `flashinfer_cutlass` uses Hopper TMA kernels and crashes. |
| `--attention-backend` | `flashinfer` | JIT-compiles for sm_121a on first use; cached in `~/.cache/flashinfer/` |
| `--enforce-eager` | — | Disables CUDAGraph — graph warm-up spikes OOM on tight unified memory |
| `--reasoning-parser` | `qwen3` | Strips `<think>` tokens from content; exposes them in `message.reasoning` |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `undefined symbol: get_cutlass_moe_mm_data_caller` | Rebuild with Patch A and `TORCH_CUDA_ARCH_LIST="12.1a"` |
| `cudaErrorIllegalInstruction` on first inference | Check `qwen3_next.py` has both `has_device_capability(90)` and `not has_device_capability(120)` |
| `Failed to initialize cutlass TMA WS grouped gemm` | Use `--moe-backend cutlass`, not `flashinfer_cutlass` |
| `Free memory ... less than desired GPU memory utilization` | Run `sudo sysctl vm.drop_caches=3` then restart; if after a crash, reboot |
| OOM kill during model load | Check swapfile is active: `swapon -s`; check `vm.swappiness=10`: `sysctl vm.swappiness` |
| 75 GiB "used" with no processes | CUDA UVM pages stuck after illegal instruction crash — reboot required |
| `content: null` in API response | `max_tokens` too low — thinking used all tokens before writing the answer; use ≥ 4096 |
