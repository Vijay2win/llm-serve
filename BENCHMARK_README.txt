#!/bin/bash
# Quick reference for vLLM performance benchmarking

cat << 'EOF'
================================================================================
vLLM PERFORMANCE BENCHMARKING - QUICK REFERENCE
================================================================================

PREREQUISITES
-------------
✓ vLLM running on port 8000
✓ GPU: NVIDIA GB10 (confirmed)
✓ Python dependencies installed

SCRIPTS CREATED
---------------
1. Test Script:      /home/vijay/vllm-src/test_vllm_throughput.py
2. Benchmark Script: /home/vijay/llm-serve/benchmark_turboquant.sh
3. Results Dir:      /home/vijay/llm-serve/benchmark_results/
4. Logs Dir:         /home/vijay/llm-serve/logs/

HOW TO USE
----------

Option 1: Run individual scenarios
-----------------------------------
# Test with current vLLM (baseline FP16)
cd /home/vijay/vllm-src
python test_vllm_throughput.py --model qwen3.5-122b --scenarios short medium long

# Test specific scenarios
python test_vllm_throughput.py --scenarios short        # 512 tokens
python test_vllm_throughput.py --scenarios long         # 16K tokens
python test_vllm_throughput.py --scenarios very_long    # 32K tokens

# Save results
python test_vllm_throughput.py --output results.json


Option 2: Full benchmark with restart (recommended)
----------------------------------------------------
# Run baseline (FP16)
cd /home/vijay/llm-serve
./benchmark_turboquant.sh baseline

# Run TurboQuant 2.5 (when enabled)
./benchmark_turboquant.sh turboquant25

# Run TurboQuant 3.5 (when enabled)
./benchmark_turboquant.sh turboquant35

# Full comparison (runs all 3 configs automatically)
./benchmark_turboquant.sh compare


SCENARIOS TESTED
----------------
Short:    512 prompt tokens, 128 completion, 10 concurrent, 20 iterations
Medium:   4K prompt tokens, 256 completion, 5 concurrent, 15 iterations
Long:     16K prompt tokens, 512 completion, 3 concurrent, 10 iterations
Very Long: 32K prompt tokens, 512 completion, 2 concurrent, 8 iterations

METRICS MEASURED
----------------
✓ Success rate
✓ Average latency (ms)
✓ Median latency (ms)
✓ P95 latency (ms)
✓ Average throughput (tokens/sec)
✓ Total throughput (tokens/sec)

EXPECTED RESULTS
----------------
Baseline (FP16):
  - Short: High throughput (~100+ tok/s)
  - Long: Moderate throughput (~20-50 tok/s)
  - Very Long: Lower throughput (~10-30 tok/s)

TurboQuant (when working):
  - Similar or better throughput (more requests fit in memory)
  - 3.8-5.3x less KV cache memory
  - Same model accuracy

MONITORING
----------
During benchmark, monitor:
- GPU memory: nvidia-smi or nvtop
- vLLM logs: tail -f /home/vijay/llm-serve/logs/vllm.log
- Request logs: tail -f /home/vijay/llm-serve/logs/vllm-benchmark.log

RESULTS FORMAT
--------------
JSON output with:
- Timestamp
- Server URL
- Model name
- Per-scenario metrics
- Detailed iteration data

EXAMPLE OUTPUT
--------------
============================================================
SUMMARY
============================================================
Scenario        Prompt     Success    Latency      Throughput     
------------------------------------------------------------
Short Context   512        20/20      1250ms       45.2 tok/s
Medium Context  4096       15/15      8500ms       22.1 tok/s
Long Context    16384      10/10      35000ms      12.5 tok/s

TROUBLESHOOTING
---------------
If benchmark fails:
1. Check vLLM is running: curl http://localhost:8000/health
2. Check logs: tail -f /home/vijay/llm-serve/logs/vllm.log
3. Check GPU memory: nvidia-smi
4. Verify model name: --model qwen3.5-122b

If vLLM crashes:
1. Check for OOM errors in logs
2. Reduce concurrency in test script
3. Lower gpu-memory-utilization in start script

COMPARISON WORKFLOW
-------------------
1. Run baseline:  ./benchmark_turboquant.sh baseline
2. Run TurboQuant: ./benchmark_turboquant.sh turboquant25
3. Compare results files in benchmark_results/
4. Look for:
   - Memory usage reduction (nvidia-smi)
   - Throughput changes (same or better = good)
   - Latency changes (similar = good)

NEXT STEPS
----------
1. Run baseline benchmark now (current setup)
2. Enable TurboQuant when ready
3. Run TurboQuant benchmark
4. Compare results
5. Decide based on performance vs memory tradeoff

================================================================================
EOF
