#!/bin/bash
# Benchmark script to compare vLLM performance with and without TurboQuant
#
# Usage:
#   ./benchmark_turboquant.sh [baseline|turboquant25|turboquant35]
#
# This will:
# 1. Stop current vLLM
# 2. Start vLLM with specified config
# 3. Run throughput benchmarks
# 4. Save results
# 5. Show comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="$HOME/vllm-src"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Configuration
BASE_URL="http://localhost:8000"
MODEL="RedHatAI/Qwen3.5-122B-A10B-NVFP4"
SUPPORTED_MODEL="qwen3.5-122b"
API_KEY="sk-spark-llm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

stop_vllm() {
    log_info "Stopping vLLM..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
}

start_vllm() {
    local config_type=$1
    local cache_dtype=$2
    local enable_turboquant=$3
    
    log_info "Starting vLLM with $config_type..."
    
    # Build command
    CMD="$HOME/miniconda3/envs/llm/bin/python -m vllm.entrypoints.openai.api_server \
        --model \"$MODEL\" \
        --host 0.0.0.0 \
        --port 8000 \
        --served-model-name \"$SUPPORTED_MODEL\" \
        --max-model-len 1048576 \
        --gpu-memory-utilization 0.85 \
        --moe-backend cutlass \
        --attention-backend flashinfer \
        --enforce-eager \
        --reasoning-parser qwen3 \
        --default-chat-template-kwargs '{\"enable_thinking\": true}' \
        --api-key \"$API_KEY\" \
        --limit-mm-per-prompt '{\"image\": 5, \"video\": 1}' \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --trust-remote-code"
    
    # Add TurboQuant flags if enabled
    if [ "$enable_turboquant" = "true" ]; then
        CMD="$CMD \
            --cache-dtype $cache_dtype \
            --cache-config enable_turboquant=True"
        log_info "  Cache dtype: $cache_dtype"
        log_info "  TurboQuant: enabled"
    else
        log_info "  Cache dtype: auto (FP16)"
        log_info "  TurboQuant: disabled"
    fi
    
    # Start vLLM in background
    eval $CMD > "$LOGS_DIR/vllm-benchmark.log" 2>&1 &
    VLLM_PID=$!
    echo $VLLM_PID > "$SCRIPT_DIR/vllm-benchmark.pid"
    
    log_info "vLLM started (PID: $VLLM_PID)"
    
    # Wait for vLLM to be ready
    log_info "Waiting for vLLM to start..."
    for i in {1..60}; do
        if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
            log_info "vLLM is ready!"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    log_error "vLLM failed to start within 2 minutes"
    return 1
}

wait_for_vllm() {
    log_info "Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
            log_info "vLLM is ready!"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    log_error "vLLM not responding"
    return 1
}

run_benchmark() {
    local config_name=$1
    local output_file="$RESULTS_DIR/${config_name}_$(date +%Y%m%d_%H%M%S).json"
    
    log_info "Running benchmark..."
    cd "$VLLM_SRC"
    
    python test_vllm_throughput.py \
        --base-url "$BASE_URL" \
        --model "$SUPPORTED_MODEL" \
        --api-key "$API_KEY" \
        --scenarios short medium long \
        --output "$output_file" \
        --verbose
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "Benchmark completed. Results saved to: $output_file"
    else
        log_error "Benchmark failed"
    fi
    
    return $exit_code
}

print_summary() {
    local baseline_file=$1
    local turbo_file=$2
    
    if [ -f "$baseline_file" ] && [ -f "$turbo_file" ]; then
        log_section "Comparison Summary"
        echo ""
        echo "Baseline (FP16):"
        cat "$baseline_file" | python -c "
import json, sys
data = json.load(sys.stdin)
for r in data['results']:
    print(f\"  {r['name']}: {r['total_throughput_tok_per_sec']:.1f} tok/s\")
" 2>/dev/null || echo "  (failed to parse)"
        
        echo ""
        echo "TurboQuant:"
        cat "$turbo_file" | python -c "
import json, sys
data = json.load(sys.stdin)
for r in data['results']:
    print(f\"  {r['name']}: {r['total_throughput_tok_per_sec']:.1f} tok/s\")
" 2>/dev/null || echo "  (failed to parse)"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    if [ -f "$SCRIPT_DIR/vllm-benchmark.pid" ]; then
        PID=$(cat "$SCRIPT_DIR/vllm-benchmark.pid")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            log_info "Stopped vLLM (PID: $PID)"
        fi
    fi
}

# Main
main() {
    local mode=${1:-"help"}
    
    case $mode in
        "baseline")
            log_section "Running Baseline Benchmark (FP16)"
            stop_vllm
            start_vllm "baseline" "auto" "false"
            wait_for_vllm
            run_benchmark "baseline"
            cleanup
            ;;
        "turboquant25")
            log_section "Running TurboQuant 2.5 Benchmark"
            stop_vllm
            start_vllm "turboquant25" "turboquant25" "true"
            wait_for_vllm
            run_benchmark "turboquant25"
            cleanup
            ;;
        "turboquant35")
            log_section "Running TurboQuant 3.5 Benchmark"
            stop_vllm
            start_vllm "turboquant35" "turboquant35" "true"
            wait_for_vllm
            run_benchmark "turboquant35"
            cleanup
            ;;
        "compare")
            log_section "Full Comparison: Baseline vs TurboQuant"
            
            # Run baseline
            log_section "Phase 1: Baseline (FP16)"
            stop_vllm
            start_vllm "baseline" "auto" "false"
            wait_for_vllm
            run_benchmark "baseline"
            BASELINE_FILE=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -1)
            
            # Run TurboQuant 2.5
            log_section "Phase 2: TurboQuant 2.5"
            stop_vllm
            start_vllm "turboquant25" "turboquant25" "true"
            wait_for_vllm
            run_benchmark "turboquant25"
            TQ25_FILE=$(ls -t "$RESULTS_DIR"/turboquant25_*.json 2>/dev/null | head -1)
            
            # Run TurboQuant 3.5
            log_section "Phase 3: TurboQuant 3.5"
            stop_vllm
            start_vllm "turboquant35" "turboquant35" "true"
            wait_for_vllm
            run_benchmark "turboquant35"
            TQ35_FILE=$(ls -t "$RESULTS_DIR"/turboquant35_*.json 2>/dev/null | head -1)
            
            # Summary
            log_section "Final Summary"
            print_summary "$BASELINE_FILE" "$TQ25_FILE"
            
            cleanup
            ;;
        "help"|*)
            echo "vLLM TurboQuant Benchmark Script"
            echo ""
            echo "Usage: $0 [mode]"
            echo ""
            echo "Modes:"
            echo "  baseline       - Run benchmark with FP16 (default cache)"
            echo "  turboquant25   - Run benchmark with TurboQuant 2.5-bit"
            echo "  turboquant35   - Run benchmark with TurboQuant 3.5-bit"
            echo "  compare        - Run full comparison (baseline + both TQ configs)"
            echo "  help           - Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 baseline"
            echo "  $0 turboquant25"
            echo "  $0 compare"
            echo ""
            echo "Results will be saved to: $RESULTS_DIR/"
            echo "Logs will be saved to: $LOGS_DIR/"
            ;;
    esac
}

# Trap to cleanup on exit
trap cleanup EXIT

main "$@"
