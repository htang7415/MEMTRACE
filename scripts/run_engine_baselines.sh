#!/usr/bin/env bash
# Phase 1 engine baselines: start one engine at a time, sweep both workloads, stop it.
#
#   scripts/run_engine_baselines.sh [vllm-metal|vllm-cpu|mlx-lm ...]   (default: all three)
#
# Same model, dtype (bf16), context length, and ~2 GiB KV-cache budget for every engine.
# Engines run one at a time so latency and power are not shared. Needs: ~/.venv-vllm-metal
# (vllm-metal installer), Docker with vllm/vllm-openai-cpu:latest-arm64, .venv with mlx-lm.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
WORKLOADS="${WORKLOADS:-sharegpt mooncake-toolagent}"
OUT_DIR="${OUT_DIR:-data/engine_bench}"
LOG_DIR="${LOG_DIR:-data/engine_logs}"
mkdir -p "$LOG_DIR"

start_engine() {
  case "$1" in
    vllm-metal)
      VLLM_SERVER_DEV_MODE=1 nohup ~/.venv-vllm-metal/bin/vllm serve "$MODEL" --port 8200 \
        --max-model-len 4096 --enable-prefix-caching --gpu-memory-utilization 0.31 \
        >"$LOG_DIR/vllm-metal.log" 2>&1 &
      PORT=8200 ;;
    vllm-cpu)
      docker rm -f memtrace-vllm-cpu >/dev/null 2>&1 || true
      # /dev/shm must exceed Docker's 64 MB default or the engine fails to start.
      docker run -d --name memtrace-vllm-cpu -p 8100:8000 -m 6g --shm-size 1g \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e VLLM_CPU_KVCACHE_SPACE=2 -e VLLM_SERVER_DEV_MODE=1 \
        vllm/vllm-openai-cpu:latest-arm64 --model "$MODEL" --max-model-len 4096 --enable-prefix-caching \
        >/dev/null
      PORT=8100 ;;
    mlx-lm)
      nohup .venv/bin/python -m mlx_lm.server --model "$MODEL" --port 8300 >"$LOG_DIR/mlx-lm.log" 2>&1 &
      PORT=8300 ;;
    *) echo "unknown engine: $1" >&2; exit 2 ;;
  esac
  for _ in $(seq 1 120); do
    curl -sf -m 2 "localhost:$PORT/v1/models" >/dev/null && return 0
    sleep 5
  done
  echo "$1 did not become ready" >&2
  return 1
}

stop_engine() {
  case "$1" in
    vllm-metal) pkill -f "vllm serve $MODEL --port 8200" || true ;;
    vllm-cpu) docker logs memtrace-vllm-cpu >"$LOG_DIR/vllm-cpu.log" 2>&1 || true
              docker rm -f memtrace-vllm-cpu >/dev/null 2>&1 || true ;;
    mlx-lm) pkill -f "mlx_lm.server --model $MODEL --port 8300" || true ;;
  esac
  sleep 10  # let memory and power settle before the next engine's idle baseline
}

for engine in "${@:-vllm-metal vllm-cpu mlx-lm}"; do
  for name in $engine; do
    start_engine "$name"
    trap 'stop_engine "$name"' EXIT
    for workload in $WORKLOADS; do
      .venv/bin/memtrace run engine-bench --engine "$name" --base-url "http://localhost:$PORT/v1" \
        --model "$MODEL" --workload "$workload" --num-requests "$NUM_REQUESTS" --out-dir "$OUT_DIR"
    done
    stop_engine "$name"
    trap - EXIT
  done
done
.venv/bin/memtrace report engines --root "$OUT_DIR"
