# shellcheck shell=bash disable=SC2034  # sourced; the variables are used by the scripts that source it
# Shared engine lifecycle for the Phase 1 scripts. Source it, then call start_engine NAME
# (sets PORT) and stop_engine NAME. Same model, dtype, context, and ~2 GiB KV budget everywhere.
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG_DIR="${LOG_DIR:-data/engine_logs}"
# Qwen3 emits Hermes-style tool calls; this only affects requests that carry `tools`.
VLLM_TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes"
mkdir -p "$LOG_DIR"
# shellcheck source=images.sh
source "$(dirname "${BASH_SOURCE[0]}")/images.sh"

start_engine() {
  case "$1" in
    vllm-metal)
      VLLM_SERVER_DEV_MODE=1 nohup ~/.venv-vllm-metal/bin/vllm serve "$MODEL" --port 8200 \
        --max-model-len 4096 --enable-prefix-caching --gpu-memory-utilization 0.31 $VLLM_TOOL_FLAGS \
        >"$LOG_DIR/vllm-metal.log" 2>&1 &
      PORT=8200 ;;
    vllm-cpu)
      pin_image "$VLLM_CPU_IMAGE" "$VLLM_CPU_DIGEST"
      docker rm -f memtrace-vllm-cpu >/dev/null 2>&1 || true
      # /dev/shm must exceed Docker's 64 MB default or the engine fails to start.
      docker run -d --name memtrace-vllm-cpu -p 8100:8000 -m 6g --shm-size 1g \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e VLLM_CPU_KVCACHE_SPACE=2 -e VLLM_SERVER_DEV_MODE=1 \
        vllm/vllm-openai-cpu:latest-arm64 --model "$MODEL" --max-model-len 4096 --enable-prefix-caching $VLLM_TOOL_FLAGS \
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

