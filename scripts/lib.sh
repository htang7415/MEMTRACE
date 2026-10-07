#!/usr/bin/env bash
# shellcheck disable=SC2034  # sourced; the variables are used by the scripts that source it
# Shared settings and helpers for the serving-platform scripts (scripts/stack.sh, scripts/studies/); sourced, not run.
#
#   MODEL_4B=1 ...      # Phase 4b: Qwen3-4B on the GPU tier, CPU simulator scaled to it (overlay qwen3-4b)
#   OVERLAY=NAME ...    # pool replicas from deploy/kind/overlays/NAME (default hetero)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CLUSTER=memtrace
CTX="kind-$CLUSTER"
ROUTER_TAG=v0.11.0
ROUTER_DIR="$HOME/.cache/memtrace/llm-d-router"
MODEL_SNAPSHOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
EPP=qwen3-0-6b-endpoint-picker
POOL_SELECTOR="app=qwen3-0-6b-inference-pool"
GATEWAY=http://localhost:30080/v1
k() { kubectl --context "$CTX" "$@"; }
# shellcheck source=images.sh
source scripts/images.sh

GPU_PORT=8210
# 0.31 = 2 GiB KV cache, as in Phase 1. (Next to a real ~5 GiB vLLM CPU pod the host paged heavily even at
# 0.2; the CPU tier is therefore simulated, see deploy/kind/base/cpu-sim.yaml.)
SERVED_MODEL=Qwen/Qwen3-0.6B
GPU_MODEL_ARGS="$SERVED_MODEL --gpu-memory-utilization 0.31"
OVERLAY="${OVERLAY:-hetero}"  # deploy/kind/overlays/<name>: the pool replicas for this mode
if [ -n "${MODEL_4B:-}" ]; then
  # Phase 4b: Qwen3-4B, MLX 4-bit, on the GPU tier; 0.35 gives a 1.37 GiB KV cache, as 0.31 gives with 0.6B.
  SERVED_MODEL=Qwen/Qwen3-4B
  GPU_MODEL_ARGS="mlx-community/Qwen3-4B-4bit --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
    --served-model-name $SERVED_MODEL --gpu-memory-utilization 0.35"
  OVERLAY=qwen3-4b  # CPU tier: the 0.6B simulator scaled to 4B
fi
GPU_ENGINE_ARGS="$GPU_MODEL_ARGS --host 127.0.0.1 --port $GPU_PORT --max-model-len 4096 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes"

served_model() { echo "$SERVED_MODEL"; }

wait_gpu_free() {
  # Another project's GPU run invalidates both measurements. Wait until no MaxionBench harness runs
  # and nothing has listened on vllm-metal's usual port 8200 for two minutes in a row.
  local quiet=0
  while [ "$quiet" -lt 120 ]; do
    if pgrep -f "[m]axionbench" >/dev/null || lsof -nP -iTCP:8200 -sTCP:LISTEN >/dev/null 2>&1; then
      quiet=0
    else
      quiet=$((quiet + 10))
    fi
    sleep 10
  done
  echo "GPU free at $(date)"
}

reset_caches() {
  curl -sf -X POST "http://127.0.0.1:$GPU_PORT/reset_prefix_cache" >/dev/null
  k rollout restart deploy/vllm-cpu-sim >/dev/null   # simulator: a restart empties its KV cache
  k rollout status deploy/vllm-cpu-sim --timeout=120s >/dev/null
}
