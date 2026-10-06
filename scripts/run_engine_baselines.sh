#!/usr/bin/env bash
# Phase 1 engine baselines: start one engine at a time, sweep both workloads, stop it.
#
#   scripts/run_engine_baselines.sh [vllm-metal|vllm-cpu|mlx-lm ...]   (default: all three)
#
# Same model, dtype (bf16), context length, and ~2 GiB KV-cache budget for every engine.
# Engines run one at a time so latency and power are not shared. Needs: ~/.venv-vllm-metal
# (vllm-metal installer), Docker with vllm/vllm-openai-cpu:latest-arm64, .venv with mlx-lm.
set -euo pipefail

NUM_REQUESTS="${NUM_REQUESTS:-100}"
WORKLOADS="${WORKLOADS:-sharegpt mooncake-toolagent}"
OUT_DIR="${OUT_DIR:-data/engine_bench}"
# shellcheck source=engines.sh
source "$(dirname "$0")/engines.sh"

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
