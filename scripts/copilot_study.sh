#!/usr/bin/env bash
# Phase 4a: real agent traffic. Replay GitHub Copilot coding-agent sessions (open loop, real cache
# structure and timing, scaled) on the heterogeneous pool and compare routing policies.
#
#   REPS=3 scripts/copilot_study.sh      # -> data/copilot/rep<N>/hetero-<policy>/copilot-agent/open
#   MODEL_4B=1 OUT_DIR=data/copilot-4b OUT_ROOT=data/copilot-4b scripts/copilot_study.sh   # Phase 4b (Qwen3-4B; run
#                                        #   `scripts/kind_platform.sh hetero` and `calibrate` with the same env first)
set -euo pipefail
cd "$(dirname "$0")/.."
P=scripts/kind_platform.sh
$P wait_gpu_free
$P hetero
$P capacity_epp
for rep in $(seq 1 "${REPS:-3}"); do
  out="${OUT_ROOT:-data/copilot}/rep$rep"
  mkdir -p "$out"
  for name in ${POLICIES:-combined capacity cache-cost}; do
    $P policy "$name"
    $P reset_caches
    .venv/bin/memtrace run engine-bench --engine "hetero-$name" --base-url http://localhost:30080/v1 \
      --model "$($P served_model)" --workload copilot-agent --sessions "${SESSIONS:-64}" --seed "${SEED:-0}" \
      --max-tokens 128 --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
      --k8s-context kind-memtrace --k8s-pool app=qwen3-0-6b-inference-pool --out-dir "$out"
  done
done
