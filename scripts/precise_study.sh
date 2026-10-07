#!/usr/bin/env bash
# Phase 4c: precise vs approximate prefix routing on the Copilot replay (heterogeneous pool, Qwen3-0.6B).
# `precise` is `combined` with the prefix index fed by the engines' KV-cache events (deploy/kind/epp/precise.yaml);
# the CPU simulator tokenizes through vllm-render (deploy/kind/overlays/precise) so its events match.
#
#   REPS=3 scripts/precise_study.sh      # -> data/precise/rep<N>/hetero-<policy>/copilot-agent/open
set -euo pipefail
cd "$(dirname "$0")/.."
P=scripts/kind_platform.sh
export OVERLAY=precise
$P wait_gpu_free
$P render_up
$P hetero
$P capacity_epp
$P kv_events
for rep in $(seq 1 "${REPS:-3}"); do
  out="${OUT_ROOT:-data/precise}/rep$rep"
  mkdir -p "$out"
  for name in ${POLICIES:-combined precise}; do
    $P policy "$name"
    $P reset_caches       # restarts the CPU simulator; its old event connection drops the EPP subscriber,
    $P kv_events_forward  # so the GPU engine's event stream is attached last
    .venv/bin/memtrace run engine-bench --engine "hetero-$name" --base-url http://localhost:30080/v1 \
      --model "$($P served_model)" --workload copilot-agent --sessions "${SESSIONS:-64}" --seed "${SEED:-0}" \
      --max-tokens 128 --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
      --k8s-context kind-memtrace --k8s-pool app=qwen3-0-6b-inference-pool --out-dir "$out"
  done
done
pkill -f memtrace-kv-forward || true
