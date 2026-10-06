#!/usr/bin/env bash
# BFCL single-call quality gate across engines and batch sizes (ADR 0002: gate on task scores).
#
#   scripts/run_bfcl_gate.sh
#
# vllm-metal at concurrency 1 is the baseline; every other configuration is gated against it.
set -euo pipefail
source "$(dirname "$0")/engines.sh"
OUT_DIR="${OUT_DIR:-data/bfcl}"
TOLERANCE="${TOLERANCE:-0.03}"

bfcl() {  # bfcl LABEL CONCURRENCY [extra args]
  local label=$1 concurrency=$2; shift 2
  .venv/bin/memtrace evaluate bfcl --base-url "http://localhost:$PORT/v1" --model "$MODEL" \
    --concurrency "$concurrency" --out-dir "$OUT_DIR/$label" "$@" || true
}

start_engine vllm-metal
bfcl vllm-metal-c1 1
bfcl vllm-metal-c8 8 --baseline "$OUT_DIR/vllm-metal-c1/summary.json" --tolerance "$TOLERANCE"
stop_engine vllm-metal

for engine in vllm-cpu mlx-lm; do
  start_engine "$engine"
  bfcl "$engine-c8" 8 --baseline "$OUT_DIR/vllm-metal-c1/summary.json" --tolerance "$TOLERANCE"
  stop_engine "$engine"
done
