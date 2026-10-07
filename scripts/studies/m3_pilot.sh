#!/usr/bin/env bash
# Phase 6 M3 pilot: does the M2 simulator predict one real engine's prefix-cache hits per gap bin?
# One vllm-metal replica (Qwen3-0.6B) with its KV cache fixed at KV_BLOCKS x 16 tokens so evictions happen; 120
# Copilot sessions replayed append-only with real gaps (capped at 10 min); then `memtrace report kv-validate`.
# Parameters were chosen with the simulator before the run (predicted hit share: <60 s 99%, 1-5 min 54%, >5 min 15%).
#
#   scripts/studies/m3_pilot.sh      # -> data/m3/pilot/
set -euo pipefail
# shellcheck source=../lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
OUT="${OUT_DIR:-data/m3/pilot}"
KV_BLOCKS="${KV_BLOCKS:-1024}"
SHARDS="copilot_agent/date=2026-06-06/shard-0000.jsonl.gz,copilot_agent/date=2026-06-06/shard-0001.jsonl.gz"
SHARDS="$SHARDS,copilot_agent/date=2026-06-06/shard-0002.jsonl.gz,copilot_agent/date=2026-06-06/shard-0003.jsonl.gz"
mkdir -p "$OUT" data/engine_logs

wait_gpu_free
pkill -f "vllm serve .*--port $GPU_PORT" || true
VLLM_SERVER_DEV_MODE=1 nohup ~/.venv-vllm-metal/bin/vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port "$GPU_PORT" \
  --max-model-len 4096 --enable-prefix-caching --enable-prompt-tokens-details --block-size 16 \
  --num-gpu-blocks-override "$KV_BLOCKS" > data/engine_logs/m3-pilot.log 2>&1 &
until curl -sf -m 2 "http://127.0.0.1:$GPU_PORT/health" >/dev/null; do sleep 3; done
grep -iE "kv cache|num_gpu_blocks|blocks" data/engine_logs/m3-pilot.log | tail -3 > "$OUT/engine_kv.txt" || true

.venv/bin/memtrace run engine-bench --engine vllm-metal-pilot --base-url "http://127.0.0.1:$GPU_PORT/v1" \
  --model Qwen/Qwen3-0.6B --workload copilot-agent --copilot-shards "$SHARDS" --sessions 120 --max-calls 10 \
  --gap-scale 1 --max-gap 600 --window-seconds 1200 --reuse full --max-tokens 128 --ignore-eos --idle-seconds 0 \
  --out-dir "$OUT"
pkill -f "vllm serve .*--port $GPU_PORT" || true

.venv/bin/memtrace report kv-validate --run "$(find "$OUT" -name requests.jsonl | head -1)" \
  --kv-tokens $((KV_BLOCKS * 16)) --out "$OUT/validate.json"
