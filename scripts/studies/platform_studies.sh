#!/usr/bin/env bash
# Studies run on the kind platform (frozen: they produced the recorded Phase 2-3 results).
#
#   scripts/studies/platform_studies.sh study      # every policy x workload x concurrency level, cold start each run
#   scripts/studies/platform_studies.sh failover   # delete one replica mid-run under load; count failures, time recovery
#   scripts/studies/platform_studies.sh burst      # low -> high -> low load with autoscaling; replica timeline
#   scripts/studies/platform_studies.sh hetero_study# policies x workloads x levels on the heterogeneous pool
#   scripts/studies/platform_studies.sh hosted_study# overflow study: GPU+CPU vs +Gemini at two capacities
# shellcheck source=../stack.sh
source "$(dirname "${BASH_SOURCE[0]}")/../stack.sh"

study() {
  local out="${OUT_DIR:-data/routing_study}" replicas="${REPLICAS:-4}"
  sims "$replicas"
  for workload in ${WORKLOADS:-mooncake-toolagent mooncake-conversation}; do
    for level in ${LEVELS:-8 16 32}; do
      for name in ${POLICIES:-random queue prefix combined}; do
        policy "$name"
        .venv/bin/memtrace run engine-bench --engine "sim$replicas-$name" --base-url "$GATEWAY" \
          --model Qwen/Qwen3-0.6B --workload "$workload" --num-requests "${NUM_REQUESTS:-500}" \
          --max-tokens "${MAX_TOKENS:-128}" --sessions "${SESSIONS:-32}" --turns "${TURNS:-8}" \
          --concurrency "$level" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
          --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" --out-dir "$out"
      done
    done
  done
}

failover() {
  local out="${OUT_DIR:-data/failover}" name="${POLICY:-combined}" level="${LEVEL:-16}" delay="${KILL_AFTER:-15}"
  sims "${REPLICAS:-4}"
  policy "$name"
  .venv/bin/memtrace run engine-bench --engine "failover-${MODE:-crash}-$name" --base-url "$GATEWAY" --model Qwen/Qwen3-0.6B \
    --workload agent-sessions --sessions "${SESSIONS:-64}" --turns "${TURNS:-8}" --max-tokens 32 --concurrency "$level" \
    --ignore-eos --no-reset-prefix-cache --idle-seconds 0 --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" \
    --out-dir "$out" &
  local bench=$!
  sleep "$delay"
  local victim
  victim=$(k get pods -l "$POOL_SELECTOR" -o jsonpath='{.items[0].metadata.name}')
  # Kill only while the victim is serving, so the test exercises in-flight failure handling.
  local running=0
  for _ in $(seq 1 200); do
    running=$(k get --raw "/api/v1/namespaces/default/pods/$victim:8000/proxy/metrics" \
      | awk '/^vllm:num_requests_running/ {s += $NF} END {print int(s)}')
    [ "$running" -ge 1 ] && break
    sleep 0.05
  done
  echo "victim in-flight requests at kill: $running" | tee "$out/inflight-${MODE:-crash}.txt"
  local killed_at
  killed_at=$(python3 -c 'import time; print(time.time())')
  if [ "${MODE:-crash}" = kill ]; then
    # Process crash: SIGKILL the engine container in place; the kubelet restarts it.
    local cid
    cid=$(docker exec "$CLUSTER-control-plane" crictl ps --pod "$(docker exec "$CLUSTER-control-plane" crictl pods --name "$victim" -q)" -q)
    docker exec "$CLUSTER-control-plane" crictl stop --timeout 0 "$cid" >/dev/null
  elif [ "${MODE:-crash}" = crash ]; then
    k delete pod "$victim" --grace-period=0 --force --wait=false  # forced delete of the pod object
  else
    k delete pod "$victim" --wait=false                           # graceful: pod drains in-flight requests
  fi
  # Recovery = a full set of Ready replicas, not counting the victim or any pod being deleted.
  until [ "$(k get pods -l "$POOL_SELECTOR" -o json | python3 -c '
import json, sys
pods = json.load(sys.stdin)["items"]
kill_mode = sys.argv[2] == "kill"
def healthy(p):
    statuses = p.get("status", {}).get("containerStatuses", [])
    ready = any(c.get("ready") for c in statuses)
    if p["metadata"]["name"] == sys.argv[1]:
        return kill_mode and ready and any(c.get("restartCount", 0) > 0 for c in statuses)
    return ready and not p["metadata"].get("deletionTimestamp")
print(sum(1 for p in pods if healthy(p)))' "$victim" "${MODE:-crash}")" -ge "${REPLICAS:-4}" ]; do
    sleep 0.5
  done
  python3 -c "import time,sys; print(f'mode {sys.argv[3]}: victim {sys.argv[1]} deleted; full Ready pool again after {time.time()-float(sys.argv[2]):.1f}s')" \
    "$victim" "$killed_at" "${MODE:-crash}" | tee "$out/recovery-${MODE:-crash}.txt"
  echo "killed_at_epoch=$killed_at" >> "$out/recovery-${MODE:-crash}.txt"
  wait "$bench"
}

burst() {
  local out="${OUT_DIR:-data/autoscale}"
  mkdir -p "$out"
  policy "${POLICY:-combined}"
  if [ -n "${FIXED:-}" ]; then  # baseline: hold the replica count with KEDA's pause annotation
    k annotate scaledobject vllm-d autoscaling.keda.sh/paused-replicas="$FIXED" --overwrite
    k scale deploy/vllm-d --replicas="$FIXED"
  else
    k annotate scaledobject vllm-d autoscaling.keda.sh/paused-replicas- 2>/dev/null || true
    k scale deploy/vllm-d --replicas=1   # KEDA owns the count from here; start from the minimum
  fi
  sleep 45
  # Timeline: ready replicas and HPA desired replicas, once per second.
  ( while true; do
      printf '%s %s %s\n' "$(python3 -c 'import time; print(time.time())')" \
        "$(k get deploy vllm-d -o jsonpath='{.status.readyReplicas}')" \
        "$(k get hpa keda-hpa-vllm-d -o jsonpath='{.status.desiredReplicas}' 2>/dev/null)"  # no HPA while paused
      sleep 1
    done ) > "$out/timeline.txt" &
  local watcher=$!
  local phase=0
  for spec in ${PHASES:-4:96 24:512 4:160}; do  # concurrency:requests
    phase=$((phase + 1))
    .venv/bin/memtrace run engine-bench --engine "phase$phase" --base-url "$GATEWAY" --model Qwen/Qwen3-0.6B \
      --workload agent-sessions --sessions $(( ${spec#*:} / 8 )) --turns 8 --seed "$phase" --max-tokens 32 \
      --concurrency "${spec%%:*}" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 --out-dir "$out"
  done
  sleep "${COOLDOWN_WATCH:-120}"  # keep watching while KEDA scales back down
  kill "$watcher"
}

hetero_study() {
  local out="${OUT_DIR:-data/hetero}"
  for workload in ${WORKLOADS:-agent-sessions mooncake-toolagent}; do
    for level in ${LEVELS:-4 8}; do
      for name in ${POLICIES:-random queue combined hw-weighted-random hw-combined}; do
        policy "$name"
        reset_caches
        .venv/bin/memtrace run engine-bench --engine "hetero-$name" --base-url "$GATEWAY" --model "$SERVED_MODEL" \
          --workload "$workload" --num-requests "${NUM_REQUESTS:-128}" --sessions "${SESSIONS:-16}" --turns "${TURNS:-8}" \
          --max-tokens 32 --concurrency "$level" --ignore-eos --no-reset-prefix-cache --idle-seconds 0 \
          --k8s-context "$CTX" --k8s-pool "$POOL_SELECTOR" --out-dir "$out"
      done
    done
  done
}

hosted_study() {
  # Overflow at a load the GPU alone cannot carry: GPU + CPU, then + Gemini at two capacities.
  local root="${OUT_ROOT:-data/hosted}" level="${LEVEL:-16}"
  mkdir -p "$root"
  for setup in ${SETUPS:-local hosted2 hosted4}; do
    case "$setup" in
      local) hosted_down ;;
      hosted*) hosted "${setup#hosted}" ;;
    esac
    for rep in $(seq 1 "${REPS:-3}"); do
      local out="$root/$setup/rep$rep"
      mkdir -p "$out"
      cp "${OUT_DIR:-data/hetero}/gpu_weight.txt" "$out/"
      OUT_DIR="$out" WORKLOADS=agent-sessions LEVELS="$level" POLICIES="${POLICY:-capacity}" SESSIONS="${SESSIONS:-32}" \
        hetero_study
    done
  done
  hosted_down
}

"$@"
